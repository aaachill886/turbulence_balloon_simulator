from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from model import BalloonPolicyNet


STATE_COLS = [
    "vel_x", "vel_y", "vel_z",
    "wind_x", "wind_y", "wind_z",
    "intent_x", "intent_y", "intent_z", "intent_active",
    "grad_00", "grad_01", "grad_02",
    "grad_10", "grad_11", "grad_12",
    "grad_20", "grad_21", "grad_22",
]

ACTION_COLS = ["residual_x", "residual_y", "residual_z"]
ACTION_SEMANTICS = "bounded_residual_v2"

EXPECTED_STAGE3_HEADER = ["t", "episode_id", "frame_index", "run_id", "mode", "action_json", "state_json", "reward"]
ALLOWED_MODES = {"stage3", "manual", "baseline", "exploration", "policy"}


@dataclass
class NormStats:
    mean: np.ndarray
    std: np.ndarray


class PolicyDataset(Dataset):
    """Build (state, action, reward) tuples from CSV logs."""

    def __init__(self, data_dir: str, normalize: bool = True):
        self.data_dir = Path(data_dir)
        self.normalize = normalize
        self.states: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []

        self._load_dir(self.data_dir)

        if not self.states:
            raise RuntimeError(f"No usable samples found in {self.data_dir}")

        self.states = np.asarray(self.states, dtype=np.float32)
        self.actions = np.asarray(self.actions, dtype=np.float32)
        self.rewards = np.asarray(self.rewards, dtype=np.float32)

        if normalize:
            self.state_mean = self.states.mean(axis=0)
            self.state_std = self.states.std(axis=0) + 1e-7
            self.states = (self.states - self.state_mean) / self.state_std
        else:
            self.state_mean = np.zeros(self.states.shape[1], dtype=np.float32)
            self.state_std = np.ones(self.states.shape[1], dtype=np.float32)

        print(f"[PolicyDataset] samples={len(self.states)} state_dim={self.states.shape[1]}")

    def _load_dir(self, data_dir: Path):
        files = []
        files.extend(sorted(data_dir.glob("balloon_log_*.csv")))

        if not files:
            raise FileNotFoundError(f"No CSV files found in {data_dir}")

        for path in files:
            loaded_any = self._process_raw_log(path)
            if loaded_any:
                continue

            try:
                df = pd.read_csv(path)
            except Exception as exc:
                print(f"[WARN] Skip {path}: {exc}")
                continue
            if not self._is_valid_stage3_header(list(df.columns)):
                print(f"[WARN] Skip {path}: invalid CSV columns {list(df.columns)}")
                continue
            if len(df) < 2:
                continue
            self._process_episode(df)

    def _process_raw_log(self, path: Path) -> bool:
        prev_action = np.zeros(3, dtype=np.float32)
        loaded = 0
        with path.open("r", encoding="utf-8", newline="") as f:
            header = next(csv.reader(f), [])
            if not self._is_valid_stage3_header(header):
                print(f"[WARN] Skip {path}: invalid header {header}")
                return False
            for line in f:
                parsed = self._parse_stage3_line(line)
                if parsed is None:
                    continue
                try:
                    state_obj = json.loads(parsed["state_json"])
                    action_obj = json.loads(parsed["action_json"])
                    state = self._state_from_json(state_obj, prev_action)
                    action = self._action_from_obj(action_obj)
                    reward = float(parsed["reward"])
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    continue
                if self._is_sample_valid(parsed, state, action, reward):
                    self.states.append(state)
                    self.actions.append(action)
                    self.rewards.append(reward)
                    prev_action = action.astype(np.float32)
                    loaded += 1
        if loaded > 0:
            print(f"[PolicyDataset] loaded {loaded} samples from {path}")
        return loaded > 0

    def _parse_stage3_line(self, line: str) -> dict | None:
        line = line.strip()
        if not line:
            return None
        try:
            parts = next(csv.reader([line]))
            if len(parts) < 8:
                return None
            return {
                "t": parts[0],
                "episode_id": parts[1],
                "frame_index": parts[2],
                "run_id": parts[3],
                "mode": parts[4],
                "action_json": parts[5],
                "state_json": parts[6],
                "reward": parts[7],
            }
        except Exception:
            return None

    def _parse_vector_object(self, text: str) -> dict:
        vals = {}
        for key, val in re.findall(r"([xyz]):\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", text):
            vals[key] = float(val)
        return vals

    def _parse_float_after(self, text: str, key: str, default: float = 0.0) -> float:
        m = re.search(rf"{re.escape(key)}:\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", text)
        return float(m.group(1)) if m else default

    def _parse_state_body(self, text: str) -> dict:
        body = "{\"vel\":" + text
        obj = {
            "vel": self._parse_named_vector(body, "vel"),
            "wind": self._parse_named_vector(body, "wind"),
            "alt_err": self._parse_float_after(body, "alt_err"),
            "waypoint_dir": self._parse_named_vector(body, "waypoint_dir"),
            "waypoint_dist": self._parse_float_after(body, "waypoint_dist"),
            "prev_action": self._parse_named_vector(body, "prev_action"),
            "grad": self._parse_grad(body),
        }
        return obj

    def _parse_named_vector(self, text: str, name: str) -> dict:
        m = re.search(rf"{re.escape(name)}:\s*\{{([^}}]*)\}}", text)
        return self._parse_vector_object(m.group(1)) if m else {"x": 0.0, "y": 0.0, "z": 0.0}

    def _parse_grad(self, text: str) -> list[float]:
        m = re.search(r"grad:\s*\[([^\]]*)\]", text)
        if not m:
            return [0.0] * 9
        vals = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", m.group(1))]
        return (vals + [0.0] * 9)[:9]

    def _parse_env(self, text: str) -> list[float]:
        m = re.search(r"env:\s*\[([^\]]*)\]", text)
        if not m:
            return [0.0] * 4
        vals = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", m.group(1))]
        return (vals + [0.0] * 4)[:4]

    def _process_episode(self, df: pd.DataFrame):
        prev_action = np.zeros(3, dtype=np.float32)
        prev_row = None
        for idx in range(len(df)):
            row = df.iloc[idx]
            try:
                state = self._extract_state(row, prev_action)
                action = self._extract_action(row)
                reward = self._extract_reward(row, prev_row)
            except Exception:
                continue

            if self._is_sample_valid(row, state, action, reward):
                self.states.append(state)
                self.actions.append(action)
                self.rewards.append(reward)

            prev_action = action.astype(np.float32)
            prev_row = row

    def _extract_state(self, row: pd.Series, prev_action: np.ndarray) -> np.ndarray:
        state_json = row.get("state_json", "")
        if not (isinstance(state_json, str) and state_json.strip()):
            raise KeyError("Missing state_json")
        obj = json.loads(state_json)
        return self._state_from_json(obj, prev_action)

    def _extract_action(self, row: pd.Series) -> np.ndarray:
        action_json = row.get("action_json", "")
        if not (isinstance(action_json, str) and action_json.strip()):
            raise KeyError("Missing action_json")
        obj = json.loads(action_json)
        return self._action_from_obj(obj)

    def _action_from_obj(self, obj: dict) -> np.ndarray:
        if "action" not in obj:
            raise KeyError("Missing action key in action_json")
        arr = obj["action"]
        if isinstance(arr, dict):
            return np.array([float(arr.get("x", 0.0)), float(arr.get("y", 0.0)), float(arr.get("z", 0.0))], dtype=np.float32)
        if isinstance(arr, list) and len(arr) >= 3:
            return np.array(arr[:3], dtype=np.float32)
        raise ValueError("action_json.action must be dict or list of length >= 3")

    def _state_from_json(self, obj: dict, prev_action: np.ndarray) -> np.ndarray:
        semantics = str(obj.get("action_semantics", "")).strip()
        if semantics != ACTION_SEMANTICS:
            raise ValueError(f"Unsupported action semantics: {semantics!r}")
        vel = obj.get("vel", {})
        wind = obj.get("wind", {})
        intent = obj.get("intent", {})
        grad = obj.get("grad", [0.0] * 9)
        if not isinstance(grad, list):
            grad = [0.0] * 9
        grad = (grad + [0.0] * 9)[:9]
        return np.array([
            vel.get("x", 0.0), vel.get("y", 0.0), vel.get("z", 0.0),
            wind.get("x", 0.0), wind.get("y", 0.0), wind.get("z", 0.0),
            intent.get("x", 0.0), intent.get("y", 0.0), intent.get("z", 0.0),
            float(obj.get("intent_active", 0.0)),
            *grad,
        ], dtype=np.float32)

    def _extract_reward(self, row: pd.Series, prev_row: pd.Series | None) -> float:
        reward_val = row.get("reward", None)
        if reward_val is None:
            raise KeyError("Missing reward")
        if isinstance(reward_val, str) and not reward_val.strip():
            raise KeyError("Empty reward")
        return float(reward_val)

    def _is_valid_stage3_header(self, header: List[str]) -> bool:
        return [h.strip() for h in header] == EXPECTED_STAGE3_HEADER

    def _is_sample_valid(self, row_or_parsed, state: np.ndarray, action: np.ndarray, reward: float) -> bool:
        try:
            mode = str(row_or_parsed.get("mode", "")).strip().lower()
        except Exception:
            mode = ""

        if mode and mode not in ALLOWED_MODES:
            return False
        if not np.all(np.isfinite(state)):
            return False
        if not np.all(np.isfinite(action)):
            return False
        if not np.isfinite(reward):
            return False
        if np.linalg.norm(action) > 0.255:
            return False
        if state.shape[0] != len(STATE_COLS):
            return False
        return True

    def _compute_reward(self, row: pd.Series, prev_row: pd.Series | None) -> float:
        raise RuntimeError("Fallback reward computation disabled; reward must come from reward column")

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.states[idx]),
            torch.from_numpy(self.actions[idx]),
            torch.tensor(self.rewards[idx], dtype=torch.float32),
        )

    def save_norm(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        state_std = np.where(self.state_std < 1e-3, 1.0, self.state_std)
        payload = {
            "state_mean": self.state_mean.tolist(),
            "state_std": state_std.tolist(),
            "state_dim": int(self.state_mean.shape[0]),
            "action_dim": int(self.actions.shape[1]),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[PolicyDataset] saved norm → {path}")


def softplus(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x)


def export_artifacts(
    model: BalloonPolicyNet,
    dataset: PolicyDataset,
    save_path: Path,
    norm_path: Path,
    meta_path: Path,
    project_root: Path,
):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "schema_version": 1,
        "model_file": save_path.name,
        "onnx_file": save_path.with_suffix('.onnx').name,
        "norm_file": norm_path.name,
        "state_dim": int(dataset.states.shape[1]),
        "action_dim": int(dataset.actions.shape[1]),
        "state_columns": STATE_COLS,
        "action_columns": ACTION_COLS,
        "action_semantics": ACTION_SEMANTICS,
        "policy_composition": "corrected_target_velocity = player_intent_velocity + predicted_residual",
        "source_data_dir": str(dataset.data_dir),
        "export_format": "pt+onnx+json",
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    try:
        torch.onnx.export(
            model,
            torch.randn(1, dataset.states.shape[1], device=next(model.parameters()).device),
            save_path.with_suffix('.onnx'),
            input_names=['state_vector'],
            output_names=['action', 'value'],
            opset_version=13,
            do_constant_folding=True,
            dynamic_axes={'state_vector': {0: 'batch'}, 'action': {0: 'batch'}, 'value': {0: 'batch'}},
        )
        print(f"[Export] ONNX → {save_path.with_suffix('.onnx')}")
    except Exception as exc:
        raise RuntimeError(f"ONNX export failed: {exc}") from exc

    shutil.copy2(save_path, save_path.with_suffix('.latest.pt'))

    resources_dir = project_root / "Assets" / "Resources" / "Stage3"
    streaming_dir = project_root / "Assets" / "StreamingAssets" / "Stage3"
    resources_dir.mkdir(parents=True, exist_ok=True)
    streaming_dir.mkdir(parents=True, exist_ok=True)
    deployed_onnx = resources_dir / "policy_net.onnx"
    deployed_norm = streaming_dir / "policy_norm.json"
    deployed_meta = streaming_dir / "policy_meta.json"
    shutil.copy2(save_path.with_suffix('.onnx'), deployed_onnx)
    shutil.copy2(norm_path, deployed_norm)
    shutil.copy2(meta_path, deployed_meta)
    print(f"[Deploy] ONNX → {deployed_onnx}")
    print(f"[Deploy] norm → {deployed_norm}")
    print(f"[Deploy] meta → {deployed_meta}")


def train_bc(model: BalloonPolicyNet, train_loader: DataLoader, val_loader: DataLoader, device: torch.device,
             epochs: int, lr: float, out_path: Path):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_val = float("inf")
    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        n_tr = 0
        for states, actions, rewards in train_loader:
            states = states.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            logits, value = model(states)
            loss_action = F.mse_loss(logits, actions)
            loss_value = F.mse_loss(value.squeeze(-1), rewards)
            loss = loss_action + 0.2 * loss_value
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * states.size(0)
            n_tr += states.size(0)

        model.eval()
        va_loss = 0.0
        n_va = 0
        with torch.no_grad():
            for states, actions, rewards in val_loader:
                states = states.to(device)
                actions = actions.to(device)
                rewards = rewards.to(device)
                logits, value = model(states)
                loss_action = F.mse_loss(logits, actions)
                loss_value = F.mse_loss(value.squeeze(-1), rewards)
                loss = loss_action + 0.2 * loss_value
                va_loss += loss.item() * states.size(0)
                n_va += states.size(0)

        tr_loss /= max(1, n_tr)
        va_loss /= max(1, n_va)
        print(f"[BC] epoch={ep:03d} train={tr_loss:.6f} val={va_loss:.6f}")

        if va_loss < best_val:
            best_val = va_loss
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "best_val": best_val,
                "epoch": ep,
            }, out_path)
            print(f"  saved best → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="stage3 log")
    parser.add_argument("--save-dir", default="checkpoints")
    parser.add_argument("--save-name", default="policy_net.pt")
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--train-split", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = project_root / data_dir

    dataset = PolicyDataset(str(data_dir), normalize=args.normalize)
    n_total = len(dataset)
    n_train = int(n_total * args.train_split)
    n_val = max(1, n_total - n_train)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = BalloonPolicyNet(state_dim=dataset.states.shape[1], hidden_dim=256, action_dim=3).to(device)
    save_dir = Path(args.save_dir)
    if not save_dir.is_absolute():
        save_dir = script_dir / save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / args.save_name
    norm_path = save_dir / "policy_norm.json"
    meta_path = save_dir / "policy_meta.json"
    dataset.save_norm(norm_path)

    train_bc(model, train_loader, val_loader, device, args.epochs, args.lr, save_path)
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"[Export] loaded best checkpoint epoch={checkpoint.get('epoch')} val={checkpoint.get('best_val')}")
    export_artifacts(model, dataset, save_path, norm_path, meta_path, project_root)
    print(f"Done. checkpoint={save_path}")


if __name__ == "__main__":
    main()
