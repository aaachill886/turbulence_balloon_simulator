# Turbulence Balloon Simulator

A Unity simulator for balloon/airship control in turbulent flow, focused on:
- physically-motivated turbulence,
- baseline vs AI prediction comparison,
- operator-assist control under uncertainty,
- reproducible trajectory/state logging for training and evaluation.

---

## 1) What this project does now

This repository currently supports a full runtime loop:

1. **Wind field generation and evolution**
   - Von Kármán-style spectral initialization
   - Advection-diffusion + nonlinear coupling evolution
   - Gust / convection / wake / tornado toggles

2. **Control and prediction separation**
   - Control mode:
     - `Manual`
     - `Control Assist`
   - Predictor source:
     - `Baseline`
     - `AI (ONNX via Sentis, optional)`

3. **History-based hold behavior (default non-cheating path)**
   - On input release, controller stores:
     - release position
     - release forward direction
     - release velocity
   - Hold uses release-time history + current vehicle state to recover.
   - Runtime HUD explicitly shows:
     - `holdMode`
     - `oracleEnv`

4. **Observable attitude recovery**
   - Vehicle state includes yaw and forward vector.
   - Attitude is updated in runtime loop and visible in HUD.

5. **Operator-facing flow guidance**
   - Semantic visualization field
   - Real-time low-resistance key suggestion (`W/A/S/D/J/K`)

6. **Data logging pipeline controls in HUD**
   - Pause/Resume logging
   - Open log folder
   - Clear logs
   - CSV files written to project-root `Logs_AI/`

---

## 2) Current runtime architecture

```text
TurbulenceField
  -> ObservationBuffer
  -> Predictor (Baseline or ONNX)
  -> AutopilotController (Manual/Assist + Hold)
  -> GameController (state integration + attitude update)
  -> DataLogger (CSV)
  -> Runtime UI / Visual layers
```

---

## 3) Current key scripts

```text
Assets/Scripts/Sim/
- TurbulenceField.cs            # wind generation + evolution
- ObservationBuffer.cs          # history buffer
- ONNXPredictor.cs              # Sentis ONNX inference path
- AutopilotController.cs        # control assist + hold logic
- BalloonState.cs               # velocity + observable attitude state
- BalloonThermodynamics.cs      # buoyancy / gas state model
- GameController.cs             # simulation loop integration
- FlowHeatmapRenderer.cs        # semantic guidance field
- VolumetricFogSimple.cs        # volumetric flow cues
- DataLogger.cs                 # CSV logger + runtime controls
- Bootstrap.cs                  # scene bootstrap

Assets/Scripts/UI/
- SimUIOnGUI.cs                 # runtime HUD + tuning panels
```

---

## 4) Controls and runtime UI

### Keyboard
- `W/A/S/D/J/K`: movement inputs
- `Tab`: toggle basic panel
- `Ctrl + I`: toggle advanced panel
- `F`: volumetric fog
- `H`: semantic field
- `M`: minimap
- `R`: reset

### Top HUD buttons
- `Manual`
- `Control Assist`
- `Predictor: Baseline/AI`
- `Reset`
- `Tornado`

### Logger buttons
- `Pause Log` / `Resume Log`
- `Open Log Folder`
- `Clear Logs`

---

## 5) Logging

Logs are stored at:

```text
<project-root>/Logs_AI/balloon_log_YYYYMMDD_HHMMSS.csv
```

Logged fields include:
- position/velocity
- observed wind and predicted wind
- prediction error and sigma
- local gradient features
- block/waypoint context
- active configuration values

---

## 6) Default operational intent

At startup/reset, this project is intended to run in assist-focused baseline testing:
- control assist available
- baseline predictor enabled by default path
- history-based hold path visible in HUD

For fair experiments, prefer:
- `holdMode: realistic-history`
- `oracleEnv: off`

Use benchmark/oracle toggles only for explicit ablation tests.

---

## 7) ONNX (optional)

To enable ONNX predictor:
1. Install Unity Sentis package.
2. Add scripting define: `ENABLE_SENTIS`.
3. Assign model asset in `ONNXPredictor`.
4. Enable predictor from runtime UI.

If ONNX is unavailable/misaligned, runtime falls back safely.

---

## 8) Project status

### Completed
- Turbulence field generation + evolution loop
- Baseline predictor integration
- ONNX predictor integration path (Sentis)
- Control/predictor mode decoupling
- History-based hold implementation
- Observable attitude recovery loop
- Runtime semantic guidance + key suggestion
- Root-folder logging workflow and controls


---

## 9) Research context

This simulator is used for controlled comparison of:
- manual vs assist behavior,
- baseline vs AI prediction,
- operator guidance effectiveness,
under turbulent conditions with reproducible logs.

---

## 10) References

- Pope, *Turbulent Flows*
- Chorin & Marsden, *A Mathematical Introduction to Fluid Mechanics*
- Unity Sentis documentation:
  https://docs.unity3d.com/Packages/com.unity.sentis@latest
