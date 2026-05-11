# Turbulence Balloon Simulator

A Unity-based simulator for balloon/airship motion in turbulent flow, with built-in tools for control testing and training-data collection.

## Highlights

- Physically-inspired turbulence field simulation
- Manual flight and control-assist modes
- Baseline and optional ONNX predictor path
- Runtime visualization: fog, semantic field, minimap
- Automated Stage 1 data collection for ML training

## Current Status

This repository currently supports:

- turbulent wind-field generation and evolution
- assist-control testing in Unity runtime
- runtime logging for debugging and evaluation
- automated collection of spatial wind-field training data

## Stage 1: Training Data Collection

The project includes an automated data-collection pipeline for wind-prediction training.

### What it records

Each training frame stores:

- time
- balloon position
- balloon velocity
- 27-point local wind-field sample 
- local wind gradient
- active environment/config parameters

Output format:

```text
108 columns = t + pos(3) + vel(3) + wind_27×3(81) + grad(9) + config(11)
```

### Output location

Training data is written to:

```text
training_data/episode_XXXX.csv
```

### In-game controls

The top runtime panel supports:

- `Start/Stop Exploration`
- `Pause/Resume Training`
- `New Episode`
- `Clear Training Data`

## Runtime Controls

### Keyboard

- `W/A/S/D/J/K` — movement
- `Tab` — toggle basic parameter panel
- `Ctrl + I` — toggle advanced panel
- `F` — volumetric fog
- `H` — semantic field
- `M` — minimap
- `R` — reset

### Top Buttons

- `Manual`
- `Control Assist`
- `Predictor: Baseline/AI`
- `Reset`
- `Tornado`

## Main Scripts

```text
Assets/Scripts/Sim/
- TurbulenceField.cs
- ObservationBuffer.cs
- AutopilotController.cs
- GameController.cs
- DataLogger.cs
- RandomExplorationAgent.cs
- TrainingDataLogger.cs
- Bootstrap.cs

Assets/Scripts/UI/
- SimUIOnGUI.cs
```

## Project Structure

```text
Assets/             Unity assets and runtime scripts
Packages/           Unity package manifest
ProjectSettings/    Unity project settings
README.md           Project overview
```

## Optional ONNX Support

To enable the ONNX inference path:

1. Install Unity Sentis
2. Add scripting define: `ENABLE_SENTIS`
3. Assign the model asset to `ONNXPredictor`
4. Enable predictor from runtime UI

If ONNX is unavailable, the project falls back safely.

## What is intentionally not included in Git

The repository excludes generated or bulky local data such as:

- `training_data/`
- `Logs_AI/`
- Unity cache/build folders

## Next Step

Planned next phase:

- Stage 2: train a wind-prediction model from collected data and export to ONNX

## References

- Pope, *Turbulent Flows*
- Chorin & Marsden, *A Mathematical Introduction to Fluid Mechanics*
- Unity Sentis documentation: https://docs.unity3d.com/Packages/com.unity.sentis@latest
