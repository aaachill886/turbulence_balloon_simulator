# Turbulence Balloon Simulator

A Unity simulator for balloon/airship control in turbulent flow, with automated training data collection for AI-based wind prediction.

**Latest Update**: Stage 1 Training Data Collection System ✨

[![Unity](https://img.shields.io/badge/Unity-2021.3+-black.svg)](https://unity.com/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🎯 Project Overview

This simulator focuses on:
- **Physically-motivated turbulence** - Von Kármán spectral initialization with advection-diffusion evolution
- **Baseline vs AI prediction** - Compare traditional and neural network-based wind prediction
- **Operator-assist control** - Help pilots navigate turbulent conditions
- **Automated data collection** - Generate training data for machine learning models
- **Reproducible logging** - Complete trajectory and state recording

---

## 🚀 Quick Start

### For Data Collection (Stage 1)

```bash
1. Open Unity project (Unity 2021.3+)
2. Click Play
3. Click "Start Exploration" button in top panel
4. Wait for data collection (~1 hour for 50,000 frames)
5. Data saved to: training_data/episode_XXXX.csv
```

### For Manual Flight

```bash
1. Open Unity project
2. Click Play
3. Use W/A/S/D/J/K to fly
4. Press Tab for parameters panel
5. Toggle "Control Assist" for AI help
```

---

## 📊 Stage 1: Training Data Collection (NEW)

### Features

- 🤖 **RandomExplorationAgent** - Autonomous flight with random exploration
- 📊 **TrainingDataLogger** - 27-point spatial wind field sampling (3×3×3 grid)
- 🔄 **Auto Condition Sweep** - Automatic parameter variation (Beaufort, Gust, Tornado, etc.)
- 💾 **Smart Episode Management** - Auto-resume, no data overwrite
- 🎮 **UI Controls** - Pause/Resume/Clear training data from in-game panel

### Data Format

**108 columns per frame**:
```
t(1) + pos(3) + vel(3) + wind_27×3(81) + grad(9) + config(11) = 108 columns
```

- **Sampling rate**: 12.5 Hz (0.08s interval)
- **Spatial sampling**: 27 points in 3×3×3 grid (0.5m spacing)
- **Output**: CSV files in `training_data/episode_XXXX.csv`

### UI Controls

**Training Data Panel**:
- `Start/Stop Exploration` - Toggle autonomous data collection
- `Pause/Resume Training` - Pause without losing context
- `New Episode` - Manually start new episode file
- `Clear Training Data` - Delete all collected data (with confirmation)

**Status Display**:
```
Episode: 42 | Frames: 59,450 | recording
```

### Documentation

- 📖 [Quick Start Guide](QUICKSTART.md) - 5-minute setup
- 📚 [Detailed Usage](STAGE1_DATA_COLLECTION.md) - Complete guide
- 📝 [Implementation Summary](STAGE1_SUMMARY.md) - Technical details
- 🐛 [Troubleshooting](TROUBLESHOOTING.md) - Common issues
- 📋 [Changelog](CHANGELOG.md) - Version history

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

## 📦 Project Status

### ✅ Completed Features

**Core Simulation**:
- ✅ Turbulence field generation + evolution loop
- ✅ Von Kármán spectral initialization
- ✅ Advection-diffusion with nonlinear coupling
- ✅ Gust / convection / wake / tornado effects

**Control Systems**:
- ✅ Manual and Control Assist modes
- ✅ Baseline predictor integration
- ✅ ONNX predictor integration path (Unity Sentis)
- ✅ History-based hold implementation
- ✅ Observable attitude recovery loop

**Data Collection & Logging**:
- ✅ **Stage 1: Automated training data collection** ✨
- ✅ RandomExplorationAgent with auto condition sweep
- ✅ TrainingDataLogger with 27-point spatial sampling
- ✅ Smart episode management (auto-resume, no overwrite)
- ✅ Runtime logging (Logs_AI/) with pause/resume controls

**Visualization & UI**:
- ✅ Runtime semantic guidance + key suggestion
- ✅ Volumetric fog visualization
- ✅ Flow heatmap renderer
- ✅ Minimap with block world
- ✅ Comprehensive parameter tuning panels

### 🚧 Roadmap

**Stage 2: Model Training** (Next):
- [ ] PyTorch data loader for training_data/
- [ ] SpatioTemporalWindPredictor implementation
- [ ] Training loop with validation
- [ ] ONNX export

**Stage 3: Unity Integration**:
- [ ] Load trained ONNX model
- [ ] Real-time inference in Unity
- [ ] Six-axis thrust allocation
- [ ] Performance optimization

**Stage 4: Online RL** (Optional):
- [ ] PPO implementation
- [ ] Online fine-tuning
- [ ] Continuous adaptation

---

## 9) Research context

This simulator is used for controlled comparison of:
- manual vs assist behavior,
- baseline vs AI prediction,
- operator guidance effectiveness,
under turbulent conditions with reproducible logs.

---

## 🛠️ Installation

### Prerequisites

- **Unity**: 2021.3 or later
- **Python**: 3.8+ (for data validation and training)
- **Git**: For version control

### Setup Steps

1. **Clone the repository**:
```bash
git clone https://github.com/aaachill886/turbulence_balloon_simulator.git
cd turbulence_balloon_simulator
```

2. **Open in Unity**:
   - Open Unity Hub
   - Click "Add" → Select project folder
   - Open with Unity 2021.3+

3. **Install Python dependencies** (optional, for data validation):
```bash
pip install pandas numpy matplotlib
```

4. **Play**:
   - Click Play button in Unity
   - Start exploring or collecting data!

---

## 📖 Usage Examples

### Example 1: Collect Training Data

```bash
1. Unity: Click Play
2. Click "Start Exploration"
3. Wait for 50,000 frames (~1 hour)
4. Click "Stop Exploration"
5. Validate: python validate_training_data.py
```

### Example 2: Manual Flight with AI Assist

```bash
1. Unity: Click Play
2. Click "Control Assist" button
3. Use W/A/S/D/J/K to fly
4. AI helps stabilize and predict wind
5. Press H to see semantic field guidance
```

### Example 3: Test Different Turbulence Conditions

```bash
1. Unity: Click Play
2. Press Tab to open parameters panel
3. Adjust Beaufort, Gust, Convection sliders
4. Click "Tornado" for extreme conditions
5. Observe balloon behavior
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Workflow

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code Style

- **C#**: Follow Unity C# coding conventions
- **Python**: Follow PEP 8
- **Comments**: Use clear, concise comments
- **Documentation**: Update README for new features

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Turbulence Theory**: Pope's *Turbulent Flows*
- **Fluid Mechanics**: Chorin & Marsden's *A Mathematical Introduction to Fluid Mechanics*
- **Unity Sentis**: [Unity Sentis Documentation](https://docs.unity3d.com/Packages/com.unity.sentis@latest)
- **Community**: Thanks to all contributors and testers!

---

## 📞 Contact

- **GitHub**: [@aaachill886](https://github.com/aaachill886)
- **Issues**: [Report bugs or request features](https://github.com/aaachill886/turbulence_balloon_simulator/issues)

---

## 📊 Project Statistics

- **Lines of Code**: ~15,000+ (C#)
- **Training Data Format**: 108 columns, 12.5 Hz sampling
- **Spatial Sampling**: 27 points (3×3×3 grid)
- **Supported Unity Version**: 2021.3+

---

## 10) References

- Pope, *Turbulent Flows*
- Chorin & Marsden, *A Mathematical Introduction to Fluid Mechanics*
- Unity Sentis documentation:
  https://docs.unity3d.com/Packages/com.unity.sentis@latest

---

**⭐ If you find this project useful, please consider giving it a star!**
