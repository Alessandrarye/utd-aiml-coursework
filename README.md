# Data science development environment

[![Sync release](https://github.com/gperdrizet/datascience-devcontainer/actions/workflows/sync-release.yml/badge.svg)](https://github.com/gperdrizet/datascience-devcontainer/actions/workflows/sync-release.yml)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-latest-189fdd)](https://xgboost.readthedocs.io/)
[![Plotly](https://img.shields.io/badge/Plotly-latest-3F4F75?logo=plotly&logoColor=white)](https://plotly.com/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Docker Pulls datascience-nvidia](https://img.shields.io/docker/pulls/gperdrizet/datascience-nvidia?label=datascience-nvidia&logo=docker)](https://hub.docker.com/r/gperdrizet/datascience-nvidia)
[![Docker Pulls datascience-cpu](https://img.shields.io/docker/pulls/gperdrizet/datascience-cpu?label=datascience-cpu&logo=docker)](https://hub.docker.com/r/gperdrizet/datascience-cpu)
[![Docker Pulls datascience-mac](https://img.shields.io/docker/pulls/gperdrizet/datascience-mac?label=datascience-mac&logo=docker)](https://hub.docker.com/r/gperdrizet/datascience-mac)

A ready-to-use data science environment for VS Code, designed for intro Python and ML bootcamp students. Covers data visualization, data cleaning, feature engineering, and traditional machine learning. Available in three configurations: NVIDIA GPU, CPU-only, and Mac (Apple Silicon).

## What's included

| Package | Purpose |
|---------|---------|
| numpy, pandas, scipy | Core data science stack |
| scikit-learn, xgboost, statsmodels | Machine learning and statistics |
| matplotlib, seaborn, plotly | Visualization |
| optuna | Hyperparameter optimization |
| jupyterlab | Interactive notebooks |
| cupy-cuda12x | GPU-accelerated arrays (NVIDIA only) |

## Devcontainer configurations

| Configuration | Image | Use when |
|---------------|-------|---------|
| **DataScience NVIDIA** | `gperdrizet/datascience-nvidia` | You have an NVIDIA GPU |
| **DataScience CPU** | `gperdrizet/datascience-cpu` | CPU-only machine (any OS) |
| **DataScience Mac** | `gperdrizet/datascience-mac` | Apple Silicon Mac (M1/M2/M3) |

## Project structure

```
datascience-devcontainer/
├── .devcontainer/
│   ├── nvidia/
│   │   └── devcontainer.json   # NVIDIA GPU dev container configuration
│   ├── cpu/
│   │   └── devcontainer.json   # CPU dev container configuration
│   └── mac/
│       └── devcontainer.json   # Mac (ARM64) dev container configuration
├── data/                       # Store datasets here
├── notebooks/
│   └── environment_test.ipynb  # Verify your setup
├── .gitignore
├── LICENSE
└── README.md
```

## Requirements

- **Docker** ([Windows](https://docs.docker.com/desktop/setup/install/windows-install) | [Linux](https://docs.docker.com/desktop/setup/install/linux))
- **VS Code** with the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

### NVIDIA configuration (additional requirements)

- **NVIDIA GPU** (Pascal or newer) with driver ≥570
- **NVIDIA Container Toolkit** (Linux): [install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

### Mac configuration

- **Docker Desktop for Mac** (Apple Silicon): [install guide](https://docs.docker.com/desktop/setup/install/mac-install/)

> **Note:** GPU acceleration is not available inside Docker containers on Apple Silicon. Metal/MPS is a macOS-only framework with no Docker passthrough. The Mac configuration provides native ARM64 CPU performance.

### GPU compatibility (NVIDIA)

| Architecture | Example GPUs | Compute Capability |
|--------------|--------------|-------------------|
| Pascal | GTX 1050-1080, Tesla P100 | 6.0-6.1 |
| Volta | Tesla V100, Titan V | 7.0 |
| Turing | RTX 2060-2080, GTX 1660 | 7.5 |
| Ampere | RTX 3060-3090, A100 | 8.0-8.6 |
| Ada Lovelace | RTX 4060-4090 | 8.9 |
| Hopper | H100, H200 | 9.0 |
| Blackwell | RTX 5070-5090, B100, B200 | 10.0 |

## Quick start

1. **Fork** this repository (click "Fork" button above)

2. **Clone** your fork:
   ```bash
   git clone https://github.com/<your-username>/datascience-devcontainer.git
   ```

3. **Open VS Code**

4. **Open Folder in Container** from the VS Code command palette (`Ctrl+Shift+P`), start typing `Open Folder in`...

   > VS Code will prompt you to choose a devcontainer configuration. Select the one that matches your hardware.

5. **Verify** by running `notebooks/environment_test.ipynb`

## Using as a template for new projects

### One-time setup: Make your fork a template

1. Go to your fork on GitHub
2. Click **Settings** → scroll to **Template repository**
3. Check the box to enable it

### Creating a new project from your template

1. Go to your fork on GitHub
2. Click the green **Use this template** button → **Create a new repository**
3. Enter your new repository name and settings, click **Create repository**
4. **Clone** your new repository:
   ```bash
   git clone https://github.com/<your-username>/my-new-project.git
   ```

Now you have a fresh data science project with the dev container configuration ready to go!

## Adding Python packages

### Using pip directly

Install packages in the container terminal:

```bash
pip install <package-name>
```

> **Note:** Packages installed this way will be lost when the container is rebuilt.

### Using requirements.txt (recommended)

1. **Create** a `requirements.txt` file in the repository root:
   ```
   lightgbm
   shap
   ```

2. **Update** the appropriate `devcontainer.json` to install packages on container creation:
   ```json
   "postCreateCommand": "pip install -r requirements.txt"
   ```

3. **Rebuild** the container (`F1` → "Dev Containers: Rebuild Container")

## Keeping your fork updated

```bash
# Add upstream (once)
git remote add upstream https://github.com/gperdrizet/datascience-devcontainer.git

# Sync
git fetch upstream
git merge upstream/main
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Docker won't start | Enable virtualization in BIOS |
| Permission denied (Linux) | Add user to docker group, then log out/in |
| GPU not detected | Update NVIDIA drivers (≥570), install NVIDIA Container Toolkit |
| Container build fails | Check internet connection |
| Module not found | Rebuild container after adding to requirements.txt |
