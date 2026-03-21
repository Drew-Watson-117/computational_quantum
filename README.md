# Computational Quantum Mechanics Using PINNs

This repository is an investigation of using Physics Informed Neural Networks (PINNs) to solve the Schrodinger equation numerically to obtain numerical solutions to quantum mechanics problems.
The `psi_net/` directory is the Python package that contains the logic for using PINNs to solve the Schrodinger equation.
The package currently supports only Cartesian coordinates, and is not yet in a working state.
Outside the `psi_net/` directory are several Jupyter notebooks responsible for testing the `psi_net` package.

## Roadmap

- [ ] Improve performance of the DE loss function by using a more efficient implementation (`schrodinger.Schrodinger.de_loss`).
- [ ] Find hyperparameters that will give correct results for the infinite square well problem.
- [ ] Add support for other coordinate systems (spherical, cylindrical, etc.).
- [ ] Improve the training pipeline to give more consistent results with fewer epochs and better learning-rate behavior.

## Running on a Remote GPU with Coiled

If you don't have a local GPU, you can run notebooks on a cloud GPU using [Coiled](https://www.coiled.io/).

### Prerequisites

- A Coiled account (free tier: 500 CPU-hours/month)
- An AWS account connected to Coiled with GPU quota enabled
  - In AWS Console: **Service Quotas** > **Amazon EC2** > **Running On-Demand G and VT instances** — request at least 4 vCPUs

### Setup

```bash
pip install "coiled[notebook]"
coiled login
```

### Launch a GPU notebook

From the project root:

```bash
coiled notebook start --gpu --sync
```

This provisions a `g4dn.xlarge` (Tesla T4) instance and syncs your local files to the remote machine. JupyterLab opens in your browser.

### Install `psi_net` on the remote

In a JupyterLab cell or terminal:

```python
import sys
!{sys.executable} -m pip install -e /scratch/synced
```

Restart the kernel, then run your notebooks. CUDA is detected automatically.

### Run tests on the remote

Open a terminal in JupyterLab (**File** > **New** > **Terminal**):

```bash
cd /scratch/synced
python -m pip install -e ".[dev]"
python -m pytest tests/
```