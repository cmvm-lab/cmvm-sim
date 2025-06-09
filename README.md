# cmvm-sim  •  Reproducibility code for the Cellular Membrane Vacuum Model
cmvm simulations
This repository accompanies the **“Cellular Membrane Vacuum Model: Foundation
and First Predictions”** paper (arXiv:YYMM.NNNNN).  
It contains two stand-alone Python scripts that reproduce every
numerical figure and table in the paper.

| Script | Purpose | Generates |
|--------|---------|-----------|
| `micro/cmvm_cell_su2_matrix_model.py` | Builds a truncated SU(2) bosonic matrix Hamiltonian, diagonalises it, and plots the ground-state energy convergence. | Fig. A3 (Appendix) |
| `echo/CMVM_echo_analysis.py` | Fits the ring-down + echo template to GW150914 strain data and plots the best-fit spectrum. | Fig. 4 (Sec. VI A) |

## Quick start
Each script writes figures to ./figures/ and prints key numbers to the console.

Dependencies
Python ≥ 3.9

numpy, matplotlib (both scripts)

qutip (micro-cell)

gwpy, scipy (echo analysis)

```bash
# clone repo (HTTPS)
git clone https://github.com/cmvm-lab/cmvm-sim.git
cd cmvm-sim

# create environment (optional but recommended)
python -m venv .venv
source .venv/Scripts/activate   # Windows
pip install -r requirements.txt  # numpy, matplotlib, qutip, gwpy …

# 1. micro-cell demo
python micro/cmvm_cell_su2_matrix_model.py

# 2. GW echo analysis (needs ∼50 MB GW150914 frame file; auto-downloads if absent)
python echo/CMVM_echo_analysis.py

