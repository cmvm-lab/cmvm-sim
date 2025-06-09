# CMVM-sim • Reproducibility code for the Cellular Membrane Vacuum Model

This repository accompanies the **“Cellular Membrane Vacuum Model:
Foundation and First Predictions”** paper (arXiv:YYMM.NNNNN).  
It contains two stand-alone Python scripts that reproduce the numerical
results in the paper.

| Script | Purpose | Generates |
|--------|---------|-----------|
| `micro/cmvm_cell_su2_matrix_model.py` | Builds a truncated SU(2) bosonic matrix Hamiltonian, diagonalises it, and plots the ground-state convergence. | Fig. A3 |
| `echo/CMVM_echo_analysis.py` | Fits the ring-down + echo template to GW150914 data and plots the best-fit spectrum. | Fig. 4 |

## Quick start

```bash
git clone https://github.com/cmvm-lab/cmvm-sim.git
cd cmvm-sim

# optional: create a virtual environment
python -m venv .venv            # Windows: py -3 -m venv .venv
.venv/Scripts/activate          # Windows: .venv\Scripts\activate

# install minimal dependencies
pip install numpy matplotlib qutip scipy gwpy

# run the demos
python micro/cmvm_cell_su2_matrix_model.py
python echo/CMVM_echo_analysis.py
