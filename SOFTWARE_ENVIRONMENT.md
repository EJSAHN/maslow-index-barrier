# SOFTWARE_ENVIRONMENT — Reproducibility guide

This repository is designed to be **OS-independent** (Windows / macOS / Linux), provided a working Python environment is available.

The core barrier/GLM computations are deterministic.  
Optional ML benchmarking uses fixed random seeds.

---

## 1. Minimal tested requirements
- **Python**: 3.9+ (recommended: 3.11)
- **Core libraries**:
  - `numpy`
  - `pandas`
  - `scipy`
  - `statsmodels`
  - `matplotlib` (only required if running figure scripts)
  - `openpyxl` (only required if exporting Excel in optional scripts)

Optional (only if you run ML benchmark scripts):
- `scikit-learn`

---

## 2. Installation (Option A: Conda recommended)

### Create environment
```bash
conda create -n maslowbarrier python=3.11 -y
conda activate maslowbarrier
pip install -r requirements.txt
(Optional) Install ML dependencies
If you plan to run the ML benchmarking scripts:

Bash
pip install -r requirements_ml.txt
3. Installation (Option B: pip / venv)
Create virtual environment
Bash
# Windows:
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux:
python3 -m venv .venv
source .venv/bin/activate
Install dependencies
Bash
pip install -r requirements.txt
4. Reproduce numerical tables (Tables-only, Recommended)
To generate all statistical tables used in the manuscript without rendering figures (fast & deterministic):

Bash
python src/reproducible_analysis.py --data data --out outputs --write-manifest
Outputs created:

outputs/tables/*.csv (Statistical summaries and model parameters)

outputs/tables/analysis_manifest.json (Run metadata)

5. Reproduce full pipeline (includes figures)
If you want to regenerate figures in addition to tables:

Bash
python src/run_maslow_barrier.py --data data --out outputs
Note: Figure rendering (fonts, spacing) can vary slightly across operating systems and matplotlib versions. The authoritative numerical results are the CSV tables under outputs/tables/.

6. Randomness / Seeds
GLM + Barrier computations: Deterministic (no seed required).

ML benchmarking scripts (if used): Fixed random seed.

random_state = 7 (used for cross-validation splits and model initialization).

Numpy seed is also fixed when generating out-of-fold probabilities.

7. Output directories convention
outputs/tables/ — Publication-ready numerical tables (CSV)

outputs/figures/ — Figures (PNG/PDF/SVG depending on scripts)

outputs/ml/ — Optional ML benchmark outputs