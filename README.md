# FYS-STK3155 — Project 2

**Author:** Eirik Bell-Mathiesen, University of Oslo  
**Contact:** eiribel@uio.no  
**Group members:** Eirik Bell-Mathiesen. The project was completed individually.

This project implements a feed-forward neural network with manual backpropagation for noisy Runge regression and 24-class AL-PUB v2 Greek-letter classification. The experiments compare optimizers, hidden-layer sizes, activation functions and weight-only L1/L2 penalties. Polynomial OLS, Ridge and Lasso provide regression comparisons. Logistic regression and a scikit-learn MLP provide classification comparisons.

The [report](report/main.pdf) contains the methods, results, discussion and AI/LLM declaration in Appendix C. Saved numerical results and model weights are included under `results/`.

## Repository contents

| Location | Contents |
| --- | --- |
| `report/main.pdf` | Project report, including Appendices A–C and references. |
| `Code/` | Nine Python files and the working notebook described below. |
| `requirements-lock.txt` | Pinned Python package versions. |
| `data/polynomial_baselines.json` | Retained Project 1 polynomial settings used for the regression comparisons. |
| `results/` | Dataset-check records, experiment settings, fold scores, predictions, curves, check receipts and weights. |
| `supplementary/` | Reproduction instructions, numerical procedure details, supplementary tables and figures. |

## Installation

Use Python 3.12. The recorded experiments used Python 3.12.7 on macOS ARM64. From the repository root:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m ipykernel install --user \
  --name fys-stk3155-project2-submission \
  --display-name "Project 2 submission"
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1` in PowerShell. Package availability on other platforms may differ from the recorded macOS environment.

The experiments used one numerical-library thread per process. Set these variables before starting Python or Jupyter on macOS/Linux:

```sh
export VECLIB_MAXIMUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

## Notebook and Python files

There is one submitted notebook. Run its regression stages in order because later stages reuse earlier data, folds and selections.

| File | Role and outputs |
| --- | --- |
| [Project2_working.ipynb](Code/Project2_working.ipynb) | Runge optimizer/learning-rate comparison, OLS comparison, activation/capacity study, and L1/L2 study against Ridge/Lasso. Writes to `results/runge_part_b_cv/`, `results/runge_part_c/` and `results/runge_part_d/`. Its final cells display the saved classification design, fold coverage, validation scores, comparisons and test scores. |
| [reused_project1.py](Code/reused_project1.py) | Shared Runge data, scaling, regression, cross-validation and optimizer routines reused from Project 1. Imported by the other code. |
| [reused_network.py](Code/reused_network.py) | Layer initialization, activations, forward passes, losses, manual gradients, training and the regression estimator. Imported by the experiment code. |
| [alpub_v2_feasibility.py](Code/alpub_v2_feasibility.py) | Checks the AL-PUB archive, exact pixel duplicates and publication groups. Writes dataset counts, the manifest and fixed holdout records to `results/alpub_v2/`. |
| [part_e_classification.py](Code/part_e_classification.py) | Classification preparation, grouped search, model comparisons and final assessment. Writes settings, folds, scores and weights to `results/classification_part_e/`, and image caches under `data/alpub_v2/processed_part_e/`. |
| [check_part_c.py](Code/check_part_c.py) | Synthetic activation, initialization and manual-regression gradient checks against JAX. Writes check records under `results/runge_part_c/`. |
| [check_part_d.py](Code/check_part_d.py) | Synthetic checks of regularized objectives, gradients, bias exclusion and zero conventions. Writes check records under `results/runge_part_d/`. |
| [check_part_e.py](Code/check_part_e.py) | Synthetic 24-class gradient, training-loop and early-stopping checks using the prepared classification design. Writes check records under `results/classification_part_e/`. |
| [present_part_e.py](Code/present_part_e.py) | Builds classification presentation tables from saved outputs, including `model_comparison.csv` and `selected_per_class.csv`. |
| [calculate_classification_baselines.py](Code/calculate_classification_baselines.py) | Calculates image-independent baselines from saved split/class counts and writes `results/classification_part_e/image_independent_baselines.json`. |

Open the notebook with:

```sh
jupyter lab Code/Project2_working.ipynb
```

Select the **Project 2 submission** kernel. Saved notebook outputs and result files can be inspected without rerunning training. Executing the full notebook repeats the regression experiments and can overwrite their generated outputs.

## Data

Runge inputs and noisy targets are generated in the notebook. The data and split seeds are 42. Network initialization and batch shuffling use seed 2026, with experiment settings saved alongside the results.

The raw AL-PUB archive and processed image caches are not included. Download [AL-PUB v2](https://www.kaggle.com/datasets/miswindall/al-pub-v2) and save the ZIP as:

```text
data/alpub_v2/ALPUB_v2.zip
```

The recorded archive SHA-256 is:

```text
1d18415ed0875e10459c40e04e2a06c8c8c2036cbc78fbc2105f2da30c95da8a
```

On macOS/Linux, check it with `shasum -a 256 data/alpub_v2/ALPUB_v2.zip`. The dataset records retain 205,776 images after excluding 21 conflicting-label pixel matches, with 159,053 development and 46,723 test images. Publication groups remain separate between these splits. Publication IDs do not establish writer identities.

## Checks and reproduction

The regression implementation checks require no external image dataset:

```sh
mkdir -p results/runge_part_c results/runge_part_d results/classification_part_e
python Code/check_part_c.py
python Code/check_part_d.py
```

The classification check also needs the prepared development image cache. After placing the AL-PUB archive at the path above:

```sh
python Code/part_e_classification.py prepare
python Code/check_part_e.py
```

For the full stage order, clean rerun instructions and numerical details, see [supplementary/REPRODUCIBILITY.md](supplementary/REPRODUCIBILITY.md). The classification stages are `prepare`, `search`, `comparisons` and `final`. Classification runs before full notebook execution because the final notebook cells read its outputs.

Use a separate copy for a full rerun and retain the supplied results as the record of the reported study. Existing classification checkpoints reuse completed fits. A full rerun is substantially more expensive than the small checks.

## Main saved results

| Folder | Selected records |
| --- | --- |
| `results/runge_part_b_cv/` | `grid.csv`, `fold_scores.csv`, `validation_picks.csv`, `test_comparison.csv`, data/fold preprocessing, training curves, predictions and weights. |
| `results/runge_part_c/` | `grid.csv`, `validation_picks.csv`, `capacity_comparisons.csv`, fold scores, checks, curves and weights. This capacity study uses development validation scores. |
| `results/runge_part_d/` | `configurations.json`, `selected_configurations.csv`, `frozen_selection.json`, `test_scores.csv`, polynomial comparisons, checks, predictions and weights. |
| `results/classification_part_e/` | Dataset memberships, shared folds, sampled configurations, search/fold scores, frozen selections and `model_comparison.csv`. `fits/` holds fold-job records. `final/` holds final weights, test probabilities, confusion matrices and per-letter scores. |

Historical verification receipts retain the source hashes from the original runs. Later comment and docstring edits do not change those receipts. The recovered numerical files were checked against their originals by SHA-256, as recorded in [supplementary/recovery_manifest.json](supplementary/recovery_manifest.json).

## Code sources and LLM use

The author specified the mathematics, algorithms, network architectures and experimental design. ChatGPT/Codex generated most implementation code, including plotting, checks, error handling, table formatting and output helpers. Earlier Project 1 and coursework code was reused or adapted, with contribution details in the source docstrings and notebook notes. Appendix C of the report gives the full declaration.
