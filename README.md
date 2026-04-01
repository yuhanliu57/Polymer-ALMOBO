# AL-MOBO for Multifunctional Polymers

This repository contains the Python implementation of active learning-enabled multi-objective Bayesian optimization (AL-MOBO) for multifunctional polymer design.

## Repository structure

```text
al-mobo-multifunctional-polymers/
├─ LICENSE
├─ environment.yml
├─ README.md
├─ data/
├─ al_mobo/
├─ model_performance_assessment/
└─ shap_postanalysis/
```

- `al_mobo/` contains the main AL-MOBO workflow.
- `model_performance_assessment/` contains cross-validation, prospective error analysis, parity plotting, and uncertainty calibration analysis for the DKL surrogates.
- `shap_postanalysis/` contains descriptor generation, SHAP computation, and SHAP visualization.

## Environment

```bash
conda env create -f environment.yml
conda activate al-mobo-multifunctional-polymers
```

## Run the main AL-MOBO workflow

```bash
python al_mobo/run_al_mobo.py
```

Input paths and default hyperparameters are defined directly in `al_mobo/run_al_mobo.py`.

## Run model performance assessment

After the AL-MOBO run is complete, use the utilities in `model_performance_assessment/` from Python:

- `run_cross_validation()` in `cross_validation.py` for surrogate cross-validation on the initial set
- `compute_and_plot_learning_curves()` in `cross_validation.py` for prospective RMSE and NLL traces
- `compute_and_plot_uq_ence()` in `uncertainty_assessment.py` for ENCE-based uncertainty calibration analysis

Use the same model and training settings as those defined in `al_mobo/run_al_mobo.py`.

## Run SHAP post-analysis

```bash
python shap_postanalysis/compute_shap_values.py
python shap_postanalysis/plot_shap_results.py
```

Default paths and analysis settings are defined directly in the corresponding scripts.

## Notes

- When `plot_full_parity()` or `batch_predict_and_plot()` from `model_performance_assessment/parity_plots.py` is used directly with the modulus model, set `flip_sign=True`.

## Reference

If you use this repository, please cite:

*Active learning-enabled multi-objective design of thermally conductive and mechanically compliant polymers*  
Yuhan Liu, Jiaxin Xu, Renzheng Zhang, Meng Jiang, Tengfei Luo  
arXiv:2603.23494
