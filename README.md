# Learning Product Demand Curves from Transaction Logs: A Two-Head Neural Network Approach

Code accompanying the master's thesis *"Learning Product Demand Curves from
Transaction Logs: A Two-Head Neural Network Approach"* by Polina Dermanovskaia.

The project estimates product-level price elasticity of demand from the public
Olist Brazilian e-commerce dataset using a two-head neural network that
guarantees monotone demand curves.

## Method

A two-head architecture separates the demand level from the price response:

```
Input features (x, z)
    |-- Context head f_ctx(x)        baseline log-demand at reference price P0
    |                                (rolling sales, calendar, freight, ...)
    |
    |-- Price head g_price(r, z)     [MONOTONE]
                                     relative price r = P / P0
                                     sum-of-sigmoids => d/dr <= 0
                                     anchored: g(1, z) = 0

Combined: log Q_hat = f_ctx(x) + g_price(r, z)
```

The price head uses a sum-of-sigmoids parameterization that guarantees a
monotonically decreasing demand curve by construction, anchored so that the
price head contributes zero at the reference price P0.

Models are evaluated with **corrected MSE (corr_mse)**: the per-product MSE
after an optimal level shift, which measures demand-curve *shape* accuracy
rather than absolute level.

## Dataset

Olist Brazilian e-commerce data, aggregated to the product-week level
(2016-2018). After filtering (min 10 items, 2 distinct prices, 5 weeks per
product): **1,218 products, 17,970 product-week observations**.

The raw data is not redistributed here. See [data/README.md](data/README.md)
for the download link and where to place the files.

## Repository layout

```
README.md
LICENSE
requirements.txt
data/README.md         how to obtain the Olist data
notebooks/             data processing, model training, and analysis
scripts/               reproducible experiment and plotting scripts
results/               tables, figures, and the trained model checkpoint
```

### Notebooks

| Notebook | Purpose |
|----------|---------|
| `Olist_EDA.ipynb` | Exploratory data analysis |
| `data_pipeline_final.ipynb` | Data processing (run first) |
| `data_pipeline_with_zeros.ipynb` | Alternative pipeline retaining zero-sales weeks |
| `baselines_day2.ipynb` | Log-log regression and XGBoost baselines |
| `nn_improved.ipynb` | Two-head neural network |
| `elasticity_diagnostics_executed.ipynb` | Monotonicity, stability, endogeneity checks |
| `reviews_elasticity_analysis.ipynb` | Heterogeneous elasticity by reviews |
| `ablation_freight.ipynb` | Ablation: freight-cost feature |
| `hurdle_model.ipynb` | Hurdle model experiment (zero-inflated demand) |
| `robustness_experiments.ipynb` | Robustness checks (hyperparameters, seeds, k) |

### Scripts

| Script | Purpose |
|--------|---------|
| `scripts/rerun_ablations.py` | Re-run the freight and reviews ablations |
| `scripts/run_bootstrap.py` | Clustered bootstrap confidence intervals |
| `scripts/generate_demand_curves_quartiles.py` | Demand curves by review-count quartile |
| `scripts/plot_sum_of_sigmoids.py` | Illustration of the sum-of-sigmoids price head |

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.10+ and PyTorch 2.9.

## Reproduction

1. Download the Olist dataset and place it under `data/olist_datasets/` as
   described in [data/README.md](data/README.md).
2. Run `notebooks/data_pipeline_final.ipynb` to build the product-week panel
   in `data/processed/`. This is the single source of truth for data
   processing (filters, feature engineering, relative price `r = P/P0`, and
   the time-based train/val/test split).
3. Run `notebooks/baselines_day2.ipynb` for the log-log and XGBoost baselines.
4. Run `notebooks/nn_improved.ipynb` to train the two-head model. The
   checkpoint is also provided at `results/improved_nn_model.pt`.
5. Run the analysis notebooks and scripts as needed:
   - `notebooks/elasticity_diagnostics_executed.ipynb`
   - `notebooks/reviews_elasticity_analysis.ipynb`
   - `python scripts/run_bootstrap.py`
   - `python scripts/rerun_ablations.py`

Scripts read the panel from `data/processed/` and write tables and figures to
`results/`.

> **Note on reproducibility:** training on Apple MPS is non-deterministic, so
> reported metrics may vary by roughly +/-2% across runs even with a fixed
> seed. The thesis reports the multi-seed mean (test corr_mse approximately
> 0.10).

## Citation

If you use this code, please cite the thesis:

> Dermanovskaia, P. *Learning Product Demand Curves from Transaction Logs:
> A Two-Head Neural Network Approach.* Master's thesis.

## License

MIT, see [LICENSE](LICENSE). The Olist dataset is distributed under its own
license by its original authors and is not included in this repository.
