<div id="top"></div>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![GPLv3 License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<h3 align="center">WOE Credit Scoring Toolkit</h3>

<p align="center">
A production-ready Python library for building Weight of Evidence (WoE) based credit scorecards. Provides a complete pipeline from data profiling and IV analysis to scorecard generation, model serialization, explainability, validation reports, HTML export, and an MCP server for LLM integration.
<br />
<br />
<a href="https://github.com/JGFuentesC/woe_credit_scoring/issues">Report Bug</a>
&middot;
<a href="https://github.com/JGFuentesC/woe_credit_scoring/issues">Request Feature</a>
</p>

---

## Features

- **End-to-end pipeline** — `AutoCreditScoring` handles partition, outlier treatment, normalization, feature selection, WoE transformation, logistic regression training, PDO scaling, and reporting in a single call.
- **sklearn-compatible** — All classes implement `get_params()` / `set_params()` and `__repr__()` for seamless integration with scikit-learn pipelines and grid search.
- **Serialization** — Save and load fitted models with `to_pickle()` / `from_pickle()`.
- **Isotonic calibration** — Calibrate probabilities via `predict_proba()` with optional isotonic regression.
- **Per-observation explainability** — `explain()` returns a feature-level score breakdown (WoE, beta, points) for any single client.
- **Feature analysis** — `feature_analysis()` provides IV, number of bins, monotonicity flag, type, and score contribution percentage per feature.
- **Built-in plots** — `plot_roc()`, `plot_ks()`, `plot_score_distribution()`, `plot_iv()`, `plot_discretized_features()`, `plot_woe_bins()` directly on the fitted model.
- **Validation reports** — `validation_report()` returns KS, Gini, AUC train/valid, overfitting gap, confusion matrix, optimal threshold, and cumulative gains chart.
- **MCP server** — 5 LLM-callable tools: `analyze_dataset`, `calculate_iv`, `build_scorecard`, `score_clients`, `explain_decision`.
- **HTML export** — `export_scorecard_html()` generates a self-contained report with metrics, scorecard table, and embedded plots.
- **EDA module** — `dataset_profile()`, `psi()`, `event_rate_by_feature()`, `woe_profile()`, `vif()`.
- **Pydantic v2 models** — `PipelineConfig`, `FeatureInfo`, `ScorecardResult`, `DatasetProfile` for validated configuration and structured results.

---

## Installation

```sh
pip install woe-credit-scoring
```

To enable the MCP server (LLM integration):

```sh
pip install woe-credit-scoring[mcp]
```

The MCP server is also available as a console script:

```sh
woe-mcp
```

---

## Quick Start

```python
import pandas as pd
from woe_credit_scoring import (
    AutoCreditScoring, PipelineConfig, dataset_profile,
)

# Load data
train = pd.read_csv("example_data/train.csv")

# Profile the dataset
profile = dataset_profile(train, target="TARGET")
print(f"Rows: {profile['basic_info']['rows']}, Target rate: {profile['target_distribution']['proportion'].values[1]:.2%}")

# Build and fit the model
acs = AutoCreditScoring(
    data=train,
    target="TARGET",
    continuous_features=[c for c in train.columns if c.startswith("C_")],
    discrete_features=[c for c in train.columns if c.startswith("D_")],
)

acs.fit(
    iv_feature_threshold=0.05,
    max_discretization_bins=6,
    strictly_monotonic=True,
    discretization_method="quantile",
    calibrate=True,
    verbose=True,
)

# Score new clients
valid = pd.read_csv("example_data/valid.csv")
scores = acs.predict(valid)
print(scores[["score"]].head())

# Calibrated probabilities
probas = acs.predict_proba(valid)

# Explain a single decision
explanation = acs.explain({
    "C_INCOME": 45000,
    "C_AGE": 34,
    "D_JOB": "Self",
})
print(explanation)

# Plot
acs.plot_roc()
acs.plot_ks()
acs.plot_score_distribution()
acs.plot_iv()

# Validation report
report = acs.validation_report()
print(report.summarize())

# Feature analysis
acs.feature_analysis()

# Serialize
acs.to_pickle("scorecard.pkl")

# Load later
from woe_credit_scoring import AutoCreditScoring
loaded = AutoCreditScoring.from_pickle("scorecard.pkl")
```

---

## Pipeline Classes

### DiscreteNormalizer

Normalizes discrete features by grouping infrequent categories below a relative frequency threshold into a default bucket (`"OTHER"` by default). Missing values are assigned to a `"MISSING"` category. If the default group still falls below the threshold, it is mapped to the most frequent category.

```python
from woe_credit_scoring import DiscreteNormalizer

dn = DiscreteNormalizer(
    normalization_threshold=0.05,
    default_category="OTHER",
)
dn.fit(train[discrete_features])
normalized = dn.transform(valid[discrete_features])
```

### Discretizer

Bins continuous features using `uniform`, `quantile`, `kmeans`, or `gaussian` strategies. Supports parallel execution via `n_threads`. The `gaussian` strategy uses `GaussianMixture` from scikit-learn. Missing values are handled and binned as `"MISSING"`.

```python
from woe_credit_scoring import Discretizer

disc = Discretizer(
    min_segments=2,
    max_segments=6,
    strategy="quantile",
)
disc.fit(train[continuous_features], n_threads=4)
discretized = disc.transform(valid[continuous_features])
```

### WoeEncoder

Transforms discrete/binned features into Weight of Evidence values. Supports `fit()`, `transform()`, and `inverse_transform()` (back to original categories).

```python
from woe_credit_scoring import WoeEncoder

encoder = WoeEncoder()
encoder.fit(X_discrete, y_target)
X_woe = encoder.transform(X_discrete)
X_back = encoder.inverse_transform(X_woe)
```

### `WoeBaseFeatureSelector`

Base class providing `_information_value()` and `_check_monotonic()` static methods used by both continuous and discrete feature selectors.

### WoeContinuousFeatureSelector

Selects continuous features by discretizing them with one or more strategies and ranking by IV. Supports two combination methods:

| Method | Description |
|---|---|
| `dcc` | Combination of discretizations (selects the best binning per feature across all strategies) |
| `dec` | Ensemble of discretizations (selects the best strategy per feature, preserving the method name) |

Single-strategy methods: `quantile`, `uniform`, `kmeans`, `gaussian`.

Can enforce strictly monotonic WoE behavior.

```python
from woe_credit_scoring import WoeContinuousFeatureSelector

selector = WoeContinuousFeatureSelector()
selector.fit(
    X=train[continuous_features],
    y=train["TARGET"],
    method="dcc",
    iv_threshold=0.05,
    max_bins=6,
    strictly_monotonic=True,
    n_threads=4,
)
candidates = selector.transform(valid[continuous_features])

# Inspect IV report
print(selector.iv_report)
```

### WoeDiscreteFeatureSelector

Selects discrete features based on their IV. Evaluates each normalized discrete feature and keeps those exceeding the IV threshold.

```python
from woe_credit_scoring import WoeDiscreteFeatureSelector

selector = WoeDiscreteFeatureSelector()
selector.fit(
    X=normalized_discrete,
    y=train["TARGET"],
    iv_threshold=0.05,
)
candidates = selector.transform(normalized_discrete)
print(selector.selected_features)
```

### CreditScoring

Implements the PDO (Points to Double the Odds) scoring methodology from Siddiqi (2012). Takes a fitted `WoeEncoder` and `LogisticRegression` model, and produces a scorecard mapping each feature attribute to points. Scores can be linearly scaled to a custom range.

```python
from woe_credit_scoring import CreditScoring

cs = CreditScoring(
    pdo=20,
    base_score=400,
    base_odds=1,
)
cs.fit(X_woe, woe_encoder, logistic_model)
scored = cs.transform(X_discrete)
print(cs.scorecard)
```

### AutoCreditScoring

Fully automated pipeline class. Handles the entire workflow:

1. **Partition** — train/validation split with target proportion tolerance
2. **Outlier treatment** — winsorization via `scipy.stats.mstats.winsorize`
3. **Discrete normalization** — infrequent category grouping
4. **Feature selection** — IV-based filtering for continuous and discrete features
5. **WoE transformation** — encoding into log-odds space
6. **Logistic regression** — model training with AUC tracking and overfitting detection
7. **PDO scoring** — scorecard generation with linear scaling to a target range
8. **Optional calibration** — isotonic regression for calibrated probabilities

Key methods:

| Method | Description |
|---|---|
| `fit(**kwargs)` | Train the full pipeline |
| `fit_predict(**kwargs)` | Fit and return scores for all data |
| `predict(X)` | Score new raw data (returns points per feature + total) |
| `predict_proba(X)` | Return calibrated (or raw) probabilities |
| `explain(observation)` | Per-feature score breakdown for one client |
| `feature_analysis()` | DataFrame with IV, bins, monotonicity, contribution % |
| `validation_report()` | `ValidationReport` object (KS, Gini, AUC, confusion matrix) |
| `plot_roc()` | ROC curve for train and validation |
| `plot_ks()` | KS chart on validation data |
| `plot_score_distribution()` | Score histogram (train + validation) |
| `plot_iv()` | IV barplot for selected features |
| `plot_discretized_features()` | Event rate bars with WoE annotations per binned feature |
| `plot_woe_bins(feature)` | Dual-axis plot (WoE bars + event rate line) for a single feature |
| `to_pickle(path)` / `from_pickle(path)` | Serialization |
| `save_reports(folder)` | Save PNG reports (requires `create_reporting=True`) |

### IVCalculator

Quick IV calculation without the full pipeline. Handles discretization and normalization internally.

```python
from woe_credit_scoring import IVCalculator

ivc = IVCalculator(
    data=train,
    target="TARGET",
    continuous_features=varc,
    discrete_features=vard,
)
report = ivc.calculate_iv(
    max_discretization_bins=5,
    discretization_method="quantile",
    discrete_normalization_threshold=0.05,
)
print(report)  # columns: feature, iv, feature_type
```

### frequency_table

Prints absolute frequency, relative frequency, and cumulative statistics for one or more columns.

```python
from woe_credit_scoring import frequency_table

frequency_table(train, ["D_JOB", "D_REASON"])
```

---

## Modules

### `eda.py` — Exploratory Data Analysis

```python
from woe_credit_scoring import (
    dataset_profile, psi, event_rate_by_feature, woe_profile, vif,
)
```

| Function | Description |
|---|---|
| `dataset_profile(df, target)` | Returns rows, columns, memory, missing %, target distribution, feature type counts |
| `psi(expected, actual, feature)` | Population Stability Index between two distributions |
| `event_rate_by_feature(df, target, feature)` | Event rate per category, sorted descending |
| `woe_profile(df, target, feature)` | WoE and IV contribution per category |
| `vif(X)` | Variance Inflation Factor for each numeric feature |

### `models.py` — Pydantic v2 Models

```python
from woe_credit_scoring import PipelineConfig, FeatureInfo, ScorecardResult, DatasetProfile
```

| Model | Fields |
|---|---|
| `PipelineConfig` | `iv_threshold`, `pdo`, `base_score`, `base_odds`, `min_score`, `max_score`, `discretization_method`, `max_discretization_bins`, `strictly_monotonic`, `n_threads`, `treat_outliers`, `outlier_threshold`, `target_proportion_tolerance`, `train_proportion` — with full validation |
| `FeatureInfo` | `feature`, `iv`, `feature_type`, `status` |
| `ScorecardResult` | `features`, `auc_train`, `auc_valid`, `n_features_total`, `n_features_selected`, `overfitting_warning`, `score_range`, `created_at` |
| `DatasetProfile` | `n_rows`, `n_columns`, `n_continuous`, `n_discrete`, `target_rate`, `missing_pct`, `timestamp` |

### `plots.py` — Visualization

```python
from woe_credit_scoring import (
    roc_curve_plot, roc_comparison_plot, ks_plot, iv_barplot,
    event_rate_plot, score_distribution_plot, event_rate_by_score_plot,
)
```

All functions return `matplotlib.figure.Figure` and accept an optional `ax` parameter for subplot composition. Uses a violet + blue color palette.

| Function | Description |
|---|---|
| `roc_curve_plot(y_true, y_score)` | Single ROC curve with AUC |
| `roc_comparison_plot(y_true_train, y_score_train, y_true_valid, y_score_valid)` | Train vs. validation ROC overlay |
| `ks_plot(y_true, y_score)` | Kolmogorov-Smirnov chart with decile annotation |
| `iv_barplot(iv_report, top_n=15)` | Horizontal bar chart of top N features by IV |
| `event_rate_plot(df, target, feature)` | Event rate per category with sample size |
| `score_distribution_plot(scores_train, scores_valid)` | Score histogram (train + validation) |
| `event_rate_by_score_plot(df, target)` | Stacked bar of good/bad proportions by score range |

### `validation.py` — Model Validation

```python
from woe_credit_scoring import ValidationReport
```

Lazily-computed properties: `ks`, `gini`, `auc_train`, `auc_valid`, `overfitting_gap`, `confusion_matrix`, `psi_report`.

Methods: `summarize()`, `to_dict()`, `plot_cumulative_gains()`.

### `export_html.py` — Stakeholder Reporting

```python
from woe_credit_scoring.export_html import export_scorecard_html

acs.fit(...)
export_scorecard_html(acs, "scorecard_report.html")
```

Generates a self-contained HTML file with:
- Summary metric cards (AUC train, AUC valid, KS)
- Full scorecard table
- Embedded base64 plots (ROC, KS, score distribution, IV, discretized features)

### `mcp_server.py` — LLM Integration

```sh
pip install woe-credit-scoring[mcp]
woe-mcp
```

Exposes 5 tools callable from any MCP-compatible client (Claude Desktop, Continue, etc.):

| Tool | Description |
|---|---|
| `analyze_dataset(data_path, target)` | Profile a CSV: rows, columns, target distribution, missing values, feature types |
| `calculate_iv(data_path, target, max_bins, method)` | IV calculation for all features |
| `build_scorecard(data_path, target, model_path, iv_threshold, max_bins, method, calibrate)` | Train and optionally save a scorecard |
| `score_clients(model_path, data_path)` | Score new clients using a saved model |
| `explain_decision(model_path, client_data)` | Per-feature score breakdown for one client |

---

## Dependencies

- Python >= 3.10
- numpy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- pydantic >= 2.0 (models)
- fastmcp >= 2.0 (optional, for MCP server)

---

## Contributing

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also open an issue with the tag "enhancement".

Don't forget to give the project a star!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p>

## License

Distributed under the GNU General Public License v3.0 License. See `LICENSE` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>

## Contact

Jose G Fuentes — [@jgusteacher](https://twitter.com/jgusteacher)

Project Link: [https://github.com/JGFuentesC/woe_credit_scoring](https://github.com/JGFuentesC/woe_credit_scoring)

<p align="right">(<a href="#top">back to top</a>)</p>

## Citing

If you use this software in scientific publications, we would appreciate citations to the following paper:

[Combination of Unsupervised Discretization Methods for Credit Risk](https://journals.plos.org/plosone/article/authors?id=10.1371/journal.pone.0289130) Jose G. Fuentes Cabrera, Hugo A. Perez Vicente, Sebastian Maldonado, Jonas Velasco

<p align="right">(<a href="#top">back to top</a>)</p>

## Acknowledgments

- [Siddiqi, N. (2012). Credit risk scorecards: developing and implementing intelligent credit scoring (Vol. 3). John Wiley & Sons.](https://books.google.com.mx/books?hl=es&lr=&id=SEbCeN3-kEUC&oi=fnd&pg=PT7&dq=siddiqi&ots=RvTR0RbOlQ&sig=_V4Iz1q_Hi_GwLAxrp-7tuHrOWY&redir_esc=y#v=onepage&q=siddiqi&f=false). For his amazing textbook.
- [@othneildrew](https://github.com/othneildrew/Best-README-Template). For his amazing README template.
- [Demo data](https://www.kaggle.com/code/gauravduttakiit/risk-analytics-in-banking-financial-services-1/data). For providing example data.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
[contributors-shield]: https://img.shields.io/github/contributors/JGFuentesC/woe_credit_scoring.svg?style=for-the-badge
[contributors-url]: https://github.com/JGFuentesC/woe_credit_scoring/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/JGFuentesC/woe_credit_scoring.svg?style=for-the-badge
[forks-url]: https://github.com/JGFuentesC/woe_credit_scoring/network/members
[stars-shield]: https://img.shields.io/github/stars/JGFuentesC/woe_credit_scoring.svg?style=for-the-badge
[stars-url]: https://github.com/JGFuentesC/woe_credit_scoring/stargazers
[issues-shield]: https://img.shields.io/github/issues/JGFuentesC/woe_credit_scoring.svg?style=for-the-badge
[issues-url]: https://github.com/JGFuentesC/woe_credit_scoring/issues
[license-shield]: https://img.shields.io/github/license/JGFuentesC/woe_credit_scoring.svg?style=for-the-badge
[license-url]: https://github.com/JGFuentesC/woe_credit_scoring/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/josegustavofuentescabrera
