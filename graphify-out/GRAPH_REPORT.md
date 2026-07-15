# Graph Report - .  (2026-07-15)

## Corpus Check
- 0 files · ~0 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 278 nodes · 440 edges · 29 communities (13 shown, 16 thin omitted)
- Extraction: 85% EXTRACTED · 15% INFERRED · 0% AMBIGUOUS · INFERRED: 67 edges (avg confidence: 0.74)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- AutoCreditScoring Core
- CreditScoring & Manual Pipeline
- Encoder Tests
- Normalizer Tests
- Discretizer Tests
- AutoCreditScoring Pipeline + Tests
- Conftest Fixtures
- IV Calculator Tests
- EDA Module
- Pydantic Models
- Base Selector Tests
- Docs & Images
- Component Docs
- Reporter Module
- Changelog
- Report Doc
- PSI Function
- VIF Function
- Package Metadata
- Dependencies
- Community 20
- Community 21
- Community 22
- Community 23
- Community 24
- Community 25
- Community 26
- Community 27
- Community 28

## God Nodes (most connected - your core abstractions)
1. `WoeEncoder` - 31 edges
2. `Discretizer` - 30 edges
3. `DiscreteNormalizer` - 27 edges
4. `AutoCreditScoring` - 24 edges
5. `CreditScoring` - 20 edges
6. `WoeContinuousFeatureSelector` - 18 edges
7. `WoeDiscreteFeatureSelector` - 18 edges
8. `IVCalculator` - 17 edges
9. `WoeBaseFeatureSelector` - 12 edges
10. `AutoCreditScoring` - 6 edges

## Surprising Connections (you probably didn't know these)
- `AutoCreditScoring` --references--> `Feature Importance Chart`  [EXTRACTED]
  README.md → images/feature_importance.png
- `AutoCreditScoring` --references--> `ROC AUC Curve`  [EXTRACTED]
  README.md → images/roc_auc_curve.png
- `AutoCreditScoring` --references--> `Score Histogram`  [EXTRACTED]
  README.md → images/score_histogram.png
- `AutoCreditScoring` --references--> `Score KDE Plot`  [EXTRACTED]
  README.md → images/score_kde.png
- `fitted_encoder()` --calls--> `WoeEncoder`  [INFERRED]
  tests/conftest.py → woe_credit_scoring/encoder.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Core Scoring Pipeline Flow** — autocreditscoring_py, feature_selectors_py, normalizer_py, encoder_py, scoring_py [EXTRACTED 1.00]
- **Exploratory Data Analysis Tools** — eda_dataset_profile, eda_psi, eda_vif [EXTRACTED 1.00]
- **WoE Feature Selection Components** — readme_woebasefeatureselector, readme_woecontinuousfeatureselector, readme_woediscretefeatureselector [EXTRACTED 1.00]
- **Credit Scoring Pipeline** — readme_discretenormalizer, readme_discretizer, readme_woeencoder, readme_creditscoring [EXTRACTED 0.90]

## Communities (29 total, 16 thin omitted)

### Community 0 - "AutoCreditScoring Core"
Cohesion: 0.12
Nodes (8): AutoCreditScoring.fit_predict, Base class for selecting features based on their Weight of Evidence (WoE)     tr, WoeBaseFeatureSelector, DataFrame, Series, Selects continuous features based on Weight of Evidence (WoE) and     Informatio, WoeContinuousFeatureSelector, WoeDiscreteFeatureSelector

### Community 1 - "CreditScoring & Manual Pipeline"
Cohesion: 0.08
Nodes (17): LogisticRegression, test_manual_pipeline_auc(), fitted_scoring(), Fixture with 2 features, 2 categories each, every cat has both classes., Fit encoder, logistic regression, and CreditScoring; return cs and data., scoring_data(), test_base_odds_zero_sets_offset_to_infinity(), test_custom_pdo_base_score_base_odds() (+9 more)

### Community 2 - "Encoder Tests"
Cohesion: 0.10
Nodes (17): test_encoder_raises_on_unfitted_inverse_transform(), test_encoder_raises_on_unfitted_transform(), test_encoder_raises_value_error_on_non_binary_target(), test_encoder_raises_value_error_on_single_value_target(), test_encoder_works_with_non_01_target(), test_woe_calculation(), test_woe_inverse_transform(), test_woe_transform() (+9 more)

### Community 3 - "Normalizer Tests"
Cohesion: 0.10
Nodes (17): test_normalizer_empty_dataframe_no_columns(), test_normalizer_raises_type_error_on_non_dataframe(), test_normalizer_threshold_over_one_everything_grouped(), test_normalizer_threshold_zero_nothing_grouped(), test_missing_value_handling(), test_no_small_categories(), test_small_category_aggregation(), test_unseen_categories() (+9 more)

### Community 4 - "Discretizer Tests"
Cohesion: 0.13
Nodes (16): test_all_nan_column(), test_empty_dataframe_returns_empty(), test_fit_gaussian_strategy(), test_fit_kmeans_strategy(), test_fit_quantile_strategy(), test_fit_uniform_strategy(), test_single_unique_value_column(), test_transform_column_naming_convention() (+8 more)

### Community 5 - "AutoCreditScoring Pipeline + Tests"
Cohesion: 0.17
Nodes (6): test_autocreditscoring_pipeline(), AutoCreditScoring, DataFrame, A class used to perform automated credit scoring using logistic regression and W, Fits the model and returns the scores for the entire dataset.          Args:, Predicts scores for a given raw dataset.          The input data should have the

### Community 7 - "IV Calculator Tests"
Cohesion: 0.15
Nodes (8): test_iv_calculator_both(), test_iv_calculator_continuous_only(), test_iv_calculator_discrete_only(), test_iv_calculator_type_error_not_dataframe(), test_iv_calculator_value_error_target_not_in_columns(), IVCalculator, DataFrame, Calculates the Information Value (IV) for both discrete and continuous features.

### Community 8 - "EDA Module"
Cohesion: 0.19
Nodes (13): dataset_profile, dataset_profile(), event_rate_by_feature(), psi(), DataFrame, Series, Computes the event rate for each category of a feature.      Args:         df (p, Calculates Weight of Evidence (WoE) and Information Value (IV) per category. (+5 more)

### Community 9 - "Pydantic Models"
Cohesion: 0.21
Nodes (10): BaseModel, DatasetProfile, FeatureInfo, PipelineConfig, Pydantic v2 configuration and result models for the WOE credit scoring library., Summary information for a single feature after WOE scoring., Complete result of a scorecard build, including performance metrics., Descriptive profile of the dataset used to build the scorecard. (+2 more)

### Community 10 - "Base Selector Tests"
Cohesion: 0.24
Nodes (9): test_check_monotonic_excludes_missing(), test_check_monotonic_returns_false(), test_check_monotonic_returns_true(), test_information_value_matches_expected(), test_information_value_none_for_perfect_separation(), test_information_value_single_category(), Series, Computes information value (IV) statistic.          Args:             X (pd.Seri (+1 more)

### Community 11 - "Docs & Images"
Cohesion: 0.29
Nodes (7): Feature Importance Chart, ROC AUC Curve, Score Histogram, Score KDE Plot, AutoCreditScoring, CreditScoring, IVCalculator

### Community 12 - "Component Docs"
Cohesion: 0.29
Nodes (7): Scoring Method Workflow Diagram, DiscreteNormalizer, Discretizer, WoeBaseFeatureSelector, WoeContinuousFeatureSelector, WoeDiscreteFeatureSelector, WoeEncoder

### Community 13 - "Reporter Module"
Cohesion: 0.50
Nodes (3): frequency_table(), DataFrame, Displays a frequency table for the specified variables in the DataFrame.      Ar

## Knowledge Gaps
- **26 isolated node(s):** `woe_credit_scoring`, `DiscreteNormalizer`, `Discretizer`, `WoeEncoder`, `WoeContinuousFeatureSelector` (+21 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **16 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `WoeEncoder` connect `Encoder Tests` to `AutoCreditScoring Core`, `CreditScoring & Manual Pipeline`, `AutoCreditScoring Pipeline + Tests`, `Conftest Fixtures`?**
  _High betweenness centrality (0.191) - this node is a cross-community bridge._
- **Why does `DiscreteNormalizer` connect `Normalizer Tests` to `AutoCreditScoring Core`, `CreditScoring & Manual Pipeline`, `AutoCreditScoring Pipeline + Tests`, `Conftest Fixtures`, `IV Calculator Tests`?**
  _High betweenness centrality (0.170) - this node is a cross-community bridge._
- **Why does `Discretizer` connect `Discretizer Tests` to `AutoCreditScoring Core`?**
  _High betweenness centrality (0.136) - this node is a cross-community bridge._
- **Are the 16 inferred relationships involving `WoeEncoder` (e.g. with `fitted_encoder()` and `test_manual_pipeline_auc()`) actually correct?**
  _`WoeEncoder` has 16 INFERRED edges - model-reasoned connections that need verification._
- **Are the 15 inferred relationships involving `Discretizer` (e.g. with `test_all_nan_column()` and `test_empty_dataframe_returns_empty()`) actually correct?**
  _`Discretizer` has 15 INFERRED edges - model-reasoned connections that need verification._
- **Are the 12 inferred relationships involving `DiscreteNormalizer` (e.g. with `fitted_normalizer()` and `test_manual_pipeline_auc()`) actually correct?**
  _`DiscreteNormalizer` has 12 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `AutoCreditScoring` (e.g. with `WoeEncoder` and `WoeContinuousFeatureSelector`) actually correct?**
  _`AutoCreditScoring` has 5 INFERRED edges - model-reasoned connections that need verification._