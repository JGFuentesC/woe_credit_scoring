# Graph Report - .  (2026-07-15)

## Corpus Check
- Corpus is ~23,370 words - fits in a single context window. You may not need a graph.

## Summary
- 152 nodes · 261 edges · 14 communities (10 shown, 4 thin omitted)
- Extraction: 87% EXTRACTED · 13% INFERRED · 0% AMBIGUOUS · INFERRED: 33 edges (avg confidence: 0.67)
- Token cost: 23,357 input · 1,384 output

## Community Hubs (Navigation)
- Binning & Discretization
- Core Module Integration
- AutoCreditScoring Pipeline
- Normalizer & Tests
- Feature Selector Base
- Unit Test Suite
- WOE Encoder Core
- Documentation & Results
- Component Documentation
- Reporter Module
- Changelog
- Package Metadata
- NumPy Dependency
- Pandas Dependency

## God Nodes (most connected - your core abstractions)
1. `AutoCreditScoring` - 23 edges
2. `DiscreteNormalizer` - 22 edges
3. `WoeEncoder` - 19 edges
4. `WoeContinuousFeatureSelector` - 14 edges
5. `WoeDiscreteFeatureSelector` - 14 edges
6. `Discretizer` - 12 edges
7. `CreditScoring` - 12 edges
8. `WoeBaseFeatureSelector` - 11 edges
9. `test_manual_pipeline_auc()` - 7 edges
10. `IVCalculator` - 7 edges

## Surprising Connections (you probably didn't know these)
- `test_woe_calculation()` --calls--> `WoeEncoder`  [INFERRED]
  tests/unit/test_encoder.py → woe_credit_scoring/encoder.py
- `test_woe_transform()` --calls--> `WoeEncoder`  [INFERRED]
  tests/unit/test_encoder.py → woe_credit_scoring/encoder.py
- `test_woe_inverse_transform()` --calls--> `WoeEncoder`  [INFERRED]
  tests/unit/test_encoder.py → woe_credit_scoring/encoder.py
- `test_woe_with_missing_values()` --calls--> `WoeEncoder`  [INFERRED]
  tests/unit/test_encoder.py → woe_credit_scoring/encoder.py
- `test_woe_discrete_feature_selector()` --calls--> `WoeDiscreteFeatureSelector`  [INFERRED]
  tests/unit/test_feature_selectors.py → woe_credit_scoring/binning.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **WoE Feature Selection Components** — readme_woebasefeatureselector, readme_woecontinuousfeatureselector, readme_woediscretefeatureselector [EXTRACTED 1.00]
- **Credit Scoring Pipeline** — readme_discretenormalizer, readme_discretizer, readme_woeencoder, readme_creditscoring [EXTRACTED 0.90]

## Communities (14 total, 4 thin omitted)

### Community 0 - "Binning & Discretization"
Cohesion: 0.12
Nodes (16): Discretizer, DataFrame, Series, Encodes continuous feature into a discrete bin.          Args:             X (pd, Discretizer class for transforming continuous data into discrete bins.      This, Transforms continuous data into its discrete form.          Args:             X, Learns the best features given an IV threshold. Monotonic risk restriction can b, Converts continuous features to their best discretization.          Args: (+8 more)

### Community 1 - "Core Module Integration"
Cohesion: 0.15
Nodes (13): LogisticRegression, test_manual_pipeline_auc(), WoeContinuousFeatureSelector is a class for selecting continuous features based, WoeContinuousFeatureSelector, WoeEncoder is a class for encoding discrete features into Weight of Evidence (Wo, WoeEncoder, CreditScoring, DataFrame (+5 more)

### Community 2 - "AutoCreditScoring Pipeline"
Cohesion: 0.18
Nodes (5): test_autocreditscoring_pipeline(), AutoCreditScoring, DataFrame, A class used to perform automated credit scoring using logistic regression and W, Predicts scores for a given raw dataset.          The input data should have the

### Community 3 - "Normalizer & Tests"
Cohesion: 0.14
Nodes (13): test_missing_value_handling(), test_no_small_categories(), test_small_category_aggregation(), test_unseen_categories(), DiscreteNormalizer, DataFrame, Series, Transforms discrete data into its normalized form.          Args:             X (+5 more)

### Community 4 - "Feature Selector Base"
Cohesion: 0.15
Nodes (8): Series, Computes information value (IV) statistic.          Args:             X (pd.Seri, Base class for selecting features based on their Weight of Evidence (WoE)     tr, Validates if a given discretized feature has monotonic risk behavior.          A, WoeBaseFeatureSelector, IVCalculator, A class to calculate the Information Value (IV) for both discrete and continuous, Initializes the IVCalculator object.          Args:             data (pd.DataFra

### Community 5 - "Unit Test Suite"
Cohesion: 0.15
Nodes (6): test_woe_calculation(), test_woe_inverse_transform(), test_woe_transform(), test_woe_with_missing_values(), test_woe_continuous_feature_selector(), test_woe_discrete_feature_selector()

### Community 6 - "WOE Encoder Core"
Cohesion: 0.22
Nodes (6): DataFrame, Series, Learns WoE encoding.          Args:             X (pd.DataFrame): Data with disc, Calculates WoE Map between discrete space and log odds space.          Args:, Performs WoE transformation.          Args:             X (pd.DataFrame): Discre, Performs Inverse WoE transformation.          Args:             X (pd.DataFrame)

### Community 7 - "Documentation & Results"
Cohesion: 0.25
Nodes (8): Feature Importance Chart, ROC AUC Curve, Score Histogram, Score KDE Plot, AutoCreditScoring, CreditScoring, IVCalculator, Scikit-Learn

### Community 8 - "Component Documentation"
Cohesion: 0.29
Nodes (7): Scoring Method Workflow Diagram, DiscreteNormalizer, Discretizer, WoeBaseFeatureSelector, WoeContinuousFeatureSelector, WoeDiscreteFeatureSelector, WoeEncoder

### Community 9 - "Reporter Module"
Cohesion: 0.50
Nodes (3): frequency_table(), DataFrame, Displays a frequency table for the specified variables in the DataFrame.      Ar

## Knowledge Gaps
- **15 isolated node(s):** `woe_credit_scoring`, `DiscreteNormalizer`, `Discretizer`, `WoeEncoder`, `WoeContinuousFeatureSelector` (+10 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `DiscreteNormalizer` connect `Normalizer & Tests` to `Binning & Discretization`, `Core Module Integration`, `AutoCreditScoring Pipeline`, `Feature Selector Base`?**
  _High betweenness centrality (0.229) - this node is a cross-community bridge._
- **Why does `WoeEncoder` connect `Core Module Integration` to `AutoCreditScoring Pipeline`, `Unit Test Suite`, `WOE Encoder Core`?**
  _High betweenness centrality (0.201) - this node is a cross-community bridge._
- **Why does `AutoCreditScoring` connect `AutoCreditScoring Pipeline` to `Binning & Discretization`, `Core Module Integration`, `Normalizer & Tests`?**
  _High betweenness centrality (0.185) - this node is a cross-community bridge._
- **Are the 5 inferred relationships involving `AutoCreditScoring` (e.g. with `WoeContinuousFeatureSelector` and `WoeDiscreteFeatureSelector`) actually correct?**
  _`AutoCreditScoring` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 10 inferred relationships involving `DiscreteNormalizer` (e.g. with `test_manual_pipeline_auc()` and `test_missing_value_handling()`) actually correct?**
  _`DiscreteNormalizer` has 10 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `WoeEncoder` (e.g. with `test_manual_pipeline_auc()` and `test_woe_calculation()`) actually correct?**
  _`WoeEncoder` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `WoeContinuousFeatureSelector` (e.g. with `test_manual_pipeline_auc()` and `test_woe_continuous_feature_selector()`) actually correct?**
  _`WoeContinuousFeatureSelector` has 5 INFERRED edges - model-reasoned connections that need verification._