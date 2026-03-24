# prob_al_metric

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/status-research-orange)]()

A Python package for evaluating active learning methods using a **probabilistic performance metric** that quantifies the likelihood that an active learning method outperforms a baseline across the entire learning process.

---

## Motivation

Traditional active learning evaluation metrics — such as the AUC of the learning curve or accuracy at specific query iterations — treat performance as a deterministic value and fail to account for:

- **Stochasticity** in training and sampling (random seed effects)
- **Increasing difficulty** of performance gains in later stages of learning, when both methods converge toward their maximum performance

This package introduces a distribution-aware metric that addresses both limitations.

---

## The Metric

At each query iteration N, the metric F(N) estimates the **probability that the active learning method outperforms the baseline**:

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?F(N)\approx\frac{1}{R_{AL}\times&space;R_{BL}}\sum_{i=1}^{R_{AL}}\sum_{j=1}^{R_{BL}}\mathbf{1}\left(f_i(AL,N)\geq&space;f_j(BL,N)\right)" />
</p>

where R_AL and R_BL are the number of independent trials for each method, and f_i, f_j are the observed performance values.

The **AUC of F(N)** across all iterations provides a single summary score of overall superiority.

| F(N) | Interpretation |
|------|----------------|
| 1.0  | AL always outperforms baseline at iteration N |
| 0.5  | AL performs no better than random against baseline |
| < 0.5 | Baseline outperforms AL at iteration N |

---

## Installation

```bash
git clone https://github.com/Dongjin-Du/prob_al_metric
cd prob_al_metric
pip install -r requirements.txt
```

Or install directly:

```bash
pip install -e .
```

---

## Quick Start

```python
from prob_al_metric import ProbALMetric, random_sampling_query
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import numpy as np

# 1. Generate dataset
X, y = make_classification(n_samples=1000, n_features=6,
                           n_informative=4, n_classes=3,
                           class_sep=1.0, random_state=42)

# 2. Define your AL query strategy
def uncertainty_sampling(model, X_labeled, y_labeled, X_unlabeled, batch_size):
    probs = model.predict_proba(X_unlabeled)
    sorted_probs = np.sort(probs, axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    return np.argsort(margin)[:batch_size]

# 3. Set up and run evaluation
metric = ProbALMetric(
    model              = LogisticRegression(max_iter=1000),
    al_query_fn        = uncertainty_sampling,
    baseline_query_fn  = random_sampling_query,
    n_queries          = 20,
    n_trials           = 30,
    batch_size         = 5,
    seed_size          = 10,
    performance_metric = 'auc',
)

results = metric.evaluate(X, y)

# 4. Outputs
results.plot()                 # F(N) curve + raw performance plot
print(results.auc_of_fn)       # scalar AUC of F(N)
print(results.fn_values)       # F(N) at each iteration (array)
print(results.al_scores)       # raw scores: shape (n_trials, n_queries+1)
print(results.bl_scores)       # raw scores: shape (n_trials, n_queries+1)
```

---

## Query Function API

Any query strategy must follow this signature:

```python
def my_query_fn(model, X_labeled, y_labeled, X_unlabeled, batch_size):
    """
    Parameters
    ----------
    model        : fitted sklearn classifier
    X_labeled    : np.ndarray, current labeled features
    y_labeled    : np.ndarray, current labeled targets
    X_unlabeled  : np.ndarray, current unlabeled features
    batch_size   : int, number of samples to select

    Returns
    -------
    np.ndarray of shape (batch_size,)
        Indices into X_unlabeled of selected samples.
    """
    return indices
```

---

## Custom Performance Metric

Pass any callable as `performance_metric`:

```python
def my_metric(model, X_test, y_test):
    from sklearn.metrics import f1_score
    y_pred = model.predict(X_test)
    return f1_score(y_test, y_pred, average='weighted')

metric = ProbALMetric(
    ...,
    performance_metric = my_metric
)
```

---

## Compare Two AL Methods

To compare two active learning methods against each other, pass the second method as `baseline_query_fn`:

```python
metric = ProbALMetric(
    model             = LogisticRegression(),
    al_query_fn       = entropy_sampling,
    baseline_query_fn = uncertainty_sampling,
    ...
)
```

---

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| model | sklearn estimator | required | Classifier with fit(), predict(), predict_proba() |
| al_query_fn | callable | required | Query strategy for the AL method |
| baseline_query_fn | callable | random_sampling_query | Query strategy for the baseline |
| n_queries | int | 20 | Number of query iterations |
| n_trials | int | 30 | Independent trials per method (20+ recommended) |
| batch_size | int | 5 | Samples queried per iteration |
| seed_size | int or dict | 10 | Initial labeled set size. Dict for class-balanced: {0: 5, 1: 5} |
| performance_metric | str or callable | 'auc' | 'auc', 'accuracy', 'f1', or custom callable |
| test_size | float | 0.2 | Proportion of data held out for evaluation |
| verbose | bool | True | Print trial progress |

---

## Outputs

| Attribute | Type | Description |
|-----------|------|-------------|
| results.fn_values | np.ndarray (n_queries+1,) | F(N) at each iteration |
| results.auc_of_fn | float | AUC of F(N) — primary summary metric |
| results.al_scores | np.ndarray (n_trials, n_queries+1) | Raw AL performance scores |
| results.bl_scores | np.ndarray (n_trials, n_queries+1) | Raw baseline performance scores |
| results.plot() | — | Plot F(N) curve and raw performance curves |

---

## Project Structure

```
prob_al_metric/
├── prob_al_metric/
│   ├── __init__.py       
│   └── core.py           
├── examples/
│   └── example_usage.py  
├── docs/
│   └── metric_description.md
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

---

## Citation

```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
