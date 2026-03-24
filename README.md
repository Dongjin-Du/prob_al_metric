# prob\_al\_metric

[!\[Python](https://img.shields.io/badge/python-3.8%252B-blue)](https://www.python.org/)
[!\[License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
\[!\[Status](https://img.shields.io/badge/status-research-orange)]()

A Python package for evaluating active learning methods using a **probabilistic performance metric** that quantifies the likelihood that an active learning method outperforms a baseline across the entire learning process.

\---

## Motivation

Traditional active learning evaluation metrics — such as the AUC of the learning curve or accuracy at specific query iterations — treat performance as a deterministic value and fail to account for:

* **Stochasticity** in training and sampling (random seed effects)
* **Increasing difficulty** of performance gains in later stages of learning, when both methods converge toward their maximum performance

This package introduces a distribution-aware metric that addresses both limitations.

\---

## The Metric

At each query iteration N, the metric F(N) estimates the **probability that the active learning method outperforms the baseline**:

```
F(N) ≈ (1 / R_AL × R_BL) × Σ_i Σ_j 1( f_i(AL, N) ≥ f_j(BL, N) )
```

where R\_AL and R\_BL are the number of independent trials for each method, and f\_i, f\_j are the observed performance values.

The **AUC of F(N)** across all iterations provides a single summary score of overall superiority.

|F(N)|Interpretation|
|-|-|
|1.0|AL always outperforms baseline at iteration N|
|0.5|AL performs no better than random against baseline|
|< 0.5|Baseline outperforms AL at iteration N|

\---

## Installation

```bash
git clone https://github.com/Dongjin-Du/prob_al_metric.git
cd prob\\\_al\\\_metric
pip install -r requirements.txt
```

Or install directly:

```bash
pip install -e .
```

\---

## Quick Start

```python
from prob\\\_al\\\_metric import ProbALMetric, random\\\_sampling\\\_query
from sklearn.linear\\\_model import LogisticRegression
from sklearn.datasets import make\\\_classification
import numpy as np

# 1. Generate dataset
X, y = make\\\_classification(n\\\_samples=1000, n\\\_features=6,
                           n\\\_informative=4, n\\\_classes=3,
                           class\\\_sep=1.0, random\\\_state=42)

# 2. Define your AL query strategy
def uncertainty\\\_sampling(model, X\\\_labeled, y\\\_labeled, X\\\_unlabeled, batch\\\_size):
    probs = model.predict\\\_proba(X\\\_unlabeled)
    sorted\\\_probs = np.sort(probs, axis=1)
    margin = sorted\\\_probs\\\[:, -1] - sorted\\\_probs\\\[:, -2]
    return np.argsort(margin)\\\[:batch\\\_size]

# 3. Set up and run evaluation
metric = ProbALMetric(
    model              = LogisticRegression(max\\\_iter=1000),
    al\\\_query\\\_fn        = uncertainty\\\_sampling,
    baseline\\\_query\\\_fn  = random\\\_sampling\\\_query,  # built-in random baseline
    n\\\_queries          = 20,
    n\\\_trials           = 30,
    batch\\\_size         = 5,
    seed\\\_size          = 10,
    performance\\\_metric = 'auc',   # 'auc', 'accuracy', 'f1', or callable
)

results = metric.evaluate(X, y)

# 4. Outputs
results.plot()                    # F(N) curve + raw performance plot
print(results.auc\\\_of\\\_fn)          # scalar AUC of F(N)
print(results.fn\\\_values)          # F(N) at each iteration (array)
print(results.al\\\_scores)          # raw scores: shape (n\\\_trials, n\\\_queries+1)
print(results.bl\\\_scores)          # raw scores: shape (n\\\_trials, n\\\_queries+1)
```

\---

## Query Function API

Any query strategy — AL method or baseline — must follow this signature:

```python
def my\\\_query\\\_fn(model, X\\\_labeled, y\\\_labeled, X\\\_unlabeled, batch\\\_size):
    """
    Parameters
    ----------
    model        : fitted sklearn classifier
    X\\\_labeled    : np.ndarray, current labeled features
    y\\\_labeled    : np.ndarray, current labeled targets
    X\\\_unlabeled  : np.ndarray, current unlabeled features
    batch\\\_size   : int, number of samples to select

    Returns
    -------
    np.ndarray of shape (batch\\\_size,)
        Indices into X\\\_unlabeled of selected samples.
    """
    ...
    return indices
```

\---

## Custom Performance Metric

You can pass any callable as `performance\\\_metric`:

```python
def my\\\_metric(model, X\\\_test, y\\\_test):
    from sklearn.metrics import f1\\\_score
    y\\\_pred = model.predict(X\\\_test)
    return f1\\\_score(y\\\_test, y\\\_pred, average='weighted')

metric = ProbALMetric(
    ...,
    performance\\\_metric = my\\\_metric
)
```

\---

## Compare Two AL Methods

To compare two active learning methods against each other, pass the second method as `baseline\\\_query\\\_fn`:

```python
metric = ProbALMetric(
    model             = LogisticRegression(),
    al\\\_query\\\_fn       = entropy\\\_sampling,
    baseline\\\_query\\\_fn = uncertainty\\\_sampling,  # another AL method as baseline
    ...
)
```

\---

## Parameters

|Parameter|Type|Default|Description|
|-|-|-|-|
|`model`|sklearn estimator|required|Classifier with `fit()`, `predict()`, `predict\\\_proba()`|
|`al\\\_query\\\_fn`|callable|required|Query strategy for the AL method|
|`baseline\\\_query\\\_fn`|callable|`random\\\_sampling\\\_query`|Query strategy for the baseline|
|`n\\\_queries`|int|20|Number of query iterations|
|`n\\\_trials`|int|30|Independent trials per method (≥ 20 recommended)|
|`batch\\\_size`|int|5|Samples queried per iteration|
|`seed\\\_size`|int or dict|10|Initial labeled set size. Dict for class-balanced: `{0: 5, 1: 5}`|
|`performance\\\_metric`|str or callable|`'auc'`|`'auc'`, `'accuracy'`, `'f1'`, or custom callable|
|`test\\\_size`|float|0.2|Proportion of data held out for evaluation|
|`verbose`|bool|True|Print trial progress|

\---

## Outputs

|Attribute|Type|Description|
|-|-|-|
|`results.fn\\\_values`|np.ndarray `(n\\\_queries+1,)`|F(N) at each iteration|
|`results.auc\\\_of\\\_fn`|float|AUC of F(N) — primary summary metric|
|`results.al\\\_scores`|np.ndarray `(n\\\_trials, n\\\_queries+1)`|Raw AL performance scores|
|`results.bl\\\_scores`|np.ndarray `(n\\\_trials, n\\\_queries+1)`|Raw baseline performance scores|
|`results.plot()`|—|Plot F(N) curve and raw performance curves|

\---

## Project Structure

```
prob\\\_al\\\_metric/
├── prob\\\_al\\\_metric/
│   ├── \\\_\\\_init\\\_\\\_.py       # package entry point
│   └── core.py           # ProbALMetric, ProbALResults, random\\\_sampling\\\_query
├── examples/
│   └── example\\\_usage.py  # three worked examples
├── docs/
│   └── metric\\\_description.md
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

\---

## Citation

If you use this metric in your research, please cite:

```bibtex
@article{YOUR\\\_CITATION,
  title   = {YOUR PAPER TITLE},
  author  = {YOUR NAME},
  journal = {YOUR JOURNAL},
  year    = {2025}
}
```

\---

## License

MIT License. See [LICENSE](LICENSE) for details.

