"""
example_usage.py
================
Example demonstrating how to use ProbALMetric.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from prob_al_metric import ProbALMetric, random_sampling_query


# ─────────────────────────────────────────────
# Define a custom AL query strategy (uncertainty sampling)
# ─────────────────────────────────────────────

def uncertainty_sampling(model, X_labeled, y_labeled, X_unlabeled, batch_size):
    """Select samples where the model is least confident."""
    probs  = model.predict_proba(X_unlabeled)
    # margin between top two probabilities
    sorted_probs = np.sort(probs, axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    # lowest margin = most uncertain
    return np.argsort(margin)[:batch_size]


# ─────────────────────────────────────────────
# Generate dataset
# ─────────────────────────────────────────────

X, y = make_classification(
    n_samples=1000,
    n_features=6,
    n_informative=4,
    n_redundant=2,
    n_classes=3,
    n_clusters_per_class=1,
    class_sep=1.0,
    random_state=42
)

# ─────────────────────────────────────────────
# Example 1: AL method vs random baseline
# ─────────────────────────────────────────────

metric = ProbALMetric(
    model               = LogisticRegression(max_iter=1000),
    al_query_fn         = uncertainty_sampling,
    baseline_query_fn   = random_sampling_query,  # default random baseline
    n_queries           = 20,
    n_trials            = 30,
    batch_size          = 5,
    seed_size           = 10,
    performance_metric  = 'auc',   # 'auc', 'accuracy', 'f1', or callable
    test_size           = 0.2,
    verbose             = True
)

results = metric.evaluate(X, y)

# Plot F(N) curve + raw performance
results.plot(title="Uncertainty Sampling vs Random (LR, class_sep=1.0)")

# Get scalar AUC of F(N)
print(f"\nAUC of F(N): {results.auc_of_fn:.4f}")

# Get raw F(N) array
print(f"F(N) values: {results.fn_values.round(3)}")


# ─────────────────────────────────────────────
# Example 2: Custom performance metric
# ─────────────────────────────────────────────

def custom_metric(model, X_test, y_test):
    """Example: weighted F1 score."""
    from sklearn.metrics import f1_score
    y_pred = model.predict(X_test)
    return f1_score(y_test, y_pred, average='weighted', zero_division=0)

metric2 = ProbALMetric(
    model               = RandomForestClassifier(n_estimators=50, random_state=0),
    al_query_fn         = uncertainty_sampling,
    baseline_query_fn   = random_sampling_query,
    n_queries           = 20,
    n_trials            = 20,
    batch_size          = 5,
    seed_size           = 10,
    performance_metric  = custom_metric,   # pass your own function
    verbose             = True
)

results2 = metric2.evaluate(X, y)
results2.plot(title="Uncertainty Sampling vs Random (RF, weighted F1)")
print(f"\nAUC of F(N): {results2.auc_of_fn:.4f}")


# ─────────────────────────────────────────────
# Example 3: Compare two AL methods against each other
# ─────────────────────────────────────────────

def random_forest_uncertainty(model, X_labeled, y_labeled, X_unlabeled, batch_size):
    """Uncertainty sampling wrapper (same function, used as 'baseline' here)."""
    return uncertainty_sampling(model, X_labeled, y_labeled, X_unlabeled, batch_size)

def entropy_sampling(model, X_labeled, y_labeled, X_unlabeled, batch_size):
    """Select samples with highest prediction entropy."""
    probs   = model.predict_proba(X_unlabeled)
    probs   = np.clip(probs, 1e-10, 1)
    entropy = -np.sum(probs * np.log(probs), axis=1)
    return np.argsort(entropy)[::-1][:batch_size]

metric3 = ProbALMetric(
    model               = LogisticRegression(max_iter=1000),
    al_query_fn         = entropy_sampling,
    baseline_query_fn   = random_forest_uncertainty,  # compare two AL methods
    n_queries           = 20,
    n_trials            = 20,
    batch_size          = 5,
    seed_size           = 10,
    performance_metric  = 'accuracy',
    verbose             = True
)

results3 = metric3.evaluate(X, y)
results3.plot(title="Entropy Sampling vs Uncertainty Sampling (LR, accuracy)")
print(f"\nAUC of F(N): {results3.auc_of_fn:.4f}")
