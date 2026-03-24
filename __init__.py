"""
prob_al_metric
==============
Probabilistic Performance Metric for Active Learning Evaluation.

Quickstart
----------
>>> from prob_al_metric import ProbALMetric, random_sampling_query
>>> metric = ProbALMetric(model=..., al_query_fn=..., n_trials=30)
>>> results = metric.evaluate(X, y)
>>> results.plot()
>>> print(results.auc_of_fn)
"""

from .core import ProbALMetric, ProbALResults, random_sampling_query

__all__ = ["ProbALMetric", "ProbALResults", "random_sampling_query"]
__version__ = "0.1.0"
