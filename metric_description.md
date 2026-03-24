# Metric Description

## Background

Previous studies on active learning have commonly used metrics such as the area under
the learning curve (AUC), relative gains over baseline models, or accuracy at specific
query iterations. These approaches typically treat the performance of both the active
learning method and the baseline model as deterministic values at each iteration.
Consequently, they do not explicitly account for the inherent uncertainty in the
performance evaluation process, which can arise from random initialization, stochastic
training procedures, or sampling variability in the active learning process.

Some studies attempt to incorporate uncertainty by reporting error bars alongside
performance metrics. However, error bars or confidence intervals summarize only limited
statistics and fail to capture the full shape of the performance distribution. As a
result, they may not accurately reflect the probability that one method truly
outperforms another.

Another important limitation of existing evaluation methods is that they do not account
for the varying difficulty of achieving performance gains throughout the learning
process. As active learning progresses, obtaining the same level of relative improvement
becomes increasingly difficult, since both the active learning method and the baseline
approach gradually converge toward their maximum achievable performance. Traditional
metrics treat such improvements uniformly across all stages of learning, introducing
bias toward methods that perform better in later stages.

---

## Formal Definition

Let f(AL, N, θ_AL) denote the performance of an active learning method with
hyperparameters θ_AL after N query iterations, and f(BL, N, θ_BL) denote the
performance of the baseline model with hyperparameters θ_BL.

The **probabilistic learning curve** F(N) is defined as:

```
F(N) = P( f(AL, N, θ_AL) ≥ f(BL, N, θ_BL) )
```

which represents the probability that the active learning method performs at least
as well as the baseline at iteration N.

---

## Empirical Estimation

In practice, F(N) is estimated from empirical distributions obtained via repeated
experimental runs. Both the active learning method and the baseline are executed
multiple times with different random seeds. Let:

- { f_i(AL, N) }_{i=1}^{R_AL} — observed AL performance values from R_AL runs
- { f_j(BL, N) }_{j=1}^{R_BL} — observed baseline performance values from R_BL runs

The empirical estimate of F(N) is:

```
F(N) ≈ (1 / R_AL × R_BL) × Σ_i Σ_j 1( f_i(AL, N) ≥ f_j(BL, N) )
```

This approach captures the full shape of the performance distribution and provides
a probabilistic interpretation of the likelihood that the active learning method
outperforms the baseline.

---

## Summary Metric: AUC of F(N)

To obtain a single summary measure, the AUC of the F(N) curve is computed across
all query iterations. The AUC reflects the average probability that the active
learning method surpasses the baseline across the entire learning process.

If a method achieves a higher AUC of F(N), one can say that its superior performance
arises from the nature of the method itself rather than from random variation. This
metric thus accounts for both the magnitude and uncertainty of performance differences,
offering a more comprehensive evaluation than traditional deterministic metrics.

---

## Interpretation

The proposed probabilistic metric provides a more consistent interpretation of
performance improvement. By evaluating the likelihood that the active learning
method outperforms the baseline, the metric maintains a relatively stable notion
of difficulty across different stages of the learning process.

The increasing difficulty of achieving improvements is implicitly captured by the
distribution of model performance: as learning progresses, the performance
distributions of different methods become more concentrated and overlap more
significantly, making it harder to achieve a higher probability of superiority.

**Key threshold:** if F(N) > 0.5 at iteration N, the active learning method is
better than random against the baseline at that iteration. An AUC of F(N) > 0.5
indicates that the active learning method is, on average across all iterations,
more likely than not to outperform the baseline.
