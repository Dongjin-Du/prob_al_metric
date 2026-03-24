"""
prob_al_metric.py
=================
Probabilistic Performance Metric for Active Learning Evaluation.

Usage
-----
from prob_al_metric import ProbALMetric, random_sampling_query

metric = ProbALMetric(
    model=LogisticRegression(),
    al_query_fn=your_query_function,
    baseline_query_fn=random_sampling_query,  # or your own
    n_queries=20,
    n_trials=30,
    batch_size=5,
    seed_size=10,
    performance_metric='auc'  # 'auc', 'accuracy', 'f1', or callable
)

results = metric.evaluate(X, y)
results.plot()
print(results.auc_of_fn)
print(results.fn_values)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import label_binarize
from copy import deepcopy


# ─────────────────────────────────────────────
# Built-in baseline: random sampling
# ─────────────────────────────────────────────

def random_sampling_query(model, X_labeled, y_labeled, X_unlabeled, batch_size):
    """
    Baseline query strategy: select samples uniformly at random.

    Parameters
    ----------
    model         : fitted sklearn classifier (not used, kept for API consistency)
    X_labeled     : np.ndarray, current labeled features
    y_labeled     : np.ndarray, current labeled targets
    X_unlabeled   : np.ndarray, current unlabeled features
    batch_size    : int, number of samples to query

    Returns
    -------
    indices : np.ndarray of shape (batch_size,)
        Indices into X_unlabeled of selected samples.
    """
    n = len(X_unlabeled)
    return np.random.choice(n, size=min(batch_size, n), replace=False)


# ─────────────────────────────────────────────
# Performance measure helpers
# ─────────────────────────────────────────────

def _compute_performance(model, X_test, y_test, metric):
    """
    Compute a scalar performance value.

    Parameters
    ----------
    model  : fitted sklearn classifier
    X_test : np.ndarray
    y_test : np.ndarray
    metric : str or callable
        'auc'      - macro-averaged ROC AUC (one-vs-rest)
        'accuracy' - classification accuracy
        'f1'       - macro-averaged F1 score
        callable   - user-defined function(model, X_test, y_test) -> float

    Returns
    -------
    float
    """
    if callable(metric):
        return metric(model, X_test, y_test)

    y_pred = model.predict(X_test)

    if metric == 'accuracy':
        return accuracy_score(y_test, y_pred)

    elif metric == 'f1':
        return f1_score(y_test, y_pred, average='macro', zero_division=0)

    elif metric == 'auc':
        classes = np.unique(y_test)
        if len(classes) == 2:
            y_prob = model.predict_proba(X_test)[:, 1]
            return roc_auc_score(y_test, y_prob)
        else:
            y_bin  = label_binarize(y_test, classes=classes)
            y_prob = model.predict_proba(X_test)
            return roc_auc_score(y_bin, y_prob, average='macro', multi_class='ovr')

    else:
        raise ValueError(f"Unknown metric '{metric}'. Use 'auc', 'accuracy', 'f1', or a callable.")


# ─────────────────────────────────────────────
# Results container
# ─────────────────────────────────────────────

class ProbALResults:
    """
    Container for evaluation results.

    Attributes
    ----------
    fn_values    : np.ndarray of shape (n_queries+1,)
        F(N) values at each query iteration.
    auc_of_fn    : float
        AUC of the F(N) curve — primary summary metric.
    iterations   : np.ndarray
        Query iteration indices (0 = seed set only).
    al_scores    : np.ndarray of shape (n_trials, n_queries+1)
        Raw performance scores for the AL method across all trials.
    bl_scores    : np.ndarray of shape (n_trials, n_queries+1)
        Raw performance scores for the baseline across all trials.
    """

    def __init__(self, fn_values, al_scores, bl_scores):
        self.fn_values  = np.array(fn_values)
        self.al_scores  = np.array(al_scores)
        self.bl_scores  = np.array(bl_scores)
        self.iterations = np.arange(len(fn_values))
        self.auc_of_fn  = float(np.trapezoid(self.fn_values, self.iterations) /
                                max(self.iterations[-1], 1))

    def plot(self, figsize=(9, 4), title=None, show_raw=True):
        """
        Plot F(N) curve and optionally raw performance curves.

        Parameters
        ----------
        figsize  : tuple
        title    : str or None
        show_raw : bool
            If True, also plot mean AL and baseline performance curves.
        """
        if show_raw:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(figsize[0]*2, figsize[1]))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=figsize)

        # ── F(N) curve ──
        ax1.plot(self.iterations, self.fn_values,
                 color='#1D4ED8', linewidth=2.0, marker='o', markersize=4,
                 label='F(N)')
        ax1.axhline(0.5, color='gray', linewidth=1.2,
                    linestyle='--', label='F(N) = 0.5 (random)')
        ax1.fill_between(self.iterations, 0.5, self.fn_values,
                         where=self.fn_values >= 0.5,
                         alpha=0.15, color='#1D4ED8', label='AL superior region')
        ax1.fill_between(self.iterations, self.fn_values, 0.5,
                         where=self.fn_values < 0.5,
                         alpha=0.15, color='#DC2626', label='Baseline superior region')
        ax1.set_xlabel("Query Iteration", fontsize=11)
        ax1.set_ylabel("F(N)", fontsize=11)
        ax1.set_ylim(-0.05, 1.05)
        ax1.set_title(f"Probabilistic Learning Curve\nAUC of F(N) = {self.auc_of_fn:.4f}",
                      fontsize=11, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.25)

        # ── Raw performance curves ──
        if show_raw:
            al_mean = self.al_scores.mean(axis=0)
            bl_mean = self.bl_scores.mean(axis=0)
            al_std  = self.al_scores.std(axis=0)
            bl_std  = self.bl_scores.std(axis=0)

            ax2.plot(self.iterations, al_mean, color='#1D4ED8',
                     linewidth=2.0, label='AL method (mean)')
            ax2.fill_between(self.iterations,
                             al_mean - al_std, al_mean + al_std,
                             alpha=0.15, color='#1D4ED8')
            ax2.plot(self.iterations, bl_mean, color='#DC2626',
                     linewidth=2.0, linestyle='--', label='Baseline (mean)')
            ax2.fill_between(self.iterations,
                             bl_mean - bl_std, bl_mean + bl_std,
                             alpha=0.15, color='#DC2626')
            ax2.set_xlabel("Query Iteration", fontsize=11)
            ax2.set_ylabel("Performance", fontsize=11)
            ax2.set_title("Raw Performance Curves\n(mean ± std)", fontsize=11, fontweight='bold')
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.25)

        if title:
            fig.suptitle(title, fontsize=13, fontweight='bold', y=1.02)

        plt.tight_layout()
        plt.show()
        return fig


# ─────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────

class ProbALMetric:
    """
    Probabilistic Performance Metric for Active Learning Evaluation.

    Evaluates the likelihood that an active learning method outperforms
    a baseline at each query iteration, estimated from repeated trials.

    Parameters
    ----------
    model : sklearn-compatible classifier
        Must implement fit(), predict(), and predict_proba() (if using 'auc').

    al_query_fn : callable
        Query strategy for the active learning method.
        Signature: fn(model, X_labeled, y_labeled, X_unlabeled, batch_size)
                   -> np.ndarray of indices into X_unlabeled

    baseline_query_fn : callable, optional
        Query strategy for the baseline. Defaults to random_sampling_query.
        Same signature as al_query_fn.

    n_queries : int
        Number of query iterations to run.

    n_trials : int
        Number of independent trials (random seeds) per method.
        Higher values give more stable F(N) estimates. Recommended >= 20.

    batch_size : int
        Number of samples queried per iteration.

    seed_size : int or dict
        Number of initial labeled samples.
        If int: total seed size, sampled randomly.
        If dict: {class_label: n_samples} for class-balanced seeding.

    performance_metric : str or callable
        'auc'      - macro-averaged ROC AUC (default)
        'accuracy' - classification accuracy
        'f1'       - macro-averaged F1 score
        callable   - fn(model, X_test, y_test) -> float

    test_size : float
        Proportion of data held out for evaluation. Default 0.2.

    verbose : bool
        Print progress. Default True.
    """

    def __init__(self,
                 model,
                 al_query_fn,
                 baseline_query_fn=None,
                 n_queries=20,
                 n_trials=30,
                 batch_size=5,
                 seed_size=10,
                 performance_metric='auc',
                 test_size=0.2,
                 verbose=True):

        self.model              = model
        self.al_query_fn        = al_query_fn
        self.baseline_query_fn  = baseline_query_fn or random_sampling_query
        self.n_queries          = n_queries
        self.n_trials           = n_trials
        self.batch_size         = batch_size
        self.seed_size          = seed_size
        self.performance_metric = performance_metric
        self.test_size          = test_size
        self.verbose            = verbose

    def _get_seed_indices(self, y_pool):
        """Sample initial labeled indices from pool."""
        if isinstance(self.seed_size, dict):
            indices = []
            for cls, n in self.seed_size.items():
                cls_idx = np.where(y_pool == cls)[0]
                indices.extend(np.random.choice(cls_idx, size=min(n, len(cls_idx)),
                                                replace=False).tolist())
            return np.array(indices)
        else:
            return np.random.choice(len(y_pool), size=min(self.seed_size, len(y_pool)),
                                    replace=False)

    def _run_single_trial(self, X_pool, y_pool, X_test, y_test, query_fn, seed):
        """Run one full active learning trial and return performance at each iteration."""
        np.random.seed(seed)
        scores = []

        # Initial labeled set
        labeled_idx   = self._get_seed_indices(y_pool)
        unlabeled_idx = np.setdiff1d(np.arange(len(y_pool)), labeled_idx)

        X_lab, y_lab   = X_pool[labeled_idx], y_pool[labeled_idx]
        X_unlab        = X_pool[unlabeled_idx]

        # Score at iteration 0 (seed only)
        model = deepcopy(self.model)
        model.fit(X_lab, y_lab)
        scores.append(_compute_performance(model, X_test, y_test, self.performance_metric))

        for q in range(self.n_queries):
            if len(X_unlab) == 0:
                scores.append(scores[-1])
                continue

            # Query
            selected = query_fn(model, X_lab, y_lab, X_unlab, self.batch_size)

            # Update labeled / unlabeled
            X_lab   = np.vstack([X_lab,  X_unlab[selected]])
            y_lab   = np.concatenate([y_lab, y_pool[unlabeled_idx[selected]]])
            unlabeled_idx = np.delete(unlabeled_idx, selected)
            X_unlab = X_pool[unlabeled_idx]

            # Retrain and evaluate
            model = deepcopy(self.model)
            model.fit(X_lab, y_lab)
            scores.append(_compute_performance(model, X_test, y_test, self.performance_metric))

        return np.array(scores)

    def _compute_fn(self, al_scores, bl_scores):
        """
        Compute F(N) at each iteration.

        F(N) = (1 / R_AL * R_BL) * Σ_i Σ_j 1(f_i(AL,N) >= f_j(BL,N))
        """
        n_iters = al_scores.shape[1]
        fn = np.zeros(n_iters)
        R_AL, R_BL = al_scores.shape[0], bl_scores.shape[0]

        for n in range(n_iters):
            count = 0
            for i in range(R_AL):
                for j in range(R_BL):
                    if al_scores[i, n] >= bl_scores[j, n]:
                        count += 1
            fn[n] = count / (R_AL * R_BL)

        return fn

    def evaluate(self, X, y):
        """
        Run the full evaluation and return a ProbALResults object.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples,)

        Returns
        -------
        ProbALResults
        """
        X_pool, X_test, y_pool, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=0, stratify=y
        )

        al_scores = []
        bl_scores = []

        # ── AL trials ──
        for t in range(self.n_trials):
            if self.verbose:
                print(f"AL trial {t+1}/{self.n_trials}...", end='\r')
            scores = self._run_single_trial(X_pool, y_pool, X_test, y_test,
                                            self.al_query_fn, seed=t)
            al_scores.append(scores)

        # ── Baseline trials ──
        for t in range(self.n_trials):
            if self.verbose:
                print(f"Baseline trial {t+1}/{self.n_trials}...", end='\r')
            scores = self._run_single_trial(X_pool, y_pool, X_test, y_test,
                                            self.baseline_query_fn, seed=t + 10000)
            bl_scores.append(scores)

        if self.verbose:
            print("Done.                          ")

        al_scores = np.array(al_scores)
        bl_scores = np.array(bl_scores)

        fn_values = self._compute_fn(al_scores, bl_scores)

        return ProbALResults(fn_values, al_scores, bl_scores)
