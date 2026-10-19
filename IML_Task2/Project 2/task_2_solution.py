# Solution to the IML task2 (predicting price_CHF with missing features + noisy
# data). The approach is:
#   1. One-hot encode the categorical `season` feature.
#   2. Impute missing feature values with an IterativeImputer whose per-feature
#      estimator is a RandomForestRegressor -- this handles non-linear
#      relationships between the European electricity prices much better than
#      the default BayesianRidge estimator.
#   3. Do *multiple imputation*: run the imputer with several random seeds to
#      get independent imputed datasets. Train one Gaussian-Process regressor
#      on each and average their predictions. This reduces the variance coming
#      from the imputation step.
#   4. Drop the 262 training rows whose price_CHF target is missing (using them
#      would require imputing the target from the features, which makes the
#      target a deterministic function of the features and inflates CV R^2).
#   5. Fit a GaussianProcessRegressor with a Matern(nu=2.5) + WhiteKernel
#      kernel. GPs automatically tune kernel hyper-parameters by maximising
#      the log marginal likelihood (Bayesian model selection).

import numpy as np
import pandas as pd

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    Matern,
    WhiteKernel,
)
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score


SEASON_ORDER = ["spring", "summer", "autumn", "winter"]
N_IMPUTATIONS = 5


def _encode_season(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the `season` column with a fixed column order."""
    season_dummies = pd.get_dummies(df["season"], prefix="season")
    for s in SEASON_ORDER:
        col = f"season_{s}"
        if col not in season_dummies.columns:
            season_dummies[col] = 0
    season_dummies = season_dummies[[f"season_{s}" for s in SEASON_ORDER]]
    return pd.concat(
        [season_dummies.astype(float), df.drop(columns=["season"])],
        axis=1,
    )


def _impute_rf(combined: pd.DataFrame, seed: int) -> np.ndarray:
    """Iterative imputation with a RandomForest per-feature estimator."""
    imputer = IterativeImputer(
        estimator=RandomForestRegressor(
            n_estimators=100,
            random_state=seed,
            n_jobs=-1,
        ),
        max_iter=10,
        random_state=seed,
    )
    return imputer.fit_transform(combined)


def load_data():
    """
    Loads train/test CSVs, one-hot encodes `season`, and returns the
    concatenated (train+test) feature matrix along with the target.

    Feature imputation is performed externally (multiple times) by
    `Model.fit_and_predict` so the returned feature matrix here only contains
    the season one-hots + the raw (still-with-NaN) foreign prices.
    """
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    print("Training data:")
    print("Shape:", train_df.shape)
    print(train_df.head(2))
    print("\nTest data:")
    print("Shape:", test_df.shape)
    print(test_df.head(2))

    train_enc = _encode_season(train_df)
    test_enc = _encode_season(test_df)

    feature_cols = list(test_enc.columns)
    assert "price_CHF" not in feature_cols

    combined_features = pd.concat(
        [train_enc[feature_cols], test_enc[feature_cols]],
        axis=0,
        ignore_index=True,
    )

    y_train_full = train_enc["price_CHF"].to_numpy(dtype=float)
    n_train = len(train_enc)

    # Drop rows with missing target (y is not used by the imputer because we
    # only impute features -- otherwise y would be a deterministic function of
    # X and CV R^2 would be meaningless).
    keep = ~np.isnan(y_train_full)
    y_train = y_train_full[keep]

    X_train_full = combined_features  # still contains NaN; imputed later
    print(f"\nKept {keep.sum()} of {n_train} training rows "
          f"(dropped {(~keep).sum()} rows with missing price_CHF).")

    assert X_train_full.shape[1] == len(feature_cols)
    return X_train_full, y_train, n_train, keep


class Model(object):
    """Gaussian Process Regressor averaged over several RF-imputations.

    The GP kernel is `Constant * Matern(nu=2.5) + WhiteKernel`. Its
    hyper-parameters are selected by maximising the log marginal likelihood
    (Bayesian model selection), with several restarts to avoid local optima.
    """

    def __init__(self, n_imputations: int = N_IMPUTATIONS):
        super().__init__()
        self.n_imputations = n_imputations
        self._imputations: list[np.ndarray] = []
        self._models: list[GaussianProcessRegressor] = []
        self._keep: np.ndarray | None = None
        self._n_train: int = 0

    @staticmethod
    def _make_kernel():
        return (
            ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
            + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1e1))
        )

    @staticmethod
    def _make_gpr():
        return GaussianProcessRegressor(
            kernel=Model._make_kernel(),
            normalize_y=True,
            n_restarts_optimizer=10,
            alpha=1e-6,
            random_state=0,
        )

    def fit(self, X_combined: pd.DataFrame, y_train: np.ndarray,
            n_train: int, keep: np.ndarray):
        """Run `n_imputations` imputations and fit one GPR to each."""
        self._keep = keep
        self._n_train = n_train
        self._imputations = []
        self._models = []
        for i in range(self.n_imputations):
            print(f"  imputation {i + 1}/{self.n_imputations}...")
            imputed = _impute_rf(X_combined, seed=i)
            self._imputations.append(imputed)
            X_tr = imputed[:n_train][keep]
            gpr = self._make_gpr()
            gpr.fit(X_tr, y_train)
            print(f"    kernel: {gpr.kernel_}  "
                  f"log-ML: {gpr.log_marginal_likelihood_value_:.3f}")
            self._models.append(gpr)

    def predict_test(self) -> np.ndarray:
        """Average predictions over imputations for the held-out test rows."""
        assert self._models, "Call fit() before predict_test()"
        preds = []
        for imputed, gpr in zip(self._imputations, self._models):
            X_te = imputed[self._n_train:]
            preds.append(gpr.predict(X_te))
        return np.mean(preds, axis=0)


def _cross_validate(X_combined: pd.DataFrame, y_train: np.ndarray,
                    n_train: int, keep: np.ndarray,
                    n_imputations: int = N_IMPUTATIONS,
                    n_splits: int = 5) -> float:
    """K-Fold CV over the training rows with kept targets.

    Uses the same multiple-imputation strategy as the final model: `n_imputations`
    imputed feature matrices, one GP per matrix, predictions averaged.
    """
    print(f"Running {n_splits}-fold CV with {n_imputations} imputations...")
    imputations = [_impute_rf(X_combined, seed=i) for i in range(n_imputations)]
    X_tr_list = [arr[:n_train][keep] for arr in imputations]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    scores = []
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_tr_list[0])):
        preds = []
        for X in X_tr_list:
            gpr = Model._make_gpr()
            gpr.fit(X[tr_idx], y_train[tr_idx])
            preds.append(gpr.predict(X[va_idx]))
        avg = np.mean(preds, axis=0)
        score = r2_score(y_train[va_idx], avg)
        scores.append(score)
        print(f"  Fold {fold + 1}/{n_splits}: R^2 = {score:.4f}")
    mean_score = float(np.mean(scores))
    print(f"Mean CV R^2: {mean_score:.4f} (+/- {np.std(scores):.4f})")
    return mean_score


if __name__ == "__main__":
    X_combined, y_train, n_train, keep = load_data()

    # Uncomment to re-check CV (slow: ~5 min with RF imputations x 5 x 5 folds).
    # _cross_validate(X_combined, y_train, n_train, keep)

    print("\nFitting final model (5 RF-imputations + GP each)...")
    model = Model(n_imputations=N_IMPUTATIONS)
    model.fit(X_combined, y_train, n_train, keep)

    y_pred = model.predict_test()
    assert y_pred.shape == (100,)

    dt = pd.DataFrame(y_pred, columns=["price_CHF"])
    dt.to_csv("results.csv", index=False)
    print("\nResults file successfully generated!")
