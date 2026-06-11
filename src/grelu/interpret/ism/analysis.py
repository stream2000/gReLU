"""QC, EDA, clustering, and paper-label reproduction benchmarks."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from grelu.interpret.ism.config import RunConfig
from grelu.interpret.ism.errors import ISMUserError
from grelu.interpret.ism.provenance import (
    output_dir,
    prepare_stage_outputs,
    record_stage,
    reject_completed_downstream_stages,
    require_completed_stage,
)


class AnalysisError(ISMUserError, RuntimeError):
    """Raised when CP5 analysis cannot satisfy its contract."""


EFFECT_SUFFIXES = (
    "__S_ref",
    "__log2fc",
    "__control_adjusted_score",
    "__motif_vs_control_z",
)


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )


def _base_feature(column: str) -> str:
    for suffix in EFFECT_SUFFIXES:
        if column.endswith(suffix):
            return column[: -len(suffix)]
    return column


def _load_candidate_matrix(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    signal = pd.read_csv(
        root / "raw_signal_feature_matrix.tsv", sep="\t", low_memory=False
    )
    delta = pd.read_csv(
        root / "raw_delta_feature_matrix.tsv", sep="\t", low_memory=False
    )
    adjusted = pd.read_csv(
        root / "control_adjusted_feature_matrix.tsv",
        sep="\t",
        low_memory=False,
    )
    registry = pd.read_csv(
        root / "feature_registry.tsv", sep="\t", low_memory=False
    )
    signal_columns = ["site_id"] + [
        column for column in signal.columns if column.endswith("__S_ref")
    ]
    matrix = signal.loc[:, signal_columns]
    matrix = matrix.merge(
        delta.loc[
            :,
            ["site_id"]
            + [
                column
                for column in delta.columns
                if column.endswith("__log2fc")
            ],
        ],
        on="site_id",
        how="outer",
        validate="one_to_one",
    )
    matrix = matrix.merge(
        adjusted.loc[
            :,
            ["site_id"]
            + [
                column
                for column in adjusted.columns
                if column.endswith(
                    ("__control_adjusted_score", "__motif_vs_control_z")
                )
            ],
        ],
        on="site_id",
        how="outer",
        validate="one_to_one",
    )
    if matrix["site_id"].duplicated().any():
        raise AnalysisError("Candidate feature matrix has duplicate site_id rows")
    return matrix, registry


def _feature_family_map(registry: pd.DataFrame) -> dict[str, str]:
    return (
        registry.drop_duplicates("feature_name")
        .set_index("feature_name")["paper_feature_family"]
        .astype(str)
        .to_dict()
    )


def _low_signal_rates(root: Path) -> dict[str, float]:
    signal = pd.read_csv(
        root / "raw_signal_feature_matrix.tsv", sep="\t", low_memory=False
    )
    rates = {}
    for column in signal.columns:
        if not column.endswith("__low_ref_signal_flag"):
            continue
        base = column[: -len("__low_ref_signal_flag")]
        values = signal[column]
        if not pd.api.types.is_bool_dtype(values):
            values = _as_bool(values)
        rates[base] = float(values.mean())
    return rates


def _feature_qc(
    matrix: pd.DataFrame,
    registry: pd.DataFrame,
    low_signal_rates: dict[str, float],
    config: RunConfig,
) -> pd.DataFrame:
    families = _feature_family_map(registry)
    rows = []
    for column in matrix.columns:
        if column == "site_id":
            continue
        numeric = pd.to_numeric(matrix[column], errors="coerce")
        finite = numeric.replace([np.inf, -np.inf], np.nan)
        q1 = finite.quantile(0.25)
        q3 = finite.quantile(0.75)
        iqr = float(q3 - q1) if np.isfinite(q1) and np.isfinite(q3) else 0.0
        missing_rate = float(finite.isna().mean())
        base = _base_feature(column)
        low_rate = float(low_signal_rates.get(base, 0.0))
        reasons = []
        if missing_rate > config.analysis.missingness_threshold:
            reasons.append("missingness")
        if low_rate > config.analysis.low_signal_rate_threshold:
            reasons.append("low_ref_signal")
        if not np.isfinite(iqr) or iqr <= np.finfo(float).eps:
            reasons.append("zero_iqr")
        rows.append(
            {
                "feature_name": column,
                "base_feature_name": base,
                "feature_family": families.get(base, "unregistered"),
                "missing_rate": missing_rate,
                "low_signal_rate": low_rate,
                "iqr": iqr,
                "included_in_eda": not reasons,
                "exclusion_reason": ",".join(reasons),
            }
        )
    return pd.DataFrame.from_records(rows)


def _impute_scale(
    matrix: pd.DataFrame,
    columns: Sequence[str],
) -> tuple[np.ndarray, Any, Any]:
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import RobustScaler

    numeric = matrix.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    imputed = imputer.fit_transform(numeric)
    return scaler.fit_transform(imputed), imputer, scaler


def _ordered_correlation(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    correlation = np.corrcoef(data, rowvar=False)
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    if correlation.shape[0] < 3:
        return correlation, np.arange(correlation.shape[0])
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    distance = np.clip(1.0 - correlation, 0.0, 2.0)
    distance = 0.5 * (distance + distance.T)
    np.fill_diagonal(distance, 0.0)
    order = leaves_list(linkage(squareform(distance), method="average"))
    return correlation, order


def _plot_heatmaps(
    scaled: pd.DataFrame,
    sites: pd.DataFrame,
    qc: pd.DataFrame,
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    top = (
        qc.loc[qc["included_in_eda"]]
        .sort_values(["iqr", "feature_name"], ascending=[False, True])
        .head(100)["feature_name"]
        .tolist()
    )
    joined = sites.loc[
        :, ["site_id", "DIC_label", "HC_DIC_label", "LC_DIC_label"]
    ].merge(scaled, on="site_id", how="inner", validate="one_to_one")
    subsets = (
        ("All", np.ones(len(joined), dtype=bool)),
        ("DIC", _as_bool(joined["DIC_label"]).to_numpy()),
        ("HC-DIC", _as_bool(joined["HC_DIC_label"]).to_numpy()),
        ("LC-DIC", _as_bool(joined["LC_DIC_label"]).to_numpy()),
    )
    figure, axes = plt.subplots(2, 2, figsize=(14, 12))
    image = None
    for axis, (title, mask) in zip(axes.flat, subsets):
        if mask.sum() < 3 or len(top) < 2:
            axis.text(0.5, 0.5, "Insufficient rows/features", ha="center")
            axis.set_axis_off()
            axis.set_title(title)
            continue
        values = joined.loc[mask, top].to_numpy(dtype=float)
        correlation, order = _ordered_correlation(values)
        image = axis.imshow(
            correlation[np.ix_(order, order)],
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            aspect="auto",
        )
        axis.set_title(f"{title} (n={int(mask.sum())})")
        axis.set_xticks([])
        axis.set_yticks([])
    if image is not None:
        figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.7)
    figure.suptitle("Pearson feature correlation; hierarchical ordering")
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _embedding_outputs(
    values: np.ndarray,
    columns: Sequence[str],
    sites: pd.DataFrame,
    config: RunConfig,
    variance_path: Path,
    plot_path: Path,
) -> tuple[np.ndarray, pd.DataFrame]:
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    n_components = min(10, values.shape[0] - 1, values.shape[1])
    if n_components < 2:
        raise AnalysisError("PCA requires at least two rows and two features")
    pca = PCA(n_components=n_components, random_state=config.analysis.random_seed)
    coordinates = pca.fit_transform(values)
    variance = pd.DataFrame(
        {
            "component": np.arange(1, n_components + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance_ratio": np.cumsum(
                pca.explained_variance_ratio_
            ),
        }
    )
    variance.to_csv(variance_path, sep="\t", index=False)

    labels = sites["paper_class"].fillna("unclassified").astype(str)
    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    for label in sorted(labels.unique()):
        mask = labels.eq(label).to_numpy()
        axes[0].scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=16,
            alpha=0.75,
            label=label,
        )
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].set_title("PCA by paper label")
    axes[0].legend(fontsize=7, loc="best")
    if config.analysis.enable_umap:
        try:
            import umap
        except ImportError as exc:
            raise AnalysisError(
                "analysis.enable_umap=true requires the umap-learn package"
            ) from exc
        embedding = umap.UMAP(
            random_state=config.analysis.random_seed
        ).fit_transform(values)
        for label in sorted(labels.unique()):
            mask = labels.eq(label).to_numpy()
            axes[1].scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                s=16,
                alpha=0.75,
                label=label,
            )
        axes[1].set_xlabel("UMAP1")
        axes[1].set_ylabel("UMAP2")
        axes[1].set_title("UMAP by paper label")
    else:
        axes[1].text(
            0.5,
            0.5,
            "UMAP disabled by configuration",
            ha="center",
            va="center",
        )
        axes[1].set_axis_off()
    figure.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return coordinates, variance


def _cluster_outputs(
    values: np.ndarray,
    columns: Sequence[str],
    sites: pd.DataFrame,
    config: RunConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from sklearn.cluster import KMeans
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.metrics import adjusted_rand_score

    if len(values) < config.analysis.kmeans_k:
        raise AnalysisError(
            f"k=10 requires at least 10 sites; observed {len(values)}"
        )
    reference = KMeans(
        n_clusters=config.analysis.kmeans_k,
        n_init=20,
        random_state=config.analysis.random_seed,
    ).fit(values)
    assignments = sites.loc[:, ["site_id"]].copy()
    assignments["cluster"] = reference.labels_
    centroids = pd.DataFrame(reference.cluster_centers_, columns=columns)
    centroids.insert(0, "cluster", np.arange(config.analysis.kmeans_k))

    rng = np.random.default_rng(config.analysis.random_seed)
    stability_rows = []
    for iteration in range(config.analysis.cluster_bootstrap_iterations):
        sample = rng.integers(0, len(values), size=len(values))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model = KMeans(
                n_clusters=config.analysis.kmeans_k,
                n_init=10,
                random_state=config.analysis.random_seed + iteration + 1,
            ).fit(values[sample])
        predicted = model.predict(values)
        stability_rows.append(
            {
                "bootstrap_iteration": iteration + 1,
                "adjusted_rand_index": adjusted_rand_score(
                    reference.labels_, predicted
                ),
                "unique_sampled_sites": int(np.unique(sample).size),
                "observed_cluster_count": int(
                    np.unique(model.labels_).size
                ),
            }
        )
    return assignments, centroids, pd.DataFrame.from_records(stability_rows)


def _cluster_enrichment(
    assignments: pd.DataFrame,
    sites: pd.DataFrame,
) -> pd.DataFrame:
    from scipy.stats import hypergeom

    table = assignments.merge(
        sites.loc[
            :,
            [
                "site_id",
                "paper_class",
                "DIC_label",
                "HC_DIC_label",
                "LC_DIC_label",
            ],
        ],
        on="site_id",
        how="left",
        validate="one_to_one",
    )
    label_sets: dict[str, pd.Series] = {}
    for paper_class in sorted(table["paper_class"].dropna().astype(str).unique()):
        label_sets[f"paper_class:{paper_class}"] = (
            table["paper_class"].astype(str).eq(paper_class)
        )
    for column in ("DIC_label", "HC_DIC_label", "LC_DIC_label"):
        label_sets[column] = _as_bool(table[column])
    rows = []
    total = len(table)
    for cluster, cluster_rows in table.groupby("cluster", sort=True):
        cluster_mask = table["cluster"].eq(cluster)
        for label, positive in label_sets.items():
            observed = int((cluster_mask & positive).sum())
            population_positive = int(positive.sum())
            expected = len(cluster_rows) * population_positive / total
            rows.append(
                {
                    "cluster": cluster,
                    "label": label,
                    "cluster_size": len(cluster_rows),
                    "observed_positive": observed,
                    "expected_positive": expected,
                    "fold_enrichment": (
                        observed / expected if expected > 0 else np.nan
                    ),
                    "hypergeometric_pvalue": hypergeom.sf(
                        observed - 1,
                        total,
                        population_positive,
                        len(cluster_rows),
                    ),
                }
            )
    return pd.DataFrame.from_records(rows)


def _ablation_report(
    values: np.ndarray,
    columns: Sequence[str],
    qc: pd.DataFrame,
    config: RunConfig,
) -> pd.DataFrame:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    family_by_column = qc.set_index("feature_name")["feature_family"].to_dict()
    families = sorted({family_by_column[column] for column in columns})
    rows = []
    selections = [("all", np.ones(len(columns), dtype=bool))]
    for family in families:
        selections.append(
            (
                f"only:{family}",
                np.asarray(
                    [family_by_column[column] == family for column in columns]
                ),
            )
        )
        selections.append(
            (
                f"without:{family}",
                np.asarray(
                    [family_by_column[column] != family for column in columns]
                ),
            )
        )
    for name, mask in selections:
        if mask.sum() < 2:
            rows.append(
                {
                    "feature_set": name,
                    "feature_count": int(mask.sum()),
                    "silhouette_score": np.nan,
                    "status": "insufficient_features",
                }
            )
            continue
        labels = KMeans(
            n_clusters=config.analysis.kmeans_k,
            n_init=10,
            random_state=config.analysis.random_seed,
        ).fit_predict(values[:, mask])
        rows.append(
            {
                "feature_set": name,
                "feature_count": int(mask.sum()),
                "silhouette_score": silhouette_score(values[:, mask], labels),
                "status": "ok",
            }
        )
    return pd.DataFrame.from_records(rows)


def _chromosome_groups(series: pd.Series) -> np.ndarray:
    return (
        series.astype(str)
        .str.replace(r"^chr", "", regex=True)
        .str.replace(r"[^0-9XYM].*$", "", regex=True)
        .to_numpy()
    )


def _splits(
    sites: pd.DataFrame,
    y: np.ndarray,
    config: RunConfig,
) -> tuple[str, list[tuple[np.ndarray, np.ndarray]]]:
    chromosome = (
        sites["chrom"].astype(str).str.replace(r"^chr", "", regex=True)
    )
    numeric = pd.to_numeric(chromosome, errors="coerce")
    holdout = numeric.between(
        config.analysis.chromosome_holdout_start,
        config.analysis.chromosome_holdout_end,
    ).to_numpy()
    if (
        holdout.any()
        and (~holdout).any()
        and np.unique(y[holdout]).size == 2
        and np.unique(y[~holdout]).size == 2
    ):
        return "chr16_22_holdout", [
            (np.flatnonzero(~holdout), np.flatnonzero(holdout))
        ]
    groups = _chromosome_groups(sites["chrom"])
    unique_groups = np.unique(groups)
    folds = min(config.analysis.grouped_cv_folds, len(unique_groups))
    if folds < 2:
        return "unavailable", []
    try:
        from sklearn.model_selection import StratifiedGroupKFold

        splitter = StratifiedGroupKFold(
            n_splits=folds,
            shuffle=True,
            random_state=config.analysis.random_seed,
        )
        result = list(splitter.split(np.zeros(len(y)), y, groups))
        strategy = "stratified_grouped_chromosome_cv"
    except (ImportError, ValueError):
        from sklearn.model_selection import GroupKFold

        splitter = GroupKFold(n_splits=folds)
        result = list(splitter.split(np.zeros(len(y)), y, groups))
        strategy = "grouped_chromosome_cv"
    valid = [
        (train, test)
        for train, test in result
        if np.unique(y[train]).size == 2 and np.unique(y[test]).size == 2
    ]
    return strategy, valid


def _model_factories(seed: int) -> dict[str, Any]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    return {
        "logistic": lambda: LogisticRegression(
            max_iter=5000, class_weight="balanced", random_state=seed
        ),
        "l1_logistic": lambda: LogisticRegression(
            penalty="l1",
            solver="liblinear",
            max_iter=5000,
            class_weight="balanced",
            random_state=seed,
        ),
        "svm_rbf": lambda: SVC(
            class_weight="balanced", probability=True, random_state=seed
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
    }


def _benchmark(
    raw: pd.DataFrame,
    columns: Sequence[str],
    sites: pd.DataFrame,
    qc: pd.DataFrame,
    registry: pd.DataFrame,
    config: RunConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.preprocessing import RobustScaler

    joined = sites.reset_index(drop=True)
    intragenic = _as_bool(joined["intragenic_flag"])
    hc = _as_bool(joined["HC_DIC_label"])
    lc = _as_bool(joined["LC_DIC_label"])
    tasks = {
        "A_intragenic_DIC": (intragenic, _as_bool(joined["DIC_label"])),
        "B_HC_DIC": (pd.Series(True, index=joined.index), hc),
        "C_LC_DIC": (pd.Series(True, index=joined.index), lc),
        "D_HC_vs_LC": (hc | lc, hc),
    }
    family = qc.set_index("feature_name")["feature_family"].to_dict()
    registry_index = registry.drop_duplicates("feature_name").set_index(
        "feature_name"
    )
    feature_sets = {
        "biological_fingerprint": list(columns),
        "leakage_control": [
            column
            for column in columns
            if bool(
                registry_index.get(
                    "leakage_control_feature", pd.Series(dtype=bool)
                ).get(_base_feature(column), False)
            )
        ],
    }
    metric_rows = []
    importance_rows = []
    factories = _model_factories(config.analysis.random_seed)
    for task_name, (eligible, labels) in tasks.items():
        eligible = eligible.to_numpy(dtype=bool)
        y = labels.loc[eligible].to_numpy(dtype=int)
        task_sites = joined.loc[eligible].reset_index(drop=True)
        if len(y) < 10 or np.unique(y).size < 2:
            metric_rows.append(
                {
                    "task": task_name,
                    "feature_set": "all",
                    "model": "all",
                    "split_strategy": "unavailable",
                    "fold": -1,
                    "status": "insufficient_classes_or_rows",
                }
            )
            continue
        strategy, split_rows = _splits(task_sites, y, config)
        for set_name, set_columns in feature_sets.items():
            set_columns = [column for column in set_columns if column in columns]
            if not set_columns:
                metric_rows.append(
                    {
                        "task": task_name,
                        "feature_set": set_name,
                        "model": "all",
                        "split_strategy": strategy,
                        "fold": -1,
                        "status": "no_features",
                    }
                )
                continue
            x = (
                raw.loc[eligible, set_columns]
                .apply(pd.to_numeric, errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .to_numpy()
            )
            if not split_rows:
                metric_rows.append(
                    {
                        "task": task_name,
                        "feature_set": set_name,
                        "model": "all",
                        "split_strategy": strategy,
                        "fold": -1,
                        "status": "no_valid_chromosome_split",
                    }
                )
                continue
            for model_name, factory in factories.items():
                for fold, (train, test) in enumerate(split_rows):
                    imputer = SimpleImputer(strategy="median")
                    scaler = RobustScaler()
                    x_train = scaler.fit_transform(
                        imputer.fit_transform(x[train])
                    )
                    x_test = scaler.transform(imputer.transform(x[test]))
                    y_train = y[train]
                    if config.analysis.enable_smote:
                        try:
                            from imblearn.over_sampling import SMOTE
                        except ImportError as exc:
                            raise AnalysisError(
                                "analysis.enable_smote=true requires "
                                "imbalanced-learn"
                            ) from exc
                        minority_count = int(
                            np.bincount(y_train).min()
                        )
                        if minority_count < 2:
                            raise AnalysisError(
                                "SMOTE requires at least two minority-class "
                                "training rows in every chromosome split"
                            )
                        x_train, y_train = SMOTE(
                            random_state=config.analysis.random_seed,
                            k_neighbors=min(5, minority_count - 1),
                        ).fit_resample(x_train, y_train)
                    model = factory()
                    model.fit(x_train, y_train)
                    probability = model.predict_proba(x_test)[:, 1]
                    predicted = model.predict(x_test)
                    metric_rows.append(
                        {
                            "task": task_name,
                            "feature_set": set_name,
                            "model": model_name,
                            "split_strategy": strategy,
                            "fold": fold,
                            "train_n": len(train),
                            "test_n": len(test),
                            "positive_test_n": int(y[test].sum()),
                            "auroc": roc_auc_score(y[test], probability),
                            "auprc": average_precision_score(
                                y[test], probability
                            ),
                            "f1": f1_score(y[test], predicted, zero_division=0),
                            "recall": recall_score(
                                y[test], predicted, zero_division=0
                            ),
                            "precision": precision_score(
                                y[test], predicted, zero_division=0
                            ),
                            "balanced_accuracy": balanced_accuracy_score(
                                y[test], predicted
                            ),
                            "status": "ok",
                        }
                    )
                    importance = getattr(model, "coef_", None)
                    if importance is not None:
                        importance = np.ravel(importance)
                    else:
                        importance = getattr(
                            model, "feature_importances_", None
                        )
                    if importance is not None:
                        for column, value in zip(set_columns, importance):
                            importance_rows.append(
                                {
                                    "task": task_name,
                                    "feature_set": set_name,
                                    "model": model_name,
                                    "fold": fold,
                                    "feature_name": column,
                                    "feature_family": family.get(
                                        column, "unregistered"
                                    ),
                                    "importance": float(value),
                                }
                            )
    metric_columns = [
        "task",
        "feature_set",
        "model",
        "split_strategy",
        "fold",
        "train_n",
        "test_n",
        "positive_test_n",
        "auroc",
        "auprc",
        "f1",
        "recall",
        "precision",
        "balanced_accuracy",
        "status",
    ]
    importance_columns = [
        "task",
        "feature_set",
        "model",
        "fold",
        "feature_name",
        "feature_family",
        "importance",
    ]
    metrics = pd.DataFrame.from_records(
        metric_rows, columns=metric_columns
    )
    if not metrics.empty and "status" in metrics:
        ok = metrics.loc[metrics["status"].eq("ok")]
        score_columns = [
            "auroc",
            "auprc",
            "f1",
            "recall",
            "precision",
            "balanced_accuracy",
        ]
        if not ok.empty:
            aggregate = (
                ok.groupby(
                    ["task", "feature_set", "model", "split_strategy"],
                    sort=True,
                )[score_columns]
                .mean()
                .reset_index()
            )
            aggregate["fold"] = "mean"
            aggregate["status"] = "aggregate"
            metrics = pd.concat([metrics, aggregate], ignore_index=True)
    importance = pd.DataFrame.from_records(
        importance_rows, columns=importance_columns
    )
    if not importance.empty:
        aggregate_importance = (
            importance.groupby(
                [
                    "task",
                    "feature_set",
                    "model",
                    "feature_name",
                    "feature_family",
                ],
                sort=True,
            )["importance"]
            .mean()
            .reset_index()
        )
        aggregate_importance["fold"] = "mean"
        importance = pd.concat(
            [importance, aggregate_importance], ignore_index=True
        )
    return metrics, importance


def _elastic_net_report(
    raw: pd.DataFrame,
    columns: Sequence[str],
    sites: pd.DataFrame,
    config: RunConfig,
) -> pd.DataFrame:
    from scipy.stats import linregress
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import ElasticNet
    from sklearn.metrics import mean_squared_error
    from sklearn.preprocessing import RobustScaler

    y_all = pd.to_numeric(
        sites["Rad21_Mvalue_E2_vs_Ctrl"], errors="coerce"
    )
    eligible = _as_bool(sites["intragenic_flag"]) & y_all.notna()
    if eligible.sum() < 10:
        return pd.DataFrame.from_records(
            [{"status": "insufficient_intragenic_mvalue_rows"}]
        )
    y = y_all.loc[eligible].to_numpy(dtype=float)
    subset_sites = sites.loc[eligible].reset_index(drop=True)
    x_raw = (
        raw.loc[eligible, columns]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .to_numpy()
    )
    _, split_rows = _splits(
        subset_sites, (y < np.nanmedian(y)).astype(int), config
    )
    if not split_rows:
        return pd.DataFrame.from_records(
            [{"status": "no_valid_grouped_chromosome_split"}]
        )
    alphas = np.logspace(-4, 1, 30)
    scores = []
    for alpha in alphas:
        fold_scores = []
        for train, test in split_rows:
            imputer = SimpleImputer(strategy="median")
            scaler = RobustScaler()
            x_train = scaler.fit_transform(
                imputer.fit_transform(x_raw[train])
            )
            x_test = scaler.transform(imputer.transform(x_raw[test]))
            model = ElasticNet(
                alpha=alpha,
                l1_ratio=config.analysis.elastic_net_l1_ratio,
                max_iter=20000,
                random_state=config.analysis.random_seed,
            ).fit(x_train, y[train])
            fold_scores.append(
                mean_squared_error(y[test], model.predict(x_test))
            )
        scores.append(float(np.mean(fold_scores)))
    best_alpha = float(alphas[int(np.argmin(scores))])
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    x = scaler.fit_transform(imputer.fit_transform(x_raw))
    model = ElasticNet(
        alpha=best_alpha,
        l1_ratio=config.analysis.elastic_net_l1_ratio,
        max_iter=20000,
        random_state=config.analysis.random_seed,
    ).fit(x, y)
    rows = []
    for index, (column, coefficient) in enumerate(zip(columns, model.coef_)):
        if abs(coefficient) <= np.finfo(float).eps:
            continue
        regression = linregress(x[:, index], y)
        rows.append(
            {
                "feature_name": column,
                "elastic_net_coefficient": float(coefficient),
                "elastic_net_alpha": best_alpha,
                "elastic_net_l1_ratio": config.analysis.elastic_net_l1_ratio,
                "univariate_slope": regression.slope,
                "univariate_rvalue": regression.rvalue,
                "univariate_r2": regression.rvalue**2,
                "univariate_pvalue": regression.pvalue,
                "row_count": len(y),
                "status": "retained",
            }
        )
    if not rows:
        rows.append(
            {
                "elastic_net_alpha": best_alpha,
                "elastic_net_l1_ratio": config.analysis.elastic_net_l1_ratio,
                "row_count": len(y),
                "status": "no_nonzero_coefficients",
            }
        )
    return pd.DataFrame.from_records(rows)


def run(
    *,
    config: RunConfig,
    config_path: str | Path,
    allow_provisional_labels: bool,
    overwrite_stage: bool,
) -> None:
    """Run CP5 EDA, clustering, ablations, and paper-label benchmarks."""

    root = output_dir(config)
    names = (
        "eda_scaled_feature_matrix.tsv",
        "feature_qc_report.tsv",
        "low_signal_rate_by_feature.tsv",
        "feature_correlation_heatmap.png",
        "pca_variance_explained.tsv",
        "pca_umap_by_paper_labels.png",
        "kmeans_k10_cluster_assignment.tsv",
        "cluster_label_enrichment.tsv",
        "cluster_centroid.tsv",
        "cluster_stability.tsv",
        "feature_family_ablation_report.tsv",
        "supervised_reproduction_metrics.tsv",
        "supervised_feature_importance.tsv",
        "elastic_net_mvalue_driver_report.tsv",
    )
    outputs = [root / name for name in names]
    require_completed_stage(config, config_path, "features")
    required = [
        root / "site_metadata.tsv",
        root / "raw_signal_feature_matrix.tsv",
        root / "raw_delta_feature_matrix.tsv",
        root / "control_adjusted_feature_matrix.tsv",
        root / "feature_registry.tsv",
    ]
    for path in required:
        if not path.exists():
            raise AnalysisError(f"Required feature artifact is missing: {path}")
    if overwrite_stage:
        reject_completed_downstream_stages(config, config_path, ("report",))
    prepare_stage_outputs(outputs, overwrite_stage)

    sites = pd.read_csv(required[0], sep="\t", low_memory=False)
    statuses = set(sites["label_status"].astype(str))
    if statuses != {"canonical"} and not allow_provisional_labels:
        raise AnalysisError(
            "Analysis for provisional labels requires --allow-provisional-labels"
        )
    candidate, registry = _load_candidate_matrix(root)
    candidate = sites.loc[:, ["site_id"]].merge(
        candidate, on="site_id", how="left", validate="one_to_one"
    )
    low_rates = _low_signal_rates(root)
    qc = _feature_qc(candidate, registry, low_rates, config)
    included = qc.loc[qc["included_in_eda"], "feature_name"].tolist()
    if len(included) < 2:
        raise AnalysisError(
            f"Feature QC retained fewer than two features: {len(included)}"
        )
    values, _, _ = _impute_scale(candidate, included)
    scaled = pd.DataFrame(values, columns=included)
    scaled.insert(0, "site_id", sites["site_id"].to_numpy())
    scaled.to_csv(outputs[0], sep="\t", index=False)
    qc.to_csv(outputs[1], sep="\t", index=False)
    qc.loc[
        :,
        [
            "feature_name",
            "base_feature_name",
            "feature_family",
            "low_signal_rate",
        ],
    ].to_csv(outputs[2], sep="\t", index=False)
    _plot_heatmaps(scaled, sites, qc, outputs[3])
    coordinates, _ = _embedding_outputs(
        values, included, sites, config, outputs[4], outputs[5]
    )
    assignments, centroids, stability = _cluster_outputs(
        values, included, sites, config
    )
    assignments["PC1"] = coordinates[:, 0]
    assignments["PC2"] = coordinates[:, 1]
    assignments.to_csv(outputs[6], sep="\t", index=False)
    _cluster_enrichment(assignments, sites).to_csv(
        outputs[7], sep="\t", index=False
    )
    centroids.to_csv(outputs[8], sep="\t", index=False)
    stability.to_csv(outputs[9], sep="\t", index=False)
    _ablation_report(values, included, qc, config).to_csv(
        outputs[10], sep="\t", index=False
    )
    metrics, importance = _benchmark(
        candidate, included, sites, qc, registry, config
    )
    metrics.to_csv(outputs[11], sep="\t", index=False)
    importance.to_csv(outputs[12], sep="\t", index=False)
    _elastic_net_report(candidate, included, sites, config).to_csv(
        outputs[13], sep="\t", index=False
    )

    label_status = "canonical" if statuses == {"canonical"} else "provisional"
    record_stage(
        config=config,
        config_path=config_path,
        stage="analyze",
        inputs=required,
        outputs=outputs,
        label_status=label_status,
        metadata={
            "candidate_feature_count": len(qc),
            "retained_feature_count": len(included),
            "kmeans_k": config.analysis.kmeans_k,
            "cluster_bootstrap_iterations": (
                config.analysis.cluster_bootstrap_iterations
            ),
            "supervised_preprocessing": "fit_within_chromosome_split",
            "contact_response_included": False,
            "umap_enabled": config.analysis.enable_umap,
            "smote_enabled": config.analysis.enable_smote,
        },
    )
