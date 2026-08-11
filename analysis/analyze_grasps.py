"""Grasp selection analysis.

- Feature correlation with cuRobo success
- Logistic regression re-ranking model
- Comparison: discriminator order vs learned re-ranking.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "order_experiment_data"
OUTPUT_DIR = Path(__file__).resolve().parent / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SCENARIOS = ["in_basket", "on_shelf", "on_table"]
FEATURE_NAMES = [
    "discriminator_score",
    "distance",
    "horizontal_angle_diff",
    "up_vector",
    "collision_free",
]
FEATURE_LABELS = {
    "discriminator_score": "Discriminator Score",
    "distance": "Distance from Base (m)",
    "horizontal_angle_diff": "Approach Angle Diff (rad)",
    "up_vector": "Approach Vector Z",
    "collision_free": "Collision-Free (GraspGen)",
}


def load_all_data() -> list[dict[str, object]]:
    """Load grasp records from all scenario JSON files.

    Returns:
        list[dict[str, object]]: Flat list of grasp records.
    """
    records = []
    for scenario in SCENARIOS:
        for i in range(1, 11):
            path = DATA_ROOT / scenario / "result_data" / f"{i}.json"
            with open(path) as f:
                data = json.load(f)
            for _rank, grasp in enumerate(data["grasps"]):
                records.append(
                    {
                        "scenario": scenario,
                        "file": i,
                        "discriminator_score": grasp["discriminator_score"],
                        "motion_plan_time": grasp["motion_plan_time"],
                        "distance": grasp["distance"],
                        "horizontal_angle_diff": grasp["horizontal_angle_diff"],
                        "up_vector": grasp["up_vector"],
                        "collision_free": 0
                        if grasp["collision_detected_by_graspgen"]
                        else 1,
                        "success": 1 if grasp["curobo_success"] == "Success" else 0,
                    }
                )
    return records


def plot_feature_distributions(records: list[dict[str, object]]) -> None:
    """Box plots: feature distributions for success vs fail grasps.

    Args:
        records (list[dict[str, object]]): Grasp records with features and labels.
    """
    success = [r for r in records if r["success"] == 1]
    fail = [r for r in records if r["success"] == 0]

    fig, axes = plt.subplots(1, len(FEATURE_NAMES), figsize=(18, 4))
    fig.suptitle(
        "Feature Distributions: Success vs Fail Grasps", fontsize=13, fontweight="bold"
    )

    for ax, feat in zip(axes, FEATURE_NAMES, strict=True):
        s_vals = [r[feat] for r in success]
        f_vals = [r[feat] for r in fail]
        bp = ax.boxplot([s_vals, f_vals], labels=["Success", "Fail"], patch_artist=True)
        bp["boxes"][0].set_facecolor("#4CAF50")
        bp["boxes"][1].set_facecolor("#F44336")
        ax.set_title(FEATURE_LABELS[feat], fontsize=9)
        ax.set_ylabel("Value")

        # Point-biserial correlation
        labels = [r["success"] for r in records]
        vals = [r[feat] for r in records]
        corr, pval = stats.pointbiserialr(labels, vals)
        ax.set_xlabel(f"r = {corr:.3f}, p = {pval:.3f}", fontsize=8)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "feature_distributions.png", dpi=150)
    print(f"Saved: {OUTPUT_DIR / 'feature_distributions.png'}")
    plt.close()


def plot_success_rate_by_bin(records: list[dict[str, object]]) -> None:
    """Success rate across binned feature values.

    Args:
        records (list[dict[str, object]]): Grasp records with features and labels.
    """
    fig, axes = plt.subplots(1, len(FEATURE_NAMES), figsize=(18, 4))
    fig.suptitle(
        "Success Rate by Feature Value (binned)", fontsize=13, fontweight="bold"
    )

    for ax, feat in zip(axes, FEATURE_NAMES, strict=True):
        vals = np.array([r[feat] for r in records])
        labels = np.array([r["success"] for r in records])

        if feat == "collision_free":
            bins = [-0.5, 0.5, 1.5]
            bin_labels = ["Colliding", "Free"]
        else:
            bins = np.percentile(vals, np.linspace(0, 100, 6))
            bins = np.unique(bins)
            bin_labels = [
                f"{bins[i]:.2f}-{bins[i + 1]:.2f}" for i in range(len(bins) - 1)
            ]

        rates, counts = [], []
        for i in range(len(bins) - 1):
            mask = (vals >= bins[i]) & (vals < bins[i + 1])
            if mask.sum() == 0:
                rates.append(0)
                counts.append(0)
            else:
                rates.append(labels[mask].mean())
                counts.append(mask.sum())

        colors = ["#4CAF50" if r > np.mean(rates) else "#F44336" for r in rates]
        bars = ax.bar(
            range(len(rates)),
            [r * 100 for r in rates],
            color=colors,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.set_xticks(range(len(bin_labels)))
        ax.set_xticklabels(bin_labels, rotation=30, ha="right", fontsize=7)
        ax.set_title(FEATURE_LABELS[feat], fontsize=9)
        ax.set_ylabel("Success Rate (%)")
        ax.axhline(
            100 * np.mean(labels),
            color="gray",
            linestyle="--",
            linewidth=1,
            label="Overall avg",
        )

        for bar, count in zip(bars, counts, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"n={count}",
                ha="center",
                va="bottom",
                fontsize=6,
            )

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "success_rate_by_bin.png", dpi=150)
    print(f"Saved: {OUTPUT_DIR / 'success_rate_by_bin.png'}")
    plt.close()


def train_and_evaluate(
    records: list[dict[str, object]],
) -> tuple[LogisticRegression, StandardScaler, XGBClassifier]:
    """Train logistic regression and XGBoost, evaluate with cross-validation.

    ROC-AUC is evaluated with leave-one-scene-out CV (each fold holds out all
    200 grasps of one scene), so the model is always scored on a scene it has
    never seen — matching how topk_table evaluates. A fold whose held-out scene
    has no successful grasp yields an undefined AUC and is skipped (NaN).

    Args:
        records (list[dict[str, object]]): Grasp records with features and labels.

    Returns:
        tuple[LogisticRegression, StandardScaler, XGBClassifier]: Trained models
            and scaler.
    """
    X = np.array([[r[f] for f in FEATURE_NAMES] for r in records])  # noqa: N806
    y = np.array([r["success"] for r in records])
    groups = np.array([f"{r['scenario']}/{r['file']}" for r in records])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # noqa: N806

    class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y)
    cw_dict = {0: class_weights[0], 1: class_weights[1]}
    scale_pos_weight = class_weights[1] / class_weights[0]

    lr = LogisticRegression(class_weight=cw_dict, max_iter=1000, random_state=42)
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )

    cv = LeaveOneGroupOut()

    lr_auc = cross_val_score(
        lr, X_scaled, y, cv=cv, groups=groups, scoring="roc_auc", error_score=np.nan
    )
    xgb_auc = cross_val_score(
        xgb, X, y, cv=cv, groups=groups, scoring="roc_auc", error_score=np.nan
    )
    lr_ap = cross_val_score(
        lr,
        X_scaled,
        y,
        cv=cv,
        groups=groups,
        scoring="average_precision",
        error_score=np.nan,
    )
    xgb_ap = cross_val_score(
        xgb, X, y, cv=cv, groups=groups, scoring="average_precision", error_score=np.nan
    )
    n_valid = np.count_nonzero(~np.isnan(lr_auc))
    if n_valid < len(lr_auc):
        print(
            f"\nNote: {len(lr_auc) - n_valid} of {len(lr_auc)} held-out scenes "
            "have no successful grasp; their folds are excluded from the AUC."
        )

    from sklearn.metrics import average_precision_score, roc_auc_score

    disc_scores = X[:, FEATURE_NAMES.index("discriminator_score")]
    disc_auc = roc_auc_score(y, disc_scores)
    disc_ap = average_precision_score(y, disc_scores)
    print(f"\nNo-skill PR-AUC baseline (positive rate): {y.mean():.3f}")
    print("\nDiscriminator Score (no model)")
    print(f"  ROC-AUC: {disc_auc:.3f}")
    print(f"  PR-AUC : {disc_ap:.3f}")

    print("\nLogistic Regression (leave-one-scene-out CV)")
    print(f"  ROC-AUC: {np.nanmean(lr_auc):.3f} ± {np.nanstd(lr_auc):.3f}")
    print(f"  PR-AUC : {np.nanmean(lr_ap):.3f} ± {np.nanstd(lr_ap):.3f}")
    lr.fit(X_scaled, y)
    print("  Feature coefficients:")
    for name, coef in zip(FEATURE_NAMES, lr.coef_[0], strict=True):
        print(f"    {FEATURE_LABELS[name]:35s}: {coef:+.3f}")

    print("\nXGBoost (leave-one-scene-out CV)")
    print(f"  ROC-AUC: {np.nanmean(xgb_auc):.3f} ± {np.nanstd(xgb_auc):.3f}")
    print(f"  PR-AUC : {np.nanmean(xgb_ap):.3f} ± {np.nanstd(xgb_ap):.3f}")
    xgb.fit(X, y)
    print("  Feature importances:")
    for name, imp in zip(FEATURE_NAMES, xgb.feature_importances_, strict=True):
        print(f"    {FEATURE_LABELS[name]:35s}: {imp:.3f}")

    return lr, scaler, xgb


def topk_table(
    records: list[dict[str, object]],
    lr: LogisticRegression,
    scaler: StandardScaler,
    xgb: XGBClassifier,
) -> None:
    """Rank grasps per strategy and report top-k success rates.

    Strategies: random ordering (averaged over N_RANDOM shuffles per scene),
    discriminator score, a collision-free-first heuristic (ties broken by
    discriminator score), and the two learned models.
    Report: % of scenes with at least one success in top-k, for k = 1,3,5,10,20.
    Also report mean attempts and mean time to first success.

    Args:
        records (list[dict[str, object]]): Grasp records with features and labels.
        lr (LogisticRegression): Trained logistic regression model.
        scaler (StandardScaler): Fitted feature scaler.
        xgb (XGBClassifier): Trained XGBoost model.
    """
    # Group by scene key
    scenes = {}
    for r in records:
        key = (r["scenario"], r["file"])
        scenes.setdefault(key, []).append(r)
    scene_keys = list(scenes.keys())

    K_VALUES = [1, 3, 5, 10, 20]  # noqa: N806
    STRATEGIES = ["rand", "disc", "cfd", "lr", "xgb"]  # noqa: N806
    STRATEGY_LABELS = {  # noqa: N806
        "rand": "Random",
        "disc": "Discriminator",
        "cfd": "CollFree+Disc",
        "lr": "Logistic Reg.",
        "xgb": "XGBoost",
    }
    N_RANDOM = 1000  # noqa: N806
    rng = np.random.default_rng(42)
    results = {k: {s: [] for s in STRATEGIES} for k in K_VALUES}
    attempts = {s: [] for s in STRATEGIES}
    times = {s: [] for s in STRATEGIES}

    def first_success(gs: list[dict[str, object]]) -> int:
        for i, g in enumerate(gs):
            if g["success"]:
                return i + 1
        return len(gs) + 1

    def time_to_first_success(gs: list[dict[str, object]]) -> float:
        total = 0.0
        for g in gs:
            total += float(g["motion_plan_time"])  # type: ignore[reportArgumentType]
            if g["success"]:
                return total
        return total

    # Leave-one-scene-out: train on all other scenes, evaluate on held-out scene
    for held_out_key in scene_keys:
        train_records = [
            r for r in records if (r["scenario"], r["file"]) != held_out_key
        ]
        test_grasps = scenes[held_out_key]

        X_train = np.array([[r[f] for f in FEATURE_NAMES] for r in train_records])  # noqa: N806
        y_train = np.array([r["success"] for r in train_records])
        X_test = np.array([[r[f] for f in FEATURE_NAMES] for r in test_grasps])  # noqa: N806

        # Logistic regression (needs scaling)
        scaler_cv = StandardScaler().fit(X_train)
        X_train_s = scaler_cv.transform(X_train)  # noqa: N806
        X_test_s = scaler_cv.transform(X_test)  # noqa: N806
        cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
        clf_lr = LogisticRegression(
            class_weight={0: cw[0], 1: cw[1]}, max_iter=1000, random_state=42
        )
        clf_lr.fit(X_train_s, y_train)
        lr_scores = clf_lr.predict_proba(X_test_s)[:, 1]

        # XGBoost (no scaling needed)
        spw = cw[1] / cw[0]
        clf_xgb = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            scale_pos_weight=spw,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        )
        clf_xgb.fit(X_train, y_train)
        xgb_scores = clf_xgb.predict_proba(X_test)[:, 1]

        grasps = test_grasps
        orders = {
            "disc": sorted(grasps, key=lambda g: -g["discriminator_score"]),
            "cfd": sorted(
                grasps,
                key=lambda g: (-g["collision_free"], -g["discriminator_score"]),
            ),
            "lr": [
                g
                for _, g in sorted(
                    zip(lr_scores, grasps, strict=True), key=lambda x: -x[0]
                )
            ],
            "xgb": [
                g
                for _, g in sorted(
                    zip(xgb_scores, grasps, strict=True), key=lambda x: -x[0]
                )
            ],
        }

        for name, ordered in orders.items():
            for k in K_VALUES:
                results[k][name].append(int(any(g["success"] for g in ordered[:k])))
            attempts[name].append(first_success(ordered))
            times[name].append(time_to_first_success(ordered))

        # Random baseline: expectation over N_RANDOM shuffles of this scene
        rand_hits = dict.fromkeys(K_VALUES, 0)
        rand_attempts, rand_times = [], []
        shuffled = list(grasps)
        for _ in range(N_RANDOM):
            rng.shuffle(shuffled)
            for k in K_VALUES:
                rand_hits[k] += any(g["success"] for g in shuffled[:k])
            rand_attempts.append(first_success(shuffled))
            rand_times.append(time_to_first_success(shuffled))
        for k in K_VALUES:
            results[k]["rand"].append(rand_hits[k] / N_RANDOM)
        attempts["rand"].append(float(np.mean(rand_attempts)))
        times["rand"].append(float(np.mean(rand_times)))

    COLORS = {"disc": "#2196F3", "lr": "#FF9800", "xgb": "#4CAF50"}  # noqa: N806
    LABELS = {"disc": "Discriminator", "lr": "Logistic Reg.", "xgb": "XGBoost"}  # noqa: N806

    print("\n--- Top-K Success Rate Table ---")
    header = "".join(f"{STRATEGY_LABELS[s]:>16}" for s in STRATEGIES)
    print(f"{'k':<6}{header}")
    print("-" * (6 + 16 * len(STRATEGIES)))
    for k in K_VALUES:
        row = "".join(f"{100 * np.mean(results[k][s]):>15.1f}%" for s in STRATEGIES)
        print(f"{k:<6}{row}")

    print("\nMean attempts to first success:")
    for s in STRATEGIES:
        print(f"  {STRATEGY_LABELS[s]:14s}: {np.mean(attempts[s]):.1f}")

    print("\nMean time to first success (s):")
    for s in STRATEGIES:
        print(f"  {STRATEGY_LABELS[s]:14s}: {np.mean(times[s]):.2f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle(
        "Grasp Selection: Discriminator vs Logistic Regression vs XGBoost",
        fontsize=13,
        fontweight="bold",
    )

    # Subplot 1: Top-K success rate
    ax = axes[0]
    x = np.arange(len(K_VALUES))
    width = 0.25
    for i, key in enumerate(["disc", "lr", "xgb"]):
        rates = [100 * np.mean(results[k][key]) for k in K_VALUES]
        _bars = ax.bar(
            x + (i - 1) * width,
            rates,
            width,
            label=LABELS[key],
            color=COLORS[key],
            alpha=0.8,
        )
        for xi, v in enumerate(rates):
            ax.text(xi + (i - 1) * width, v + 0.5, f"{v:.0f}%", ha="center", fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Top-{k}" for k in K_VALUES])
    ax.set_ylabel("% Scenes with ≥1 Success")
    ax.set_title("Success Rate in Top-K Attempts")
    ax.legend()

    # Subplot 2: Mean attempts to first success
    ax2 = axes[1]
    attempt_means = [np.mean(attempts[s]) for s in ["disc", "lr", "xgb"]]
    bars2 = ax2.bar(
        list(LABELS.values()),
        attempt_means,
        color=list(COLORS.values()),
        alpha=0.8,
        edgecolor="black",
    )
    ax2.set_ylabel("Mean Attempts to First Success")
    ax2.set_title("Efficiency: Attempts to First Success")
    for bar, v in zip(bars2, attempt_means, strict=True):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.3,
            f"{v:.1f}",
            ha="center",
            fontweight="bold",
        )

    # Subplot 3: Mean time to first success
    ax3 = axes[2]
    time_means = [np.mean(times[s]) for s in ["disc", "lr", "xgb"]]
    bars3 = ax3.bar(
        list(LABELS.values()),
        time_means,
        color=list(COLORS.values()),
        alpha=0.8,
        edgecolor="black",
    )
    ax3.set_ylabel("Mean Time to First Success (s)")
    ax3.set_title("Efficiency: Time to First Success")
    for bar, v in zip(bars3, time_means, strict=True):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.1,
            f"{v:.2f}s",
            ha="center",
            fontweight="bold",
        )

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "topk_comparison.png", dpi=150)
    print(f"Saved: {OUTPUT_DIR / 'topk_comparison.png'}")
    plt.close()


def main() -> None:
    """Run the full grasp selection analysis pipeline."""
    print("Loading data...")
    records = load_all_data()
    print(
        f"Loaded {len(records)} grasps, "
        f"{sum(bool(r['success']) for r in records)} successes "
        f"({100 * sum(bool(r['success']) for r in records) / len(records):.1f}%)"
    )

    print("\n--- Point-biserial correlation with curobo_success ---")
    labels = np.array([r["success"] for r in records])
    for feat in FEATURE_NAMES:
        vals = np.array([r[feat] for r in records])
        corr, pval = stats.pointbiserialr(labels, vals)
        print(f"  {FEATURE_LABELS[feat]:35s}: r={corr:+.3f}  p={pval:.4f}")

    print("\nPlotting feature distributions...")
    plot_feature_distributions(records)

    print("\nPlotting success rate by bin...")
    plot_success_rate_by_bin(records)

    print("\nTraining models...")
    lr, scaler, xgb = train_and_evaluate(records)

    print("\nGenerating top-k table...")
    topk_table(records, lr, scaler, xgb)

    print("\nDone. Figures saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
