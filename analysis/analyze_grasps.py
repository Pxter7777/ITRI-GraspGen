"""
Grasp selection analysis:
- Feature correlation with cuRobo success
- Logistic regression re-ranking model
- Comparison: discriminator order vs learned re-ranking
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from scipy import stats

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "order_experiment_data"
OUTPUT_DIR = Path(__file__).resolve().parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

SCENARIOS = ["in_basket", "on_shelf", "on_table"]
FEATURE_NAMES = ["distance", "horizontal_angle_diff", "up_vector", "collision_free"]
FEATURE_LABELS = {
    "distance": "Distance from Base (m)",
    "horizontal_angle_diff": "Approach Angle Diff (rad)",
    "up_vector": "Approach Vector Z",
    "collision_free": "Collision-Free (GraspGen)",
}


def load_all_data():
    records = []
    for scenario in SCENARIOS:
        for i in range(1, 6):
            path = DATA_ROOT / scenario / "result_data" / f"{i}.json"
            with open(path) as f:
                data = json.load(f)
            for rank, grasp in enumerate(data["grasps"]):
                records.append({
                    "scenario": scenario,
                    "file": i,
                    "distance": grasp["distance"],
                    "horizontal_angle_diff": grasp["horizontal_angle_diff"],
                    "up_vector": grasp["up_vector"],
                    "collision_free": 0 if grasp["collision_detected_by_graspgen"] else 1,
                    "success": 1 if grasp["curobo_success"] == "Success" else 0,
                })
    return records


def plot_feature_distributions(records):
    """Box plots: feature distributions for success vs fail grasps."""
    success = [r for r in records if r["success"] == 1]
    fail    = [r for r in records if r["success"] == 0]

    fig, axes = plt.subplots(1, len(FEATURE_NAMES), figsize=(18, 4))
    fig.suptitle("Feature Distributions: Success vs Fail Grasps", fontsize=13, fontweight="bold")

    for ax, feat in zip(axes, FEATURE_NAMES):
        s_vals = [r[feat] for r in success]
        f_vals = [r[feat] for r in fail]
        bp = ax.boxplot([s_vals, f_vals], labels=["Success", "Fail"], patch_artist=True)
        bp["boxes"][0].set_facecolor("#4CAF50")
        bp["boxes"][1].set_facecolor("#F44336")
        ax.set_title(FEATURE_LABELS[feat], fontsize=9)
        ax.set_ylabel("Value")

        # Point-biserial correlation
        labels = [r["success"] for r in records]
        vals   = [r[feat] for r in records]
        corr, pval = stats.pointbiserialr(labels, vals)
        ax.set_xlabel(f"r = {corr:.3f}, p = {pval:.3f}", fontsize=8)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "feature_distributions.png", dpi=150)
    print(f"Saved: {OUTPUT_DIR / 'feature_distributions.png'}")
    plt.close()


def plot_success_rate_by_bin(records):
    """Success rate across binned feature values."""
    fig, axes = plt.subplots(1, len(FEATURE_NAMES), figsize=(18, 4))
    fig.suptitle("Success Rate by Feature Value (binned)", fontsize=13, fontweight="bold")

    for ax, feat in zip(axes, FEATURE_NAMES):
        vals    = np.array([r[feat] for r in records])
        labels  = np.array([r["success"] for r in records])

        if feat == "collision_free":
            bins = [-0.5, 0.5, 1.5]
            bin_labels = ["Colliding", "Free"]
        else:
            bins = np.percentile(vals, np.linspace(0, 100, 6))
            bins = np.unique(bins)
            bin_labels = [f"{bins[i]:.2f}–{bins[i+1]:.2f}" for i in range(len(bins)-1)]

        rates, counts = [], []
        for i in range(len(bins) - 1):
            mask = (vals >= bins[i]) & (vals < bins[i+1])
            if mask.sum() == 0:
                rates.append(0)
                counts.append(0)
            else:
                rates.append(labels[mask].mean())
                counts.append(mask.sum())

        colors = ["#4CAF50" if r > np.mean(rates) else "#F44336" for r in rates]
        bars = ax.bar(range(len(rates)), [r * 100 for r in rates], color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(bin_labels)))
        ax.set_xticklabels(bin_labels, rotation=30, ha="right", fontsize=7)
        ax.set_title(FEATURE_LABELS[feat], fontsize=9)
        ax.set_ylabel("Success Rate (%)")
        ax.axhline(100 * np.mean(labels), color="gray", linestyle="--", linewidth=1, label="Overall avg")

        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"n={count}", ha="center", va="bottom", fontsize=6)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "success_rate_by_bin.png", dpi=150)
    print(f"Saved: {OUTPUT_DIR / 'success_rate_by_bin.png'}")
    plt.close()


def train_and_evaluate(records):
    """Train logistic regression, evaluate with cross-validation."""
    X = np.array([[r[f] for f in FEATURE_NAMES] for r in records])
    y = np.array([r["success"] for r in records])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y)
    cw_dict = {0: class_weights[0], 1: class_weights[1]}

    model = LogisticRegression(class_weight=cw_dict, max_iter=1000, random_state=42)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="roc_auc")

    print(f"\nLogistic Regression (5-fold CV)")
    print(f"  ROC-AUC: {auc_scores.mean():.3f} ± {auc_scores.std():.3f}")

    model.fit(X_scaled, y)
    print(f"\nFeature coefficients:")
    for name, coef in zip(FEATURE_NAMES, model.coef_[0]):
        print(f"  {FEATURE_LABELS[name]:35s}: {coef:+.3f}")

    return model, scaler, auc_scores


def plot_ranking_comparison(records, model, scaler):
    """
    Compare grasp selection strategies per scene:
    - Vanilla: try grasps in discriminator order (rank 0 first)
    - Learned: try grasps in model-predicted order
    Metric: how many attempts needed before first success?
    """
    # Group by scene
    scenes = {}
    for r in records:
        key = (r["scenario"], r["file"])
        scenes.setdefault(key, []).append(r)

    vanilla_attempts, learned_attempts = [], []

    for key, grasps in scenes.items():
        grasps_sorted_vanilla = sorted(grasps, key=lambda r: r["discriminator_rank"])

        X = np.array([[r[f] for f in FEATURE_NAMES] for r in grasps])
        X_scaled = scaler.transform(X)
        scores = model.predict_proba(X_scaled)[:, 1]
        grasps_sorted_learned = [g for _, g in sorted(zip(scores, grasps), key=lambda x: -x[0])]

        def attempts_to_first_success(sorted_grasps):
            for i, g in enumerate(sorted_grasps):
                if g["success"] == 1:
                    return i + 1
            return len(sorted_grasps) + 1  # never succeeded

        vanilla_attempts.append(attempts_to_first_success(grasps_sorted_vanilla))
        learned_attempts.append(attempts_to_first_success(grasps_sorted_learned))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Attempts Before First Successful Grasp", fontsize=13, fontweight="bold")

    scene_labels = [f"{s[0].replace('_',' ')}\n#{s[1]}" for s in scenes.keys()]
    x = np.arange(len(scene_labels))
    width = 0.35

    ax = axes[0]
    ax.bar(x - width/2, vanilla_attempts, width, label="Discriminator Order", color="#2196F3", alpha=0.8)
    ax.bar(x + width/2, learned_attempts, width, label="Learned Re-ranking", color="#FF9800", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(scene_labels, fontsize=7)
    ax.set_ylabel("Attempts to First Success")
    ax.legend()
    ax.set_title("Per Scene")

    ax2 = axes[1]
    ax2.bar(["Discriminator\nOrder", "Learned\nRe-ranking"],
            [np.mean(vanilla_attempts), np.mean(learned_attempts)],
            color=["#2196F3", "#FF9800"], alpha=0.8, edgecolor="black")
    ax2.set_ylabel("Mean Attempts to First Success")
    ax2.set_title("Average Across All Scenes")
    for i, v in enumerate([np.mean(vanilla_attempts), np.mean(learned_attempts)]):
        ax2.text(i, v + 0.3, f"{v:.1f}", ha="center", fontweight="bold")

    print(f"\nRanking Comparison:")
    print(f"  Vanilla (discriminator order) - mean attempts: {np.mean(vanilla_attempts):.1f}")
    print(f"  Learned re-ranking            - mean attempts: {np.mean(learned_attempts):.1f}")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "ranking_comparison.png", dpi=150)
    print(f"Saved: {OUTPUT_DIR / 'ranking_comparison.png'}")
    plt.close()


def main():
    print("Loading data...")
    records = load_all_data()
    print(f"Loaded {len(records)} grasps, {sum(r['success'] for r in records)} successes ({100*sum(r['success'] for r in records)/len(records):.1f}%)")

    print("\nPlotting feature distributions...")
    plot_feature_distributions(records)

    print("\nPlotting success rate by bin...")
    plot_success_rate_by_bin(records)

    print("\nTraining model...")
    model, scaler, auc_scores = train_and_evaluate(records)

    print("\nDone. Figures saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
