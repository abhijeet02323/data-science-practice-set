"""
Unsupervised Learning and Clustering Analysis
Dataset: UCI / scikit-learn Wine dataset (publicly available)

Install:
    pip install pandas numpy matplotlib seaborn scikit-learn

Run:
    python unsupervised_clustering_analysis.py

Outputs:
    outputs/wine_clusters.csv
    outputs/clustering_metrics.txt
    outputs/figures/elbow_method.png
    outputs/figures/silhouette_scores.png
    outputs/figures/kmeans_pca_clusters.png
    outputs/figures/hierarchical_dendrogram.png
    outputs/figures/hierarchical_pca_clusters.png

Note:
    Wine target labels are not used to train clustering. They are retained only
    for optional post-hoc comparison after clustering, which is appropriate for
    evaluating unsupervised results without leaking labels into the algorithm.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.cluster.hierarchy import linkage, dendrogram

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"
FIGURE_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")


def load_and_prepare():
    """Load public Wine data and standardize features for distance-based models."""
    bunch = load_wine()
    X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    # Keep the known wine cultivar target separate; it is not used in fitting.
    reference_labels = pd.Series(bunch.target, name="reference_class")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X, X_scaled, reference_labels, bunch.target_names


def choose_k(X_scaled, max_k=10):
    """Compute inertia and silhouette scores over candidate cluster counts."""
    inertias, silhouettes = [], []
    candidate_ks = list(range(2, max_k + 1))
    for k in candidate_ks:
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(X_scaled)
        inertias.append(model.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
    return candidate_ks, inertias, silhouettes


def save_elbow_and_silhouette(ks, inertias, silhouettes):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(ks, inertias, marker="o")
    ax.set_title("K-Means Elbow Method")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia (within-cluster sum of squares)")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "elbow_method.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(ks, silhouettes, marker="o")
    ax.set_title("Silhouette Score by Cluster Count")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Mean silhouette score")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "silhouette_scores.png", dpi=160)
    plt.close(fig)


def plot_pca(X_scaled, labels, filename, title):
    """Project scaled data to two principal components for visualization only."""
    pca = PCA(n_components=2, random_state=42)
    coordinates = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    scatter = ax.scatter(
        coordinates[:, 0], coordinates[:, 1],
        c=labels, cmap="viridis", s=42, alpha=0.85, edgecolor="black", linewidth=0.25
    )
    ax.set_title(title)
    ax.set_xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    ax.set_ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    legend = ax.legend(*scatter.legend_elements(), title="Cluster", loc="best")
    ax.add_artist(legend)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, dpi=160)
    plt.close(fig)
    return pca, coordinates


def main():
    X, X_scaled, reference_labels, target_names = load_and_prepare()
    ks, inertias, silhouettes = choose_k(X_scaled, max_k=10)
    save_elbow_and_silhouette(ks, inertias, silhouettes)

    # Select k using both the elbow shape and silhouette score. For this dataset,
    # three clusters is a useful, interpretable starting point; inspect metrics too.
    chosen_k = 3
    kmeans = KMeans(n_clusters=chosen_k, random_state=42, n_init=20)
    kmeans_labels = kmeans.fit_predict(X_scaled)

    # Hierarchical clustering provides a second view; Ward linkage tends to create
    # compact clusters and is suitable for standardized Euclidean features.
    hierarchical = AgglomerativeClustering(n_clusters=chosen_k, linkage="ward")
    hierarchical_labels = hierarchical.fit_predict(X_scaled)

    plot_pca(
        X_scaled, kmeans_labels, "kmeans_pca_clusters.png",
        "K-Means Clusters Visualized with PCA"
    )
    plot_pca(
        X_scaled, hierarchical_labels, "hierarchical_pca_clusters.png",
        "Hierarchical Clusters Visualized with PCA"
    )

    # Dendrogram for a readable subset of samples; clustering itself uses all rows.
    linked = linkage(X_scaled, method="ward")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    dendrogram(linked, truncate_mode="lastp", p=30, leaf_rotation=90,
               leaf_font_size=8, show_contracted=True, ax=ax)
    ax.set_title("Hierarchical Clustering Dendrogram (Truncated View)")
    ax.set_xlabel("Merged sample groups")
    ax.set_ylabel("Ward linkage distance")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "hierarchical_dendrogram.png", dpi=160)
    plt.close(fig)

    # Cluster profiles use original feature units for interpretability.
    result = X.copy()
    result["kmeans_cluster"] = kmeans_labels
    result["hierarchical_cluster"] = hierarchical_labels
    result["reference_class"] = reference_labels.to_numpy()
    result.to_csv(OUT_DIR / "wine_clusters.csv", index=False)

    profile = result.groupby("kmeans_cluster").mean(numeric_only=True)
    sizes = result["kmeans_cluster"].value_counts().sort_index()
    metrics = {
        "chosen_k": chosen_k,
        "kmeans_silhouette": silhouette_score(X_scaled, kmeans_labels),
        "hierarchical_silhouette": silhouette_score(X_scaled, hierarchical_labels),
        "ari_vs_reference_labels_kmeans": adjusted_rand_score(reference_labels, kmeans_labels),
        "ari_vs_reference_labels_hierarchical": adjusted_rand_score(reference_labels, hierarchical_labels),
        "inertia": kmeans.inertia_,
    }

    lines = [
        "UNSUPERVISED CLUSTERING ANALYSIS — WINE DATASET",
        "=" * 52,
        f"Number of observations: {len(X)}",
        f"Number of features: {X.shape[1]}",
        f"Selected K-Means k: {chosen_k}",
        "",
        "Candidate k diagnostics:",
        *[f"k={k}: inertia={i:.3f}, silhouette={s:.4f}"
          for k, i, s in zip(ks, inertias, silhouettes)],
        "",
        "Final metrics:",
        *[f"{key}: {value:.4f}" if isinstance(value, (float, np.floating)) else f"{key}: {value}"
          for key, value in metrics.items()],
        "",
        "K-Means cluster sizes:",
        sizes.to_string(),
        "",
        "K-Means cluster profile (original feature units):",
        profile.round(3).to_string(),
        "",
        "Interpretation note: clusters are arbitrary IDs; compare their profiles, not their numeric order.",
        "Reference classes are only used after fitting for an external comparison, not as input features."
    ]
    (OUT_DIR / "clustering_metrics.txt").write_text("\\n".join(lines), encoding="utf-8")
    print("\\n".join(lines))
    print(f"\\nSaved clustered data to: {OUT_DIR / 'wine_clusters.csv'}")
    print(f"Saved figures to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
