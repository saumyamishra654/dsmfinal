import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import community as community_louvain
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from pathlib import Path

ROOT    = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "db" / "sqlite" / "dsm.db"
OUT_DIR = ROOT / "outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# builds one row per state with mean and slope features for tele-density and GER
def build_features():
    con = sqlite3.connect(DB_PATH)

    td = pd.read_sql_query("""
        SELECT s.state_name AS state, t.year, AVG(t.tele_density) AS tele_density
        FROM tele_density t
        JOIN states s USING (state_id)
        GROUP BY s.state_name, t.year
    """, con)

    ger = pd.read_sql_query("""
        SELECT s.state_name AS state, e.year, e.ger
        FROM education_ger e
        JOIN states s USING (state_id)
        WHERE e.gender = 'Total' AND e.category = 'All Categories' AND e.ger IS NOT NULL
    """, con)
    con.close()

    def slope(series):
        y = series.values
        x = np.arange(len(y))
        if len(x) < 2:
            return np.nan
        return float(np.polyfit(x, y, 1)[0])

    td_feats = td.groupby("state").agg(
        mean_tele_density=("tele_density", "mean"),
        tele_density_slope=("tele_density", slope),
    ).reset_index()

    ger_feats = ger.groupby("state").agg(
        mean_ger=("ger", "mean"),
        ger_slope=("ger", slope),
    ).reset_index()

    features = td_feats.merge(ger_feats, on="state", how="inner")
    return features.reset_index(drop=True)


# standardizes features and runs k-means with silhouette score to pick k
def run_kmeans(features):
    X = features[["mean_tele_density", "tele_density_slope", "mean_ger", "ger_slope"]].values
    X_scaled = StandardScaler().fit_transform(X)

    best_k, best_score, best_labels = 2, -1, None
    for k in range(2, min(7, len(features))):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        score  = silhouette_score(X_scaled, labels)
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels

    features = features.copy()
    features["kmeans_cluster"] = best_labels
    print(f"K-means: best k={best_k}, silhouette={best_score:.3f}")
    return features, best_k


# builds a cosine-similarity graph and runs Louvain community detection
def run_louvain(features):
    X = features[["mean_tele_density", "tele_density_slope", "mean_ger", "ger_slope"]].values
    X_scaled = StandardScaler().fit_transform(X)

    norms = np.linalg.norm(X_scaled, axis=1, keepdims=True)
    norms[norms == 0] = 1
    X_norm = X_scaled / norms
    sim_matrix = X_norm @ X_norm.T

    G = nx.Graph()
    states = features["state"].tolist()
    G.add_nodes_from(states)
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            w = float(sim_matrix[i, j])
            if w > 0:
                G.add_edge(states[i], states[j], weight=w)

    partition = community_louvain.best_partition(G, weight="weight", random_state=42)
    features = features.copy()
    features["louvain_community"] = features["state"].map(partition)
    n_communities = features["louvain_community"].nunique()
    print(f"Louvain: {n_communities} communities detected")
    return features, G, partition


# PCA biplot of states colored by k-means cluster with feature loading arrows
def plot_pca_biplot(features):
    feat_cols = ["mean_tele_density", "tele_density_slope", "mean_ger", "ger_slope"]
    X        = features[feat_cols].values
    X_scaled = StandardScaler().fit_transform(X)

    pca      = PCA(n_components=2)
    coords   = pca.fit_transform(X_scaled)
    labels   = features["kmeans_cluster"].values
    states   = features["state"].values

    colors = cm.tab10(labels / labels.max())

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=80, edgecolors="k", linewidths=0.5)
    for i, name in enumerate(states):
        ax.annotate(name, (coords[i, 0], coords[i, 1]),
                    fontsize=7, ha="left", va="bottom",
                    xytext=(4, 3), textcoords="offset points")

    loadings = pca.components_.T
    scale    = 2.5
    for j, col in enumerate(feat_cols):
        ax.annotate("", xy=(loadings[j, 0] * scale, loadings[j, 1] * scale),
                    xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="tomato", lw=1.5))
        ax.text(loadings[j, 0] * scale * 1.1, loadings[j, 1] * scale * 1.1,
                col.replace("_", "\n"), color="tomato", fontsize=8, ha="center")

    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% variance)")
    ax.set_title("PCA Biplot — State Clustering by Tele-density & GER Features")
    ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")
    ax.axvline(0, color="grey", linewidth=0.5, linestyle=":")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "obj4_pca_biplot.png", dpi=150)
    plt.close()
    print("  Saved obj4_pca_biplot.png")


# spring-layout graph of Louvain communities with edges above median weight
def plot_louvain_graph(features, G, partition):
    communities  = features["louvain_community"].unique()
    color_map    = cm.tab10(np.linspace(0, 1, len(communities)))
    node_colors  = [color_map[partition[n]] for n in G.nodes()]

    weights      = [d["weight"] for _, _, d in G.edges(data=True)]
    threshold    = np.median(weights)
    edges_to_draw = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] >= threshold]
    edge_weights  = [G[u][v]["weight"] * 2 for u, v in edges_to_draw]

    pos = nx.spring_layout(G, weight="weight", seed=42)

    fig, ax = plt.subplots(figsize=(11, 8))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=600, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=edges_to_draw, width=edge_weights,
                           alpha=0.4, edge_color="grey", ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=7, ax=ax)
    ax.set_title("Louvain Community Detection — State Similarity Graph\n"
                 "(edges = cosine similarity ≥ median; colour = community)")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "obj4_louvain_graph.png", dpi=150)
    plt.close()
    print("  Saved obj4_louvain_graph.png")


# bar chart of mean feature values per k-means cluster
def plot_cluster_profiles(features):
    feat_cols   = ["mean_tele_density", "tele_density_slope", "mean_ger", "ger_slope"]
    feat_labels = ["Mean Tele-density", "Tele-density Slope", "Mean GER", "GER Slope"]

    profile = features.groupby("kmeans_cluster")[feat_cols].mean()
    profile_scaled = pd.DataFrame(
        StandardScaler().fit_transform(profile),
        index=profile.index, columns=feat_labels,
    )

    fig, ax = plt.subplots(figsize=(9, 4))
    x     = np.arange(len(feat_labels))
    width = 0.8 / len(profile_scaled)
    colors = cm.tab10(np.linspace(0, 1, len(profile_scaled)))

    for i, (cluster, row) in enumerate(profile_scaled.iterrows()):
        states_in = features[features["kmeans_cluster"] == cluster]["state"].tolist()
        label = f"Cluster {cluster} ({len(states_in)} states)"
        ax.bar(x + i * width, row.values, width, label=label, color=colors[i])

    ax.set_xticks(x + width * (len(profile_scaled) - 1) / 2)
    ax.set_xticklabels(feat_labels, fontsize=9)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Standardised value")
    ax.set_title("K-means Cluster Profiles (standardised features)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "obj4_cluster_profiles.png", dpi=150)
    plt.close()
    print("  Saved obj4_cluster_profiles.png")


# forward-looking gap from each laggard state to the leader community mean
def gap_analysis(features):
    comm_means    = features.groupby("louvain_community")["mean_tele_density"].mean()
    leader_comm   = comm_means.idxmax()
    laggard_comm  = comm_means.idxmin()

    leader_states  = [s for s in features[features["louvain_community"] == leader_comm]["state"].tolist()
                      if s != "Delhi"]
    laggard_states = features[features["louvain_community"] == laggard_comm]["state"].tolist()

    leader_2021 = features[features["state"].isin(leader_states)]["mean_tele_density"].mean()

    rows = []
    for _, row in features[features["state"].isin(laggard_states)].iterrows():
        current = row["mean_tele_density"]
        slope   = row["tele_density_slope"]
        if slope <= 0:
            continue
        gap = leader_2021 - current
        rows.append({
            "state":              row["state"],
            "tele_density_mean":  round(current, 1),
            "annual_growth":      round(slope, 2),
            "gap_to_leader":      round(gap, 1),
            "years_to_close_gap": round(gap / slope, 1),
        })

    return pd.DataFrame(rows).sort_values("years_to_close_gap", ascending=False)


# full feature matrix with kmeans_cluster and louvain_community columns
def get_cluster_data():
    features, _  = run_kmeans(build_features())
    features, *_ = run_louvain(features)
    return features


# returns (G, partition, pos) for building an interactive Plotly network graph
def get_louvain_graph(features):
    X        = features[["mean_tele_density", "tele_density_slope",
                          "mean_ger", "ger_slope"]].values
    X_scaled = StandardScaler().fit_transform(X)
    norms    = np.linalg.norm(X_scaled, axis=1, keepdims=True)
    norms[norms == 0] = 1
    X_norm   = X_scaled / norms
    sim      = X_norm @ X_norm.T

    G      = nx.Graph()
    states = features["state"].tolist()
    G.add_nodes_from(states)
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            w = float(sim[i, j])
            if w > 0:
                G.add_edge(states[i], states[j], weight=w)

    partition = community_louvain.best_partition(G, weight="weight", random_state=42)
    pos       = nx.spring_layout(G, weight="weight", seed=42)
    return G, partition, pos


# wrapper so dashboard can call gap_analysis without re-running clustering
def get_gap_analysis(features):
    return gap_analysis(features)


if __name__ == "__main__":
    print("Building feature matrix...")
    features = build_features()
    print(features[["state", "mean_tele_density", "tele_density_slope",
                     "mean_ger", "ger_slope"]].to_string(index=False))
    print()

    print("Running K-means...")
    features, best_k = run_kmeans(features)
    print("\nK-means cluster assignments:")
    for cluster in sorted(features["kmeans_cluster"].unique()):
        states = features[features["kmeans_cluster"] == cluster]["state"].tolist()
        print(f"  Cluster {cluster}: {', '.join(states)}")
    print()

    print("Running Louvain...")
    features, G, partition = run_louvain(features)
    print("\nLouvain community assignments:")
    for comm in sorted(features["louvain_community"].unique()):
        states = features[features["louvain_community"] == comm]["state"].tolist()
        print(f"  Community {comm}: {', '.join(states)}")
    print()

    print("Gap analysis (laggard states vs leader cluster):")
    gaps = gap_analysis(features)
    print(gaps.to_string(index=False))
    print()

    print("Generating plots...")
    plot_pca_biplot(features)
    plot_louvain_graph(features, G, partition)
    plot_cluster_profiles(features)

    print(f"\nDone. Outputs saved to {OUT_DIR}")
