# modules/superspreaders.py
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
from itertools import combinations
import os

def _read_csv_flex(path):
    """Robust CSV loader returning DataFrame or None."""
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, dtype=str)

def build_interaction_graph(posts_csv="output/raw_posts.csv",
                            comments_csv="output/comments_raw.csv",
                            min_edge_weight=1):
    """
    Build interaction graph using:
      - edges from post author <-> commenter (commenter interacts with post author)
      - co-comment edges: commenters on same post are connected (co-discussion)
    Returns a NetworkX Graph (undirected, weighted).
    """
    posts_df = _read_csv_flex(posts_csv)
    comments_df = _read_csv_flex(comments_csv)

    if posts_df is None:
        raise FileNotFoundError(f"Posts file not found: {posts_csv}")
    if comments_df is None:
        raise FileNotFoundError(f"Comments file not found: {comments_csv}")

    # --- Detect post id/permalink column and post author column robustly ---
    post_id_col = None
    post_author_col = None
    for c in posts_df.columns:
        lc = c.lower()
        if lc in ("id", "post_id", "link_id", "name", "permalink", "url"):
            post_id_col = post_id_col or c
        if lc in ("author", "post_author", "username", "by"):
            post_author_col = post_author_col or c

    # If not found, try some fallback guesses
    if post_id_col is None:
        # prefer columns that look like ids or urls
        for c in posts_df.columns:
            if "id" in c.lower() or "link" in c.lower() or "perma" in c.lower() or "url" in c.lower():
                post_id_col = c
                break
    if post_author_col is None:
        for c in posts_df.columns:
            if "author" in c.lower() or "user" in c.lower() or "by" in c.lower():
                post_author_col = c
                break

    # If still missing, create placeholders (author unknown) and use title/permalink as id
    if post_author_col is None:
        posts_df["post_author"] = posts_df.get("author", "[unknown]")
        post_author_col = "post_author"
    if post_id_col is None:
        # try to build a synthetic id from permalink or url or title
        if "permalink" in posts_df.columns:
            posts_df["post_id"] = posts_df["permalink"]
            post_id_col = "post_id"
        elif "url" in posts_df.columns:
            posts_df["post_id"] = posts_df["url"]
            post_id_col = "post_id"
        else:
            posts_df["post_id"] = posts_df.index.astype(str)
            post_id_col = "post_id"

    # --- Detect comment post link/id and comment author column robustly ---
    comment_post_col = None
    for c in comments_df.columns:
        lc = c.lower()
        if lc in ("post_id", "link_id", "post", "permalink", "url"):
            comment_post_col = comment_post_col or c

    if comment_post_col is None:
        # fallback: pick the first column that looks like a link/id or the first column
        for c in comments_df.columns:
            if "post" in c.lower() or "link" in c.lower() or "perma" in c.lower() or "url" in c.lower():
                comment_post_col = c
                break
        if comment_post_col is None:
            comment_post_col = comments_df.columns[0]

    comment_author_col = None
    for c in comments_df.columns:
        if c.lower() in ("author", "comment_author", "user", "username", "by"):
            comment_author_col = c
            break
    if comment_author_col is None:
        # fallback to second column if available else first
        comment_author_col = comments_df.columns[1] if len(comments_df.columns) > 1 else comments_df.columns[0]

    # Normalize and select just necessary columns
    posts_df = posts_df[[post_id_col, post_author_col]].fillna("[deleted]").copy()
    posts_df.columns = ["post_id", "post_author"]

    comments_df = comments_df[[comment_post_col, comment_author_col]].fillna("[deleted]").copy()
    comments_df.columns = ["post_id", "comment_author"]

    # Normalize post_id values: if they are full URLs/permalinks, keep them consistent
    posts_df["post_id"] = posts_df["post_id"].astype(str).str.strip()
    comments_df["post_id"] = comments_df["post_id"].astype(str).str.strip()

    # Build undirected weighted graph
    G = nx.Graph()

    # Add nodes: all unique users
    all_users = set(posts_df["post_author"].unique()).union(set(comments_df["comment_author"].unique()))
    for u in all_users:
        G.add_node(u)

    # author <-> commenter edges
    merged = comments_df.merge(posts_df, on="post_id", how="left")
    author_comment_pairs = []
    for _, row in merged.iterrows():
        commenter = str(row["comment_author"])
        post_author = str(row["post_author"]) if pd.notna(row["post_author"]) else "[deleted]"
        # ignore if commenter missing
        if commenter == "" or commenter == "[deleted]":
            continue
        if commenter == post_author:
            continue
        author_comment_pairs.append((post_author, commenter))

    # Count and add weights
    counter_ac = Counter(author_comment_pairs)
    for (a, b), w in counter_ac.items():
        if w >= min_edge_weight:
            if G.has_edge(a, b):
                G[a][b]["weight"] += w
            else:
                G.add_edge(a, b, weight=w, type="author_comment")

    # co-comment edges: commenters who commented on same post
    for pid, group in comments_df.groupby("post_id"):
        commenters = list(group["comment_author"].unique())
        if len(commenters) < 2:
            continue
        for u, v in combinations(commenters, 2):
            if u == v:
                continue
            if G.has_edge(u, v):
                G[u][v]["weight"] += 1
            else:
                G.add_edge(u, v, weight=1, type="co_comment")

    return G

def compute_centralities(G):
    """Compute degree, betweenness, and pagerank. Returns DataFrame."""
    if G is None or G.number_of_nodes() == 0:
        return pd.DataFrame(columns=["user", "degree", "betweenness", "pagerank"])

    # degree centrality (normalized)
    deg = nx.degree_centrality(G)
    btw = nx.betweenness_centrality(G, normalized=True)
    try:
        pr = nx.pagerank(G, weight="weight")
    except Exception:
        pr = {n: 0.0 for n in G.nodes()}

    rows = []
    for n in G.nodes():
        rows.append({
            "user": n,
            "degree": float(deg.get(n, 0.0)),
            "betweenness": float(btw.get(n, 0.0)),
            "pagerank": float(pr.get(n, 0.0))
        })
    df = pd.DataFrame(rows)
    df = df.sort_values(by=["pagerank", "degree", "betweenness"], ascending=False).reset_index(drop=True)
    return df

def save_top_influencers(df_central, path="output/superspreaders.csv", top_n=50):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df_central.head(top_n).to_csv(path, index=False)
    print(f"Saved top {top_n} influencers to {path}")

def draw_graph(G, centrality_df, out_png="output/charts/superspreaders_graph.png", top_n=50):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    # pick top nodes by pagerank
    top_nodes = list(centrality_df.head(top_n)["user"])
    if not top_nodes:
        print("[WARN] No top nodes to draw.")
        return

    H = G.subgraph(top_nodes).copy()

    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(H, k=0.5, seed=42)
    pr_map = centrality_df.set_index("user")["pagerank"].to_dict()
    sizes = [5000 * pr_map.get(n, 0.001) + 100 for n in H.nodes()]
    nx.draw_networkx_nodes(H, pos, node_size=sizes, alpha=0.85)
    nx.draw_networkx_edges(H, pos, alpha=0.4)
    labels = {n: n if pr_map.get(n, 0, ) > 0.001 else "" for n in H.nodes()}
    nx.draw_networkx_labels(H, pos, labels, font_size=8)
    plt.title("Top Superspreaders (subgraph)")
    plt.axis("off")
    plt.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved graph image to {out_png}")

def detect_superspreaders(posts_csv="output/raw_posts.csv",
                          comments_csv="output/comments_raw.csv",
                          out_csv="output/superspreaders.csv",
                          out_graph="output/charts/superspreaders_graph.png",
                          top_n=50):
    """
    High-level function to run the full detection pipeline and save outputs.
    """
    print("Building interaction graph...")
    G = build_interaction_graph(posts_csv=posts_csv, comments_csv=comments_csv)
    print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print("Computing centralities...")
    df_cent = compute_centralities(G)
    save_top_influencers(df_cent, path=out_csv, top_n=top_n)
    draw_graph(G, df_cent, out_png=out_graph, top_n=top_n)
    return df_cent
