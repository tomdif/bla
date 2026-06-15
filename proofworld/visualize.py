#!/usr/bin/env python3
"""proofworld.visualize -- render the model's "picture" of a problem. Three lenses, three image artifacts:

  1. corpus_map.png  -- WHERE a problem lives. A 2D embedding (TF-IDF -> SVD -> t-SNE) of every verified corpus
                        statement, colored by domain; a query goal is placed in the map and its retrieved
                        neighbours circled. This is the model's "spatial intuition": nearby = structurally related.
  2. atlas_graph.png -- the STRATEGIC attack-map. A node-edge obstruction atlas of the sigma-law problem: the
                        problem branches into conjectured routes, each terminating in a WALL (refuted, with the
                        witness) or a SURVIVOR (kernel-verified, axiom-clean). Walls/survivors are the frontier.
  3. proof_dag.png   -- the STRUCTURE of a solution. The cited-lemma dependency DAG of a verified theorem, expanded
                        through premises that are themselves project theorems (kernel-sourced from premises.jsonl).

These are structured pictures of problem-space (statements/proofs/structures), not pixels of geometry -- the
faithful visualization for a proof world model. Run: python3 -m proofworld.visualize
"""
from __future__ import annotations
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))
VIZ = os.path.join(HERE, "viz")
CORPUS = os.path.join(HERE, "corpus", "corpus.jsonl")
PREMISES = os.path.join(HERE, "corpus", "premises.jsonl")


def load(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


# ----------------------------- 1. corpus embedding map -----------------------------
def corpus_map(query="∀ p, p.Prime → RamanujanTau.sigma3 p = (p:ℤ)^3 + 1", k=6):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.manifold import TSNE
    from sklearn.metrics.pairwise import cosine_similarity
    corpus = load(CORPUS)
    if not corpus:
        print("  (no corpus.jsonl -- skip corpus_map)"); return
    stmts = [r["statement"] for r in corpus]
    projects = [r["project"] for r in corpus]
    docs = stmts + [query]
    vec = TfidfVectorizer(token_pattern=r"[A-Za-z_][A-Za-z0-9_]*", min_df=2)
    X = vec.fit_transform(docs)
    svd = TruncatedSVD(n_components=min(50, X.shape[1] - 1), random_state=0)
    Xr = svd.fit_transform(X)
    emb = TSNE(n_components=2, init="pca", random_state=0,
               perplexity=min(30, len(docs) - 1)).fit_transform(Xr)
    qi = len(docs) - 1
    sims = cosine_similarity(Xr[qi:qi + 1], Xr[:qi])[0]
    nn = np.argsort(-sims)[:k]
    fig, ax = plt.subplots(figsize=(12, 9))
    for proj, c in [("RamanujanTau", "#1f77b4"), ("PlonkLean", "#ff7f0e")]:
        idx = [i for i, p in enumerate(projects) if p == proj]
        ax.scatter(emb[idx, 0], emb[idx, 1], s=14, c=c, alpha=0.45, label=f"{proj} ({len(idx)})", linewidths=0)
    ax.scatter(emb[nn, 0], emb[nn, 1], s=130, facecolors="none", edgecolors="crimson",
               linewidths=1.8, label=f"retrieved neighbours (k={k})")
    ax.scatter(emb[qi, 0], emb[qi, 1], s=320, marker="*", c="red", edgecolors="black",
               linewidths=0.8, label="query goal", zorder=5)
    for i in nn[:4]:
        ax.annotate(corpus[i]["name"].split(".")[-1], (emb[i, 0], emb[i, 1]),
                    fontsize=7, alpha=0.8, xytext=(4, 4), textcoords="offset points")
    ax.set_title("proofworld corpus map — where the problem lives\n(TF-IDF→SVD→t-SNE of 773 verified statements; nearby = structurally related)",
                 fontsize=11)
    ax.legend(loc="best", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); out = os.path.join(VIZ, "corpus_map.png"); fig.savefig(out, dpi=130); plt.close(fig)
    print(f"  wrote {os.path.relpath(out, HERE)}  (query placed; {k} neighbours circled)")


# ----------------------------- 2. obstruction-atlas graph -----------------------------
def atlas_graph():
    """the sigma-law problem as a route/wall/survivor atlas (grounded in conjecture.py's verified outcomes)."""
    G = nx.DiGraph()
    nodes = {  # id: (label, kind)
        "P":  ("σ_k(n) problem\n(divisor-power sums)", "problem"),
        "C1": ("route: σ_k(n)=n^k+1\nfor ALL n", "route"),
        "C2": ("route: σ_k(p)=p^k+1\nfor PRIME p", "route"),
        "C3": ("route: geometric-series\nclosed form (LLM)", "route"),
        "W1": ("WALL: refuted by data\nσ₃(4)=73 ≠ 65", "wall"),
        "W2": ("WALL: refuted by data\nσ₃(2)=9 ≠ 2", "wall"),
        "S":  ("SURVIVOR (kernel-verified,\naxiom-clean): σ_{3,7,9,11}(p)=p^k+1", "survivor"),
    }
    for nid, (lab, kind) in nodes.items():
        G.add_node(nid, label=lab, kind=kind)
    for a, b in [("P", "C1"), ("P", "C2"), ("P", "C3"), ("C1", "W1"), ("C3", "W2"), ("C2", "S")]:
        G.add_edge(a, b)
    pos = {"P": (0, 0), "C1": (1, 1.1), "C2": (1, 0), "C3": (1, -1.1),
           "W1": (2, 1.1), "W2": (2, -1.1), "S": (2, 0)}
    color = {"problem": "#9467bd", "route": "#7f7f7f", "wall": "#d62728", "survivor": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(12, 7))
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowsize=18, edge_color="#888", width=1.4,
                           node_size=4800, connectionstyle="arc3,rad=0.02")
    for nid, (x, y) in pos.items():
        kind = nodes[nid][1]
        ax.add_patch(FancyBboxPatch((x - 0.42, y - 0.32), 0.84, 0.64, boxstyle="round,pad=0.02",
                     fc=color[kind], ec="black", alpha=0.85, mutation_scale=0.4, zorder=2))
        ax.text(x, y, nodes[nid][0], ha="center", va="center", fontsize=8.5,
                color="white" if kind != "route" else "black", zorder=3, fontweight="bold")
    handles = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=color[k], markersize=12, label=k)
               for k in ("problem", "route", "wall", "survivor")]
    ax.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, frameon=False)
    ax.set_title("proofworld obstruction atlas — the strategic attack-map\n(routes terminate in a WALL (refuted, with witness) or a SURVIVOR (kernel-verified))",
                 fontsize=11)
    ax.set_xlim(-0.7, 2.7); ax.set_ylim(-1.9, 1.9); ax.axis("off")
    fig.tight_layout(); out = os.path.join(VIZ, "atlas_graph.png"); fig.savefig(out, dpi=130); plt.close(fig)
    print(f"  wrote {os.path.relpath(out, HERE)}  (problem→routes→walls/survivor)")


# ----------------------------- 3. proof dependency DAG -----------------------------
def proof_dag(root_short="tau_ne_zero_iff_qExpansion_ne_zero", max_depth=3):
    pairs = load(PREMISES)
    if not pairs:
        print("  (no premises.jsonl -- skip proof_dag)"); return
    bythm = {r["theorem"]: r for r in pairs}
    short2full = {r["theorem"].split(".")[-1]: r["theorem"] for r in pairs}
    root = short2full.get(root_short) or next((r["theorem"] for r in pairs if root_short in r["theorem"]), None)
    if root is None:
        root = pairs[0]["theorem"]
    G = nx.DiGraph(); depth = {root: 0}; frontier = [root]
    while frontier:
        cur = frontier.pop()
        if depth[cur] >= max_depth or cur not in bythm:
            continue
        for prem in bythm[cur]["premises"][:6]:
            G.add_edge(cur, prem)
            if prem not in depth:
                depth[prem] = depth[cur] + 1
                if prem in bythm:                       # expand only premises that are themselves project theorems
                    frontier.append(prem)
    if root not in G:
        G.add_node(root)
    # layered layout by depth
    for n in G.nodes():
        G.nodes[n]["layer"] = depth.get(n, max_depth)
    pos = nx.multipartite_layout(G, subset_key="layer", align="vertical")
    fig, ax = plt.subplots(figsize=(13, 8))
    leaves = [n for n in G.nodes if n not in bythm]
    internal = [n for n in G.nodes if n in bythm and n != root]
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowsize=12, edge_color="#bbb", width=1.0)
    nx.draw_networkx_nodes(G, pos, nodelist=[root], node_color="#d62728", node_size=900, ax=ax, label="goal")
    nx.draw_networkx_nodes(G, pos, nodelist=internal, node_color="#1f77b4", node_size=520, ax=ax, label="verified project lemma")
    nx.draw_networkx_nodes(G, pos, nodelist=leaves, node_color="#2ca02c", node_size=300, ax=ax, alpha=0.7, label="cited lemma (leaf)")
    nx.draw_networkx_labels(G, pos, labels={n: n.split(".")[-1] for n in G.nodes}, font_size=6.5, ax=ax)
    ax.set_title(f"proofworld proof DAG — the structure of a solution\nroot: {root.split('.')[-1]}  (cited-lemma dependencies, kernel-sourced via getUsedConstants)",
                 fontsize=11)
    ax.legend(loc="upper left", fontsize=9, scatterpoints=1); ax.axis("off")
    fig.tight_layout(); out = os.path.join(VIZ, "proof_dag.png"); fig.savefig(out, dpi=130); plt.close(fig)
    print(f"  wrote {os.path.relpath(out, HERE)}  ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")


def main():
    os.makedirs(VIZ, exist_ok=True)
    print("=== proofworld.visualize :: rendering the model's picture of the problem (3 lenses) ===\n")
    corpus_map()
    atlas_graph()
    proof_dag()
    print(f"\n  3 visualizers rendered -> {os.path.relpath(VIZ, HERE)}/  (corpus_map, atlas_graph, proof_dag)")


if __name__ == "__main__":
    main()
