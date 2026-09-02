"""
ring_detection.py
=================
Graph-based fraud ring detection on the SecurePay transaction data.

Three structural fraud patterns are detected:
  1. CYCLE      — Money loops back to its origin  (A → B → C → A)
  2. FAN_IN     — Many distinct accounts all funnel into one destination
  3. LAYERING   — Sequential relay chains (A → B → C, where B is also a
                  sender and each hop is a TRANSFER or CASH_OUT), a classic
                  money-laundering structuring technique.

Usage
-----
    from ring_detection import detect_rings
    import pandas as pd

    df = pd.read_parquet("splitted_DS/val.parquet")
    results = detect_rings(df)
    print(results.head())

Or run directly:
    python ring_detection.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict

# ── constants ────────────────────────────────────────────────────────────────
ROOT             = Path(__file__).resolve().parent
FANIN_THRESHOLD  = 3        # min unique senders → flag as fan-in hub
CHAIN_MIN_LEN    = 3        # min hops for a layering chain to be flagged
CYCLE_MAX_LEN    = 8        # max cycle length to search (perf guard)


# ─────────────────────────────────────────────────────────────────────────────
def _build_graph(df: pd.DataFrame) -> nx.DiGraph:
    """Build a directed weighted multi-edge graph from transactions."""
    G = nx.MultiDiGraph()
    for _, row in df[["nameOrig", "nameDest", "amount", "isFraud"]].iterrows():
        G.add_edge(
            row["nameOrig"],
            row["nameDest"],
            weight=row["amount"],
            is_fraud=row["isFraud"],
        )
    return G


def _find_cycles(G: nx.DiGraph, length_bound: int) -> list[list]:
    """Return all simple cycles up to length_bound."""
    # Convert to simple DiGraph for cycle search (MultiDiGraph collapses fine)
    simple_G = nx.DiGraph(G)
    try:
        cycles = list(nx.simple_cycles(simple_G, length_bound=length_bound))
    except TypeError:
        # networkx < 3.1 doesn't support length_bound — brute-force with filter
        cycles = [c for c in nx.simple_cycles(simple_G) if len(c) <= length_bound]
    return cycles


def _find_fanin_hubs(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """
    Return accounts that receive money from >= threshold distinct senders.
    Scored by: unique_senders / max_unique_senders (normalised 0-1).
    """
    fanin = (
        df.groupby("nameDest")["nameOrig"]
        .nunique()
        .rename("unique_senders")
        .reset_index()
        .rename(columns={"nameDest": "account_id"})
    )
    hubs = fanin[fanin["unique_senders"] >= threshold].copy()
    max_val = hubs["unique_senders"].max() if len(hubs) else 1
    hubs["fanin_score"] = hubs["unique_senders"] / max_val
    return hubs[["account_id", "unique_senders", "fanin_score"]]


def _find_layering_chains(df: pd.DataFrame, min_len: int) -> dict[str, float]:
    """
    Detect relay chains in TRANSFER/CASH_OUT transactions.
    A chain is a path A0 → A1 → A2 → ... where each Ai both receives and
    then sends money onward (a classic smurfing / layering pattern).

    Returns: {account_id: chain_depth_score}  (score in [0, 1])
    """
    # Only high-risk transaction types
    mask   = (df["type_TRANSFER"] == 1) | (df["type_CASH_OUT"] == 1)
    df_sub = df[mask][["nameOrig", "nameDest", "amount"]].copy()

    senders   = set(df_sub["nameOrig"])
    receivers = set(df_sub["nameDest"])
    relay_nodes = senders & receivers          # accounts that both send & receive

    if not relay_nodes:
        return {}

    # Build adjacency for relay-only nodes
    # BFS / DFS to find the longest chain each relay node is part of
    # Build a simple DiGraph restricted to relay nodes + their neighbours
    G_relay = nx.DiGraph()
    for _, row in df_sub.iterrows():
        o, d = row["nameOrig"], row["nameDest"]
        if o in relay_nodes or d in relay_nodes:
            G_relay.add_edge(o, d, weight=row["amount"])

    # Find weakly connected components, then longest path within each
    chain_depth: dict[str, int] = defaultdict(int)
    for component in nx.weakly_connected_components(G_relay):
        sub = G_relay.subgraph(component)
        if sub.number_of_nodes() < min_len:
            continue
        # dag_longest_path only works on DAGs; most chains are DAGs
        try:
            path = nx.dag_longest_path(sub)
            depth = len(path)
            if depth >= min_len:
                for node in path:
                    chain_depth[node] = max(chain_depth[node], depth)
        except nx.NetworkXUnfeasible:
            # Has a cycle — already handled by cycle detector
            pass

    if not chain_depth:
        return {}

    max_depth = max(chain_depth.values())
    return {acct: depth / max_depth for acct, depth in chain_depth.items()}


# ─────────────────────────────────────────────────────────────────────────────
def detect_rings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main entry point.

    Parameters
    ----------
    df : DataFrame with columns: nameOrig, nameDest, amount, isFraud,
         type_TRANSFER, type_CASH_OUT  (standard SecurePay schema).

    Returns
    -------
    DataFrame with columns:
        account_id   — account identifier
        ring_score   — float in [0, 1]; higher = more suspicious
        pattern      — 'CYCLE' | 'FAN_IN' | 'LAYERING' | 'CYCLE+FAN_IN' | etc.
        detail       — human-readable detail string
    """
    required = {"nameOrig", "nameDest", "amount", "isFraud",
                "type_TRANSFER", "type_CASH_OUT"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input DataFrame is missing columns: {missing}")

    scores: dict[str, dict] = defaultdict(
        lambda: {"cycle": 0.0, "fanin": 0.0, "layering": 0.0, "patterns": set(), "detail": []}
    )

    # ── 1. CYCLE detection ──────────────────────────────────────────────────
    print("  [1/3] Building transaction graph and searching for cycles...")
    G = _build_graph(df)
    cycles = _find_cycles(G, length_bound=CYCLE_MAX_LEN)
    print(f"        Cycles found: {len(cycles)}")

    if cycles:
        max_cycle_len = max(len(c) for c in cycles)
        for cycle in cycles:
            # Score: longer cycles = more sophisticated = higher score
            cycle_score = len(cycle) / max_cycle_len
            total_flow  = sum(
                G[u][v][0].get("weight", 0)
                for u, v in zip(cycle, cycle[1:] + [cycle[0]])
                if G.has_edge(u, v)
            )
            for acct in cycle:
                scores[acct]["cycle"] = max(scores[acct]["cycle"], cycle_score)
                scores[acct]["patterns"].add("CYCLE")
                scores[acct]["detail"].append(
                    f"Cycle of length {len(cycle)}, total flow ${total_flow:,.0f}"
                )

    # ── 2. FAN-IN detection ─────────────────────────────────────────────────
    print("  [2/3] Detecting fan-in hubs...")
    fanin_df = _find_fanin_hubs(df, threshold=FANIN_THRESHOLD)
    print(f"        Fan-in accounts flagged (≥{FANIN_THRESHOLD} unique senders): {len(fanin_df)}")

    for _, row in fanin_df.iterrows():
        acct = row["account_id"]
        scores[acct]["fanin"] = row["fanin_score"]
        scores[acct]["patterns"].add("FAN_IN")
        scores[acct]["detail"].append(
            f"Fan-in hub: {int(row['unique_senders'])} unique senders"
        )

    # ── 3. LAYERING / CHAIN detection ───────────────────────────────────────
    print("  [3/3] Detecting layering chains (relay nodes in TRANSFER/CASH_OUT)...")
    layer_scores = _find_layering_chains(df, min_len=CHAIN_MIN_LEN)
    print(f"        Layering accounts flagged (chain ≥{CHAIN_MIN_LEN} hops): {len(layer_scores)}")

    for acct, lscore in layer_scores.items():
        scores[acct]["layering"] = lscore
        scores[acct]["patterns"].add("LAYERING")
        scores[acct]["detail"].append(f"Layering chain participant (score={lscore:.2f})")

    # ── Combine into final DataFrame ────────────────────────────────────────
    records = []
    for acct, s in scores.items():
        # ring_score = weighted combination; cycles are rarest so weighted highest
        ring_score = min(1.0, (
            0.50 * s["cycle"] +
            0.30 * s["fanin"] +
            0.20 * s["layering"]
        ) if s["cycle"] > 0 else (
            0.60 * s["fanin"] +
            0.40 * s["layering"]
        ) if s["fanin"] > 0 else s["layering"])

        patterns = "+".join(sorted(s["patterns"])) if s["patterns"] else "NONE"
        detail   = "; ".join(s["detail"]) if s["detail"] else ""

        records.append({
            "account_id": acct,
            "ring_score": round(ring_score, 4),
            "pattern":    patterns,
            "detail":     detail,
        })

    result = (
        pd.DataFrame(records)
        .sort_values("ring_score", ascending=False)
        .reset_index(drop=True)
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
def _print_examples(results: pd.DataFrame, df_raw: pd.DataFrame, n: int = 3):
    """Print n detailed examples of flagged accounts."""
    print(f"\n{'─'*70}")
    print(f"  TOP {n} FLAGGED ACCOUNTS — DETAILED VIEW")
    print(f"{'─'*70}")

    for i, row in results.head(n).iterrows():
        acct = row["account_id"]
        print(f"\n  [{i+1}] Account : {acct}")
        print(f"       Pattern  : {row['pattern']}")
        print(f"       Score    : {row['ring_score']:.4f}")
        print(f"       Detail   : {row['detail']}")

        sent     = df_raw[df_raw["nameOrig"] == acct][["nameDest", "amount", "isFraud"]]
        received = df_raw[df_raw["nameDest"] == acct][["nameOrig", "amount", "isFraud"]]

        if not sent.empty:
            top_sent = sent.nlargest(3, "amount")
            print(f"       Sent to  :")
            for _, t in top_sent.iterrows():
                fraud_tag = " ⚠ FRAUD" if t["isFraud"] else ""
                print(f"                  → {t['nameDest']}  ${t['amount']:>12,.2f}{fraud_tag}")

        if not received.empty:
            top_recv = received.nlargest(3, "amount")
            print(f"       Received from:")
            for _, t in top_recv.iterrows():
                fraud_tag = " ⚠ FRAUD" if t["isFraud"] else ""
                print(f"                  ← {t['nameOrig']}  ${t['amount']:>12,.2f}{fraud_tag}")

    print(f"\n{'─'*70}\n")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("  SecurePay — Fraud Ring Detection")
    print("=" * 70)

    print("\nLoading val.parquet and test.parquet...")
    val  = pd.read_parquet(ROOT / "splitted_DS" / "val.parquet")
    test = pd.read_parquet(ROOT / "splitted_DS" / "test.parquet")
    df   = pd.concat([val, test], ignore_index=True)
    print(f"  Loaded {len(df):,} transactions  |  "
          f"{df['nameOrig'].nunique():,} unique senders  |  "
          f"{df['nameDest'].nunique():,} unique receivers")
    print(f"  Known fraud transactions: {df['isFraud'].sum():,} "
          f"({df['isFraud'].mean()*100:.3f}%)\n")

    print("Running ring detection...\n")
    results = detect_rings(df)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    total_flagged = len(results)
    print(f"  Total accounts flagged : {total_flagged:,}")
    print(f"  Score distribution:")
    print(f"    High   (≥0.7) : {(results.ring_score >= 0.7).sum():,}")
    print(f"    Medium (≥0.4) : {((results.ring_score >= 0.4) & (results.ring_score < 0.7)).sum():,}")
    print(f"    Low    (<0.4) : {(results.ring_score < 0.4).sum():,}")
    print()
    print("  Pattern breakdown:")
    print(results["pattern"].value_counts().to_string())

    # ── Examples ─────────────────────────────────────────────────────────────
    _print_examples(results, df, n=3)

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = ROOT / "ring_scores.csv"
    results.to_csv(out_path, index=False)
    print(f"  Saved full results → {out_path}")
    print(f"  Top 10 preview:\n")
    print(results.head(10).to_string(index=False))
    print()
