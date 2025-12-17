# --- Standard and third-party imports (PEP 8: at top of file) ---
import os
import argparse
import csv
import logging
import polars as pl
import numpy as np
import yaml
from typing import List, Sequence, Optional, Dict
import json
from collections import defaultdict, deque
from pathlib import Path
from datetime import datetime, timedelta


def _detect_time_col(header: Sequence[str]) -> Optional[str]:
    """Best-effort detection of the time/timestamp list column."""
    if "time_list" in header:
        return "time_list"
    if "timestamp_list" in header:
        return "timestamp_list"
    # fallback: first column containing both 'time' and 'list'
    for col in header:
        low = col.lower()
        if "time" in low and "list" in low:
            return col
    # last resort: any column containing 'time'
    for col in header:
        if "time" in col.lower():
            return col
    return None


# --- Abnormality generation functions ---
def perturb_rids(rid_list, level, road_id_pool, rng, strong: bool = False):
    """Replace some road IDs while keeping transitions valid when possible.

    If a road graph is available (loaded from roadmap.rel), replacements are chosen
    to preserve both (prev -> new) and (new -> next) edges.
    """
    n = len(rid_list)
    n_perturb = {"low": 1, "medium": max(1, n // 10), "high": max(1, n // 5)}.get(
        level, max(1, n // 10)
    )
    if strong:
        n_perturb = max(n_perturb, min(n, max(1, n // 3)))
    if n < 3:
        return rid_list, []

    new_rid_list = rid_list.copy()
    perturbed: List[tuple[int, str, str]] = []

    graph = globals().get("GLOBAL_GRAPH")
    if graph is not None:
        outgoing = graph["outgoing"]
        incoming = graph["incoming"]

        candidate_indices = list(range(1, n - 1))
        if not candidate_indices:
            return rid_list, []
        indices = rng.choice(
            candidate_indices,
            size=min(n_perturb, len(candidate_indices)),
            replace=False,
        )
        for idx in indices:
            prev_r = new_rid_list[idx - 1]
            next_r = new_rid_list[idx + 1]
            old = new_rid_list[idx]

            # Choose a replacement that keeps both edges valid.
            candidates = list(
                set(outgoing.get(prev_r, [])) & incoming.get(next_r, set())
            )
            candidates = [c for c in candidates if c != old]
            if not candidates:
                continue
            new = str(rng.choice(candidates))
            new_rid_list[idx] = new
            perturbed.append((idx, old, new))
        return new_rid_list, perturbed

    # Fallback: pool-based replacement (may produce invalid transitions).
    if len(road_id_pool) < 2:
        return rid_list, []
    indices = rng.choice(range(n), size=min(n_perturb, n), replace=False)
    for idx in indices:
        old = new_rid_list[idx]
        candidates = [r for r in road_id_pool if r != old]
        if not candidates:
            continue
        new = str(rng.choice(candidates))
        new_rid_list[idx] = new
        perturbed.append((idx, old, new))
    return new_rid_list, perturbed


def route_switch(rid_list, other_rid_list, level, rng, strong: bool = False):
    # Replace a segment of rid_list with a segment from other_rid_list
    # Level controls the length of the replaced segment
    # Ensure that the function has the correct context
    n = len(rid_list)
    m = len(other_rid_list)
    if n < 4 or m < 4:
        return rid_list, None, None
    seg_len = {"low": 2, "medium": 3, "high": 4}.get(level, 3)
    # strengthen by increasing segment length when requested
    if strong:
        seg_len = min(max(3, seg_len * 2), n - 1)
    if n <= seg_len or m <= seg_len:
        return rid_list, None, None
    # pick start positions; rng.integers high is exclusive so add +1
    start1 = int(rng.integers(1, n - seg_len + 1))
    start2 = int(rng.integers(1, m - seg_len + 1))
    seg2 = other_rid_list[start2 : start2 + seg_len]
    new_rid_list = rid_list[:start1] + seg2 + rid_list[start1 + seg_len :]
    return new_rid_list, (start1, start1 + seg_len), seg2


def route_switch_from_pool(rid_list, road_pool, level, rng, strong: bool = False):
    """Route-switch that draws a replacement segment from the global road pool.

    This variant is streaming-friendly (does not need another trajectory in memory).
    """
    n = len(rid_list)
    if n < 4 or len(road_pool) < 4:
        return rid_list, None, None
    seg_len = {"low": 2, "medium": 3, "high": 4}.get(level, 3)
    if strong:
        # choose a longer replacement segment for stronger anomalies
        seg_len = min(max(3, seg_len * 2), n - 1, len(road_pool))
    if n <= seg_len or len(road_pool) <= seg_len:
        return rid_list, None, None
    # pick a valid start index (1 .. n-seg_len) inclusive; add +1 to high
    start1 = int(rng.integers(1, n - seg_len + 1))
    # pick a contiguous slice from the pool (wrap if necessary)
    pool = list(road_pool)
    # when strong, pick a segment that is less likely to be local (uniform over pool)
    start2 = int(rng.integers(0, max(1, len(pool) - seg_len)))
    seg2 = pool[start2 : start2 + seg_len]
    new_rid_list = rid_list[:start1] + seg2 + rid_list[start1 + seg_len :]
    return new_rid_list, (start1, start1 + seg_len), seg2


def _build_graph_from_rel(rel_path: Path) -> dict[str, object]:
    """Build adjacency structures from roadmap.rel.

    Returns:
        Dict with keys:
            - outgoing: dict[str, list[str]]
            - incoming: dict[str, set[str]]
            - edge_set: set[tuple[str, str]]
    """
    rel_df = pl.read_csv(str(rel_path))
    if "origin_id" not in rel_df.columns or "destination_id" not in rel_df.columns:
        raise ValueError(f"Unexpected roadmap.rel schema at {rel_path}")
    outgoing: dict[str, list[str]] = defaultdict(list)
    incoming: dict[str, set[str]] = defaultdict(set)
    edge_set: set[tuple[str, str]] = set()

    for o, d in zip(rel_df["origin_id"].to_list(), rel_df["destination_id"].to_list()):
        o_s = str(int(o))
        d_s = str(int(d))
        outgoing[o_s].append(d_s)
        incoming[d_s].add(o_s)
        edge_set.add((o_s, d_s))

    return {
        "outgoing": dict(outgoing),
        "incoming": dict(incoming),
        "edge_set": edge_set,
    }


def _is_valid_walk(rids: Sequence[str], edge_set: set[tuple[str, str]]) -> bool:
    return all((a, b) in edge_set for a, b in zip(rids[:-1], rids[1:]))


def _adjust_time_list_str(time_list: str, target_len: int) -> str:
    """Adjust a comma-separated ISO8601 '...Z' time list to a desired length.

    Strategy:
      - If already the right length: return unchanged.
      - If longer: truncate.
      - If shorter: linearly interpolate between first and last timestamp.
        If parsing fails, pad by repeating the last token.
    """
    tokens = [t for t in str(time_list).split(",") if t != ""]
    if target_len <= 0:
        return ""
    if len(tokens) == target_len:
        return ",".join(tokens)
    if len(tokens) > target_len:
        return ",".join(tokens[:target_len])
    if not tokens:
        return ""
    if len(tokens) == 1:
        return ",".join(tokens + [tokens[0]] * (target_len - 1))

    try:
        start = datetime.strptime(tokens[0], "%Y-%m-%dT%H:%M:%SZ")
        end = datetime.strptime(tokens[-1], "%Y-%m-%dT%H:%M:%SZ")
        if target_len == 1:
            return tokens[0]
        total = (end - start).total_seconds()
        if total < 0:
            # Non-monotonic input; fall back to padding.
            raise ValueError("Non-monotonic time_list")
        step = total / float(target_len - 1)
        out = [
            (start + timedelta(seconds=round(step * i))).strftime("%Y-%m-%dT%H:%M:%SZ")
            for i in range(target_len)
        ]
        return ",".join(out)
    except Exception:
        return ",".join(tokens + [tokens[-1]] * (target_len - len(tokens)))


def _find_bounded_path(
    start: str,
    end: str,
    outgoing: dict[str, list[str]],
    rng: np.random.Generator,
    max_edges: int,
    forbidden_edges: Optional[set[tuple[str, str]]] = None,
) -> Optional[List[str]]:
    """Find a path from start->end using BFS limited by max_edges.

    max_edges is the maximum number of edges allowed in the returned path.
    Returns a list of nodes including start and end.
    """
    if start == end:
        return [start]
    if max_edges <= 0:
        return None

    q: deque[str] = deque([start])
    parent: dict[str, str] = {}
    depth: dict[str, int] = {start: 0}

    forbidden = forbidden_edges or set()

    while q:
        node = q.popleft()
        node_depth = depth[node]
        if node_depth >= max_edges:
            continue

        nbrs = outgoing.get(node, [])
        if not nbrs:
            continue
        # Randomize neighbor order for variety, but keep it deterministic per-row.
        if len(nbrs) > 1:
            nbrs = list(nbrs)
            rng.shuffle(nbrs)

        for nxt in nbrs:
            if (node, nxt) in forbidden:
                continue
            if nxt in depth:
                continue
            parent[nxt] = node
            depth[nxt] = node_depth + 1
            if nxt == end:
                # Reconstruct
                path = [end]
                cur = end
                while cur != start:
                    cur = parent[cur]
                    path.append(cur)
                path.reverse()
                return path
            q.append(nxt)
    return None


# NOTE: Previously a separate `make_strong_anomaly` function implemented a
# set of strong strategies. To keep the codebase simpler and avoid duplicated
# generation paths, the 'strong' behavior is integrated into the core
# generators above (insert_detour, perturb_rids, route_switch_from_pool).


ABNORMALITY_TYPES = ["detour", "route_switch", "perturb"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate abnormal trajectories for HOSER datasets."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Input directory with original CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for abnormal CSVs.",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=None,
        help="Splits to process (overrides config).",
    )
    parser.add_argument(
        "--abnormality-types",
        nargs="*",
        default=None,
        help="Abnormality types to generate (overrides config).",
    )
    parser.add_argument(
        "--level",
        type=str,
        default=None,
        help="Abnormality level (e.g., low, medium, high) (overrides config)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed (overrides config)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/generate_hoser_abnormalities.yaml",
        help="Path to YAML config file (defaults to config/generate_hoser_abnormalities.yaml)",
    )
    parser.add_argument(
        "--abnormality-rate",
        type=float,
        default=None,
        help="Total abnormality rate (0-1) across all types. Overrides config.",
    )
    parser.add_argument(
        "--abnormality-weights",
        type=str,
        default=None,
        help=(
            "Comma-separated weights for abnormality types (same order as --abnormality-types), "
            "or key:val pairs like detour:0.5,perturb:0.25. Overrides config."
        ),
    )
    parser.add_argument(
        "--ensure-change",
        action="store_true",
        help="If set, probabilistic mode will retry other types until a generator makes a change.",
    )
    parser.add_argument(
        "--strong-prob",
        type=float,
        default=None,
        help="Probability (0-1) that a chosen abnormality is generated in 'strong' mode (larger, more detectable changes).",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=None,
        help="Logging progress interval (overrides config).",
    )
    return parser.parse_args()


def load_yaml_config(path: str) -> Dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r") as fh:
        return yaml.safe_load(fh) or {}


def parse_weights_arg(
    weights_arg: Optional[str],
    types: Sequence[str],
    config_weights: Optional[Dict] = None,
) -> List[float]:
    """Return a list of weights aligned with `types`.

    `weights_arg` formats supported:
      - comma-separated values: "0.6,0.2,0.2"
      - key:value pairs: "detour:0.6,perturb:0.4"
    If `weights_arg` is None, `config_weights` (dict) is used if provided; otherwise equal weights.
    """
    if weights_arg:
        # try key:val pairs
        if ":" in weights_arg:
            pairs = [p.strip() for p in weights_arg.split(",") if p.strip()]
            weight_map = {}
            for p in pairs:
                if ":" in p:
                    k, v = p.split(":", 1)
                    try:
                        weight_map[k.strip()] = float(v)
                    except ValueError:
                        weight_map[k.strip()] = 0.0
            return [float(weight_map.get(t, 0.0)) for t in types]
        else:
            parts = [p.strip() for p in weights_arg.split(",") if p.strip()]
            vals = []
            for p in parts:
                try:
                    vals.append(float(p))
                except ValueError:
                    vals.append(0.0)
            if len(vals) == len(types):
                return vals
            # otherwise fallback to config or equal
    if config_weights:
        return [float(config_weights.get(t, 0.0)) for t in types]
    # equal weights
    return [1.0 / len(types)] * len(types)


def load_csv_with_flexible_columns(path: str) -> pl.DataFrame:
    df = pl.read_csv(path)
    cols = df.columns
    if cols[1] not in ["entity_id", "user_id"]:
        raise ValueError(f"Unexpected second column: {cols[1]}")
    return df


def add_abnormality_info_column(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([pl.lit("normal").alias("abnormality_info")])


def save_with_abnormality_info(df: pl.DataFrame, out_path: str):
    # Ensure abnormality_info is last column
    cols = df.columns
    if cols[-1] != "abnormality_info":
        cols = [c for c in cols if c != "abnormality_info"] + ["abnormality_info"]
        df = df.select(cols)
    df.write_csv(out_path)


def insert_detour(rid_list, level, road_id_pool, rng, strong: bool = False):
    """Insert a detour segment.

    If a road graph is available, detours are inserted as a valid path segment
    between two consecutive nodes in the original trajectory.
    """
    n_insert = {"low": 1, "medium": 2, "high": 3}.get(level, 2)
    if strong:
        n_insert = max(n_insert * 2 + 1, n_insert)

    graph = globals().get("GLOBAL_GRAPH")
    if graph is None or len(rid_list) < 2:
        # Fallback: pool-based insertion.
        if len(road_id_pool) == 0:
            return rid_list, []
        detour_roads = rng.choice(
            road_id_pool, size=min(n_insert, len(road_id_pool)), replace=False
        ).tolist()
        insert_pos = rng.integers(0, max(1, len(rid_list) + 1), size=len(detour_roads))
        new_rid_list = rid_list.copy()
        for pos, road in sorted(zip(insert_pos, detour_roads)):
            new_rid_list.insert(int(pos), str(road))
        return new_rid_list, [str(x) for x in detour_roads]

    outgoing = graph["outgoing"]
    edge_set = graph["edge_set"]
    if not _is_valid_walk(rid_list, edge_set):
        return rid_list, []

    # Try multiple insertion locations to increase success rate.
    base_max_edges = {"low": 4, "medium": 7, "high": 12}.get(level, 7)
    if strong:
        base_max_edges = min(16, base_max_edges + 3)
    max_trials = {"low": 6, "medium": 10, "high": 16}.get(level, 10)

    for _ in range(max_trials):
        insert_after = int(rng.integers(0, len(rid_list) - 1))
        start = rid_list[insert_after]
        end = rid_list[insert_after + 1]

        # Prefer a non-trivial detour: forbid taking the direct edge start->end.
        path = _find_bounded_path(
            start,
            end,
            outgoing,
            rng,
            max_edges=base_max_edges,
            forbidden_edges={(start, end)},
        )
        if path is None or len(path) < 3:
            continue

        detour_nodes = path[1:-1]
        new_rid_list = (
            rid_list[: insert_after + 1] + detour_nodes + rid_list[insert_after + 1 :]
        )
        if _is_valid_walk(new_rid_list, edge_set):
            return new_rid_list, detour_nodes

    return rid_list, []


def route_switch_graph(
    rid_list: List[str],
    level: str,
    rng: np.random.Generator,
    strong: bool = False,
) -> tuple[List[str], Optional[tuple[int, int]], Optional[List[str]]]:
    """Replace a segment with an alternative valid path in the road graph."""
    graph = globals().get("GLOBAL_GRAPH")
    if graph is None:
        return rid_list, None, None
    outgoing = graph["outgoing"]
    edge_set = graph["edge_set"]
    if len(rid_list) < 5 or not _is_valid_walk(rid_list, edge_set):
        return rid_list, None, None

    base_seg_len = {"low": 2, "medium": 3, "high": 4}.get(level, 3)
    if strong:
        base_seg_len = min(max(3, base_seg_len * 2), len(rid_list) - 2)

    # Try progressively shorter segments if a long replacement is hard to find.
    for seg_len in range(base_seg_len, 1, -1):
        if len(rid_list) <= seg_len + 2:
            continue
        trials = {"low": 8, "medium": 12, "high": 20}.get(level, 12)
        for _ in range(trials):
            start1 = int(rng.integers(1, len(rid_list) - seg_len - 1))
            a = rid_list[start1 - 1]
            b = rid_list[start1 + seg_len]

            # Allow some slack so we can route around, but keep bounded.
            max_edges = min(16, max(seg_len + 6, 10))
            path = _find_bounded_path(a, b, outgoing, rng, max_edges=max_edges)
            if path is None or len(path) < 3:
                continue
            replacement = path[1:-1]
            original = rid_list[start1 : start1 + seg_len]
            if replacement == original:
                continue
            new_rid_list = (
                rid_list[:start1] + replacement + rid_list[start1 + seg_len :]
            )
            if _is_valid_walk(new_rid_list, edge_set):
                return new_rid_list, (start1, start1 + seg_len), replacement

    return rid_list, None, None


def build_road_pool_stream(path: str, rid_col: str) -> List[str]:
    """Build a deterministic, sorted road id pool by streaming the CSV once.

    Returns a sorted list of unique road ids as strings.
    """
    pool_set = set()
    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        if rid_col not in reader.fieldnames:
            raise ValueError(f"RID column '{rid_col}' not found in {path}")
        for row in reader:
            rids = row[rid_col].split(",") if row[rid_col] else []
            for r in rids:
                pool_set.add(r)
    # return a deterministic ordering
    return sorted(pool_set)


def process_split_streaming(
    input_path: str,
    output_path: str,
    seed: int,
    level: str,
    abnormal_types: Sequence[str],
    abnormality_rate: Optional[float] = None,
    abnormality_weights: Optional[Sequence[float]] = None,
    ensure_change: bool = False,
    progress_interval: int = 10000,
    strong_prob: float = 0.5,
):
    """Stream-process one CSV split deterministically and write output CSV.

    Two-pass approach: first pass builds the global road pool deterministically,
    second pass streams rows and writes originals + generated abnormal rows.
    """
    # detect rid column name by reading header
    with open(input_path, "r", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
    if "rid_list" in header:
        rid_col = "rid_list"
    else:
        rid_col = next((c for c in header if "rid" in c.lower()), "rid_list")

    time_col = _detect_time_col(header)

    logging.info("Building road pool for %s (rid_col=%s)", input_path, rid_col)
    road_pool = build_road_pool_stream(input_path, rid_col)
    logging.info("Road pool size=%d", len(road_pool))

    # If roadmap.rel is available alongside the input split, build a graph so
    # generated anomalies remain valid walks.
    dataset_dir = Path(input_path).parent
    rel_path = dataset_dir / "roadmap.rel"
    if rel_path.exists():
        try:
            globals()["GLOBAL_GRAPH"] = _build_graph_from_rel(rel_path)
            logging.info("Loaded road graph from %s", rel_path)
        except Exception as e:
            globals()["GLOBAL_GRAPH"] = None
            logging.warning("Failed to load roadmap.rel (%s): %s", rel_path, e)
    else:
        globals()["GLOBAL_GRAPH"] = None

    # open reader and writer for streaming second pass
    with (
        open(input_path, "r", newline="") as infh,
        open(output_path, "w", newline="") as outfh,
    ):
        reader = csv.DictReader(infh)
        fieldnames = list(reader.fieldnames) + ["abnormality_info"]
        writer = csv.DictWriter(outfh, fieldnames=fieldnames)
        writer.writeheader()

        global_seed = int(seed)
        types = list(abnormal_types)
        rate_budget = 0.0
        # prepare normalized probabilities if probabilistic mode
        if abnormality_rate is not None:
            probs = None
            if abnormality_weights is not None:
                probs = list(abnormality_weights)
            else:
                probs = [1.0 / len(types)] * len(types)
            s = sum(probs)
            if s <= 0:
                probs = [1.0 / len(types)] * len(types)
            else:
                probs = [p / s for p in probs]

        for idx, row in enumerate(reader):
            if idx % progress_interval == 0 and idx > 0:
                logging.info("Processed %d rows", idx)
            # write original row with 'normal' abnormality_info
            out_row = dict(row)
            out_row["abnormality_info"] = "normal"
            writer.writerow(out_row)

            # prepare common values
            rid_list = row.get(rid_col, "")
            rids = rid_list.split(",") if rid_list else []
            time_list = row.get(time_col, "") if time_col else ""
            # per-row deterministic RNG
            rng = np.random.default_rng(global_seed + idx)

            # prepare an injected index recorder file (append mode) for reproducibility
            injected_index_file = output_path + ".injected_indices.jsonl"

            # legacy: write one abnormal row per requested type
            if abnormality_rate is None:
                for a_type in abnormal_types:
                    # determine strong flag per generated abnormality
                    is_strong = rng.random() < float(strong_prob)
                    if a_type == "detour":
                        new_rids, detour = insert_detour(
                            rids, level, road_pool, rng, strong=is_strong
                        )
                        changed = bool(detour)
                        if detour:
                            info = {"type": "detour", "level": level, "detour": detour}
                        else:
                            info = {"type": "detour", "level": level}
                    elif a_type == "perturb":
                        new_rids, perturbed = perturb_rids(
                            rids, level, road_pool, rng, strong=is_strong
                        )
                        changed = bool(perturbed)
                        info = {
                            "type": "perturb",
                            "level": level,
                            "perturbed": perturbed,
                        }
                    elif a_type == "route_switch":
                        new_rids, seg_range, seg2 = route_switch_graph(
                            rids, level, rng, strong=is_strong
                        )
                        changed = seg_range is not None
                        info = {"type": "route_switch", "level": level}
                        if seg_range is not None:
                            info["seg_range"] = seg_range
                            info["seg2"] = seg2
                    else:
                        continue

                    # If graph is present, require that the result is a valid walk.
                    graph = globals().get("GLOBAL_GRAPH")
                    if changed and graph is not None:
                        if not _is_valid_walk(new_rids, graph["edge_set"]):
                            continue

                    if not changed:
                        continue

                    # annotate whether generated change was strong
                    if is_strong:
                        info["strength"] = "strong"

                    # attach original trajectory for future reference
                    info["real"] = {"rid_list": rid_list, "time_list": time_list}

                    # write abnormal row
                    new_row = dict(row)
                    new_row[rid_col] = ",".join(new_rids)
                    if time_col:
                        new_row[time_col] = _adjust_time_list_str(
                            row.get(time_col, ""), target_len=len(new_rids)
                        )
                    # write abnormality_info as compact string
                    new_row["abnormality_info"] = str(info)
                    writer.writerow(new_row)
                    # record injected index
                    try:
                        info_compact = dict(info)
                        info_compact.pop("real", None)
                        with open(injected_index_file, "a") as jf:
                            jf.write(
                                json.dumps(
                                    {
                                        "idx": idx,
                                        "type": info.get("type"),
                                        "info": info_compact,
                                    }
                                )
                                + "\n"
                            )
                    except Exception:
                        pass
            else:
                # probabilistic mode: decide whether this row gets an abnormality.
                # When ensure_change is enabled, we treat `abnormality_rate` as a
                # target ratio and use a carry-over budget so failures don't reduce
                # the realized abnormal proportion.
                rate_budget += float(abnormality_rate)
                should_attempt = False
                if ensure_change:
                    if rate_budget >= 1.0:
                        should_attempt = True
                else:
                    u = rng.random()
                    should_attempt = u < float(abnormality_rate)

                if should_attempt:
                    max_attempts = 12 if ensure_change else 1
                    attempts = 0
                    changed = False
                    info = None
                    new_rids = rids
                    seg_range = None
                    detour = []
                    perturbed = []

                    while attempts < max_attempts and not changed:
                        attempts += 1
                        chosen = rng.choice(types, p=probs)
                        is_strong = rng.random() < float(strong_prob)

                        if chosen == "detour":
                            new_rids, detour = insert_detour(
                                rids, level, road_pool, rng, strong=is_strong
                            )
                            changed = bool(detour)
                            info = {"type": "detour", "level": level}
                            if detour:
                                info["detour"] = detour
                        elif chosen == "perturb":
                            new_rids, perturbed = perturb_rids(
                                rids, level, road_pool, rng, strong=is_strong
                            )
                            changed = bool(perturbed)
                            info = {"type": "perturb", "level": level}
                            if perturbed:
                                info["perturbed"] = perturbed
                        elif chosen == "route_switch":
                            new_rids, seg_range, seg2 = route_switch_graph(
                                rids, level, rng, strong=is_strong
                            )
                            changed = seg_range is not None
                            info = {"type": "route_switch", "level": level}
                            if seg_range is not None:
                                info["seg_range"] = seg_range
                                info["seg2"] = seg2
                        else:
                            changed = False

                        graph = globals().get("GLOBAL_GRAPH")
                        if changed and graph is not None:
                            if not _is_valid_walk(new_rids, graph["edge_set"]):
                                changed = False

                    if changed and info is not None:
                        if is_strong:
                            info["strength"] = "strong"

                        info["real"] = {"rid_list": rid_list, "time_list": time_list}

                        new_row = dict(row)
                        new_row[rid_col] = ",".join(new_rids)
                        if time_col:
                            new_row[time_col] = _adjust_time_list_str(
                                row.get(time_col, ""), target_len=len(new_rids)
                            )
                        new_row["abnormality_info"] = str(info)
                        writer.writerow(new_row)
                        if ensure_change:
                            rate_budget -= 1.0

                        try:
                            info_compact = dict(info)
                            info_compact.pop("real", None)
                            with open(injected_index_file, "a") as jf:
                                jf.write(
                                    json.dumps(
                                        {
                                            "idx": idx,
                                            "type": info.get("type"),
                                            "info": info_compact,
                                        }
                                    )
                                    + "\n"
                                )
                        except Exception:
                            pass


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    # Load config file (if present) and merge with CLI args (CLI overrides config)
    config = load_yaml_config(
        args.config
        if hasattr(args, "config")
        else "config/generate_hoser_abnormalities.yaml"
    )

    splits = (
        args.splits
        if args.splits is not None
        else config.get("splits", ["train", "val", "test"])
    )
    abnormal_types = (
        args.abnormality_types
        if args.abnormality_types is not None
        else config.get("types", ABNORMALITY_TYPES)
    )
    level = args.level if args.level is not None else config.get("level", "medium")
    seed = int(args.seed) if args.seed is not None else int(config.get("seed", 42))

    abnormality_rate = (
        args.abnormality_rate
        if getattr(args, "abnormality_rate", None) is not None
        else config.get("total_rate")
    )
    # parse weights from CLI or config
    if getattr(args, "abnormality_weights", None):
        weights_list = parse_weights_arg(
            args.abnormality_weights, abnormal_types, config.get("weights")
        )
    else:
        weights_list = parse_weights_arg(None, abnormal_types, config.get("weights"))

    ensure_change = bool(
        getattr(args, "ensure_change", False) or config.get("ensure_change", False)
    )
    progress_interval = (
        int(args.progress_interval)
        if getattr(args, "progress_interval", None) is not None
        else int(config.get("progress_interval", 10000))
    )

    os.makedirs(args.output_dir, exist_ok=True)

    for split in splits:
        in_path = os.path.join(args.input_dir, f"{split}.csv")
        out_path = os.path.join(args.output_dir, f"{split}.csv")
        if not os.path.exists(in_path):
            logging.warning("%s does not exist, skipping.", in_path)
            continue

        logging.info("Processing split=%s", split)
        # Determine strong probability for the configured level.
        # Priority: CLI `--strong-prob` > config `strong_prob_per_level[level]` > config `strong_prob` > default 0.0
        if getattr(args, "strong_prob", None) is not None:
            strong_prob = float(args.strong_prob)
        else:
            spp = config.get("strong_prob_per_level", {}) or {}
            if isinstance(spp, dict) and level in spp:
                strong_prob = float(spp.get(level, 0.0))
            else:
                strong_prob = float(config.get("strong_prob", 0.0))

        process_split_streaming(
            input_path=in_path,
            output_path=out_path,
            seed=seed,
            level=level,
            abnormal_types=abnormal_types,
            abnormality_rate=abnormality_rate,
            abnormality_weights=weights_list,
            ensure_change=ensure_change,
            progress_interval=progress_interval,
            strong_prob=strong_prob,
        )
        logging.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
