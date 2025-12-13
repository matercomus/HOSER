**Generate HOSER Abnormalities**

- **File:**: `generate_hoser_abnormalities.py`
- **Location:**: repository root
- **Purpose:**: Stream-process HOSER CSV splits to produce abnormal (perturbed) trajectories alongside originals, writing outputs with an `abnormality_info` column that records what was changed.

**Overview**
- **What it does:**: The script reads HOSER CSV splits (`train`, `val`, `test`) and for each trajectory optionally generates one or more abnormal variants (detour, route_switch, perturb). It writes a CSV per split that contains the original rows (tagged `normal`) and additional abnormal rows with `abnormality_info` describing the change.
- **Design:**: Streaming-first, two-pass per split:
  - Pass 1: stream the input CSV once to build a deterministic, sorted global road-id pool (unique values from the `rid_list` column). This keeps memory low while still enabling pool-based generators.
  - Pass 2: stream rows and for each row: write the original row (with `abnormality_info='normal'`) and generate configured abnormal rows using a per-row deterministic RNG.
- **Determinism:**: RNG is derived per-row as `np.random.default_rng(global_seed + row_index)`. With the same seed and unchanged input order, the outputs are deterministic.

**High-level flow / main points**
- **CLI:**: the script exposes flags for `--input-dir`, `--output-dir`, `--splits`, `--abnormality-types`, `--level`, and `--seed`.
- **Streaming approach:**: two-pass streaming avoids loading the whole split into memory; most heavy datasets should run with minimal memory overhead (the road pool is a set of unique ids then sorted).
- **Outputs:**: each split's output CSV has the same columns as the input plus the extra `abnormality_info` column. Original rows have `abnormality_info` set to `normal`. Abnormal rows have a compact Python-dict-like string describing the type and parameters (e.g., `{'type': 'detour', 'level': 'medium', 'detour': ['101','203']}`).

**Abnormality generators (detailed)**

**Detour**
- **Function:** `insert_detour(rid_list, level, road_id_pool, rng)`
- **Purpose:** insert 1–3 road IDs into the trajectory at random positions to simulate a detour.
- **Level mapping:** `low` → 1 inserted road, `medium` → 2, `high` → 3.
- **Inputs:**
  - `rid_list`: list of road IDs (strings) for the trajectory, e.g. `['101','102','103','104']`.
  - `road_id_pool`: list of candidate road IDs (strings) built from the split.
  - `rng`: per-row numpy RNG for deterministic draws.
- **Behavior:** randomly select `n_insert` distinct road IDs from `road_id_pool` (excluding duplicates if possible), pick insertion positions using `rng.integers(...)`, and insert them into copies of `rid_list`.
- **Example:**
  - Input: `rid_list=['101','102','103','104']`, `level='medium'` (2 inserted), `road_id_pool=['201','202','203','101','102']` and deterministic RNG.
  - Output `new_rid_list`: e.g. `['101','202','102','103','104']` (specific result depends on seed/index) and `detour=['202','203']` recorded in `abnormality_info`.

**Perturb**
- **Function:** `perturb_rids(rid_list, level, road_id_pool, rng)`
- **Purpose:** replace a small number of road IDs in the trajectory with other IDs from the pool to simulate small noise/measurement error.
- **Level mapping:** `low` → 1 replaced index, `medium` → `max(1, n//10)`, `high` → `max(1, n//5)`.
- **Behavior:** choose `n_perturb` distinct indices in the trajectory and replace each with a random candidate from the pool (excluding the original value). The function returns `(new_rid_list, perturbed)` where `perturbed` is a list of triples `(index, old, new)`.
- **Example:**
  - Input: `rid_list=['101','102','103','104','105']`, `level='low'`.
  - Possible Output: `new_rid_list=['101','999','103','104','105']`, `perturbed=[(1,'102','999')]` and `abnormality_info` contains `{'type':'perturb','level':'low','perturbed':[(1,'102','999')]}`.

**Route Switch**
- Two variants exist in the codebase:
  - `route_switch(rid_list, other_rid_list, level, rng)`: replaces a contiguous segment of `rid_list` with a contiguous segment taken from `other_rid_list` (used in the original in-memory implementation where another trajectory can be sampled).
  - `route_switch_from_pool(rid_list, road_pool, level, rng)`: streaming-friendly variant that replaces a contiguous segment of the current trajectory with a contiguous slice sampled from the global road pool (so no second-trajectory memory is required).
- **Purpose:** simulate a trajectory switching to a different route segment (e.g., taking another path for part of the trip).
- **Level mapping:** `low` → replace ~2 roads, `medium` → ~3, `high` → ~4 (exact mapping in code uses `{'low':2,'medium':3,'high':4}`).
- **Behavior (from-pool):** pick a start index in the original trajectory and pick a start index in the pool; replace the contiguous segment of length `seg_len` with the pool slice; return the new list and metadata (`seg_range`, `seg2`).
- **Example:**
  - Input: `rid_list=['10','11','12','13','14','15']`, `level='medium'` ⇒ `seg_len=3`.
  - If the sampled pool slice is `['201','202','203']` and start1=2, the new trajectory becomes `['10','11','201','202','203','15']` and `abnormality_info` records the segment and inserted sequence.

**Abnormality info format**
- In streaming output the script writes a compact Python-dict-like string as `abnormality_info`, e.g. `"{'type':'detour','level':'medium','detour':['202','203']}"`.
- Original rows receive `abnormality_info='normal'`.

**CLI usage examples**
- Basic run over splits `train,val,test`:

  `uv run python generate_hoser_abnormalities.py --input-dir data/Beijing --output-dir data/Beijing_abnormal --splits train val test --abnormality-types detour route_switch perturb --level medium --seed 42`

- Notes:
  - `--seed` controls determinism; with the same seed and identical input order, results are reproducible.
  - `--abnormality-types` accepts any combination of `detour`, `route_switch`, `perturb`.

**Performance & determinism notes**
- The script uses a two-pass, streaming design so each split is streamed twice: one pass to collect the road-id pool and another to emit original and abnormal rows. This keeps memory usage low and predictable.
- Determinism is achieved via a per-row RNG derived from the global seed plus the row index: `rng = np.random.default_rng(seed + row_index)`. If you reorder rows, generated abnormalities will change — keep input ordering consistent for reproducible outputs.

**Developer notes & extension points**
- Single-pass alternative: if you prefer a strict single-pass pipeline (no pool), the code can be adapted to use hash-based deterministic sampling from the row content and seed, but the distribution of inserted/perturbed road IDs will differ compared to pool-based selection.
- Route-switch realism: `route_switch_from_pool` uses contiguous pool slices which is a reasonable approximation, but if you need a route-switch to use real contiguous segments from other trajectories, use the original in-memory `route_switch` variant (requires keeping or sampling other trajectories) or maintain a small cache of sampled segments gathered during the first pass.

**Where to look in the code**
- `process_split_streaming(...)` — orchestration for a single split (two-pass streaming + per-row RNG).
- `build_road_pool_stream(...)` — collects unique road IDs from the `rid_list` column in a deterministic, sorted order.
- `insert_detour(...)` — detour generator.
- `perturb_rids(...)` — perturb generator.
- `route_switch(...)` and `route_switch_from_pool(...)` — route-switch variants.

**Quick troubleshooting**
- If the script appears to hang on large files, ensure it is actually processing rows by watching INFO logs (the script logs pool size and `Processed N rows` every 10k rows). You can raise logging verbosity or reduce the progress interval if preferred.
- For bitwise-identical deterministic tests, run the same command twice with the same seed and compare output files (e.g., `sha256sum` the output split files).

**Contact / next steps**
- If you want, I can add a small integration test that runs the CLI twice on a tiny sample and asserts outputs are identical to prove determinism end-to-end.


**Real-run examples (Porto & Beijing)**

Below are excerpts and metrics from recent runs on real HOSER datasets (these are representative logs produced by `generate_hoser_abnormalities.py` when run with `--seed 42 --level medium --abnormality-types detour route_switch perturb`). Use these to validate expected behavior and to illustrate scale.

Porto (example)
- Sample log excerpt:

  2025-12-13 17:18:14,490 INFO Processed 470000 rows
  2025-12-13 17:19:51,796 INFO Processed 480000 rows
  2025-12-13 17:20:03,848 INFO Wrote data/porto_hoser_abnormal/train.csv
  2025-12-13 17:20:03,848 INFO Processing split=val
  2025-12-13 17:20:03,861 INFO Building road pool for data/porto_hoser/val.csv (rid_col=rid_list)
  2025-12-13 17:20:04,980 INFO Road pool size=9925
  2025-12-13 17:21:32,613 INFO Processed 10000 rows

- Notes:
  - The Porto training split processed ~480k rows in this example run and wrote the abnormal output at `data/porto_hoser_abnormal/train.csv`.
  - The `val` split's road pool contained 9,925 unique road IDs (used by pool-based generators).

Beijing (example)
- Sample log excerpt:

  2025-12-13 16:05:11,203 INFO Processing split=train
  2025-12-13 16:05:11,223 INFO Building road pool for data/Beijing/train.csv (rid_col=rid_list)
  2025-12-13 16:05:24,017 INFO Road pool size=36862
  2025-12-13 16:09:49,167 INFO Processed 10000 rows
  2025-12-13 16:14:08,726 INFO Processed 20000 rows
  2025-12-13 16:18:23,386 INFO Processed 30000 rows

- Notes:
  - The Beijing training split had a larger road pool (36,862 unique road IDs) indicating a denser candidate set for detours/perturbations.
  - Progress logs are emitted every 10k rows (adjustable in code) so long runs can be monitored for throughput and progress.

Using these examples
- Road pool sizes directly affect the diversity of inserted/perturbed IDs: larger pools produce more diverse abnormalities.
- The output files (for example `data/porto_hoser_abnormal/train.csv` and `data/Beijing_abnormal/train.csv`) contain original rows plus abnormal rows; compare file sizes or row counts to estimate how many abnormal rows were generated.
