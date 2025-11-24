Title: Robust loader for road_id -> token mapping (fix for Issue #63)

Summary
-------
Issue #63 reports failures when mapping files are supplied in non-uniform JSON
formats (nested dict values instead of plain ints). This change introduces a
robust loader that normalizes multiple formats to a single NumPy array output
and adds TDD-style unit tests and fixtures.

Files to be added/modified
-------------------------
- Add `critics/mapping_utils.py` — new module: implements `load_road_to_token_mapping`.
- Add `tests/test_mapping_loader.py` — unit tests covering plain, nested, and dict inputs.
- Add fixtures in `tests/fixtures/` — `mapping_plain.json` and `mapping_nested.json`.
- Update callers (follow-up PR):
  - `tools/run_lmtad_spatial_pipeline.py` — attempt to load an existing mapping file using the new loader when available and pass the normalized `np.ndarray` into `evaluate_spatial_abnormal_trajectories`.
  - `tools/evaluate_lmtad_spatial_abnormal.py` — accept `road_to_token` arrays returned by loader (already compatible) and document expected format.

Rationale
---------
- Many mapping files in the wild use slightly different shapes. Failing fast
  on unexpected formats caused runtime crashes during evaluation and CI.
- Returning a consistent array (with -1 for unknown indices) keeps downstream
  logic simple and safe (existing `map_roads_to_tokens` already handles -1).
- Tests and fixtures enable reproducible regression checks and make the fix
  maintainable.

Risk assessment
---------------
- Backwards compatible: callers receiving a NumPy array are unchanged.
- Low risk: loader is conservative, uses -1 for unknown or unextractable
  entries rather than throwing in production code paths; optionally strict
  behavior can be enabled later.

Next steps (follow-up PR)
------------------------
1. Update pipeline to attempt loading `eval_dir/road_to_token.json` (or other
   canonical paths) and pass loaded mapping into evaluation functions.
2. Add integration tests that run `tools/run_lmtad_spatial_pipeline.py` with
   `--skip-generation --skip-extraction` using a tiny eval dir to verify end-to-end behavior.
3. Add a short migration note in `docs/` explaining common mapping file formats
   and how to regenerate a canonical plain mapping using `GridMapper.map_all()`.

Acceptance criteria
-------------------
- New unit tests pass: `pytest tests/test_mapping_loader.py` succeeds.
- Existing consumers that expect an `np.ndarray` continue to operate without change.
