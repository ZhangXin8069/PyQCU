# data/

Only this contract set is tracked: `.gitignore`, `AGENTS.md`, `README.md`. Everything
else under `data/` is ignored by `.gitignore` (`*` plus explicit un-ignores), because
build products, caches and measurement tensors are reproducible and do not belong in
the history.

## What used to be here and where it went

`data/rho_20261125/` held the capture-fraction measurement: a script, its two basis
arrays (786,560 B each) and a JSON table. The scripts belong in code, and the arrays
are build products, so they moved:

| was | now |
|---|---|
| `data/rho_20261125/build_and_measure.py` | `pyqcu/testing/python/rho_capture_fraction.py` |
| `data/rho_20261125/rho2.json` | the `logs/v20261123.txt` section 182 table |
| `data/rho_20261125/sb_*_lonv.npy` | not tracked; rebuild the two caches |

To regenerate the table, build the two caches (one per null-vector builder) and run
the script with both directories given explicitly, because neither input is in Git:

```bash
python3 pyqcu/testing/python/rho_capture_fraction.py \
    --small  /path/to/dir/with/gauge_and_source \
    --caches /path/to/dir/with/sb_inverse_lonv.npy_and_sb_legacy_lonv.npy \
    --json   /tmp/rho2.json
```

The `--small` directory needs `gauge_8x8x8x16_c64.h5` and
`source_point_8x8x8x16_c64.h5`. Verified 2026-11-26: the two gates pass (single basis
vector gives rho = 1.000000, dense random field lands at 107.8% / 106.3% of capacity)
and all eleven rows reproduce the logged values exactly, e.g. k=16 legacy 0.088202 and
k=20 leg/inv 1.311.

## What is worth keeping here

Nothing else. If a measurement needs a tensor that takes minutes to build and cannot
be rebuilt on demand, say so in `AGENTS.md` before adding an exception — the gate is
machine-enforced by `form-audit.sh` (`TRACKED_DATA_CONTENT`), and loosening it silently
is exactly what it exists to prevent.
