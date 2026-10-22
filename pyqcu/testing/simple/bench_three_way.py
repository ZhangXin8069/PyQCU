#!/usr/bin/env python3
"""Deprecated alias for ``compare_three_way.py``.

The old monolithic benchmark was split so production can use the same case
preparation without any timing machinery:

* ``prepare_case.py``  - gauge/source preparation and SHA256 manifest
* ``solve_case.py``    - one production solve on cuda, quda or kupl
* ``compare_three_way.py`` - warm-up + repeated timing comparison

This file only forwards its arguments; all new work should call
``compare_three_way.py`` directly.  It will be removed after one release.
"""

from __future__ import annotations

import sys

try:
    from .compare_three_way import main
except ImportError:  # direct script execution
    from compare_three_way import main


if __name__ == "__main__":
    print("warning: bench_three_way.py is deprecated; use "
          "compare_three_way.py", file=sys.stderr)
    raise SystemExit(main())
