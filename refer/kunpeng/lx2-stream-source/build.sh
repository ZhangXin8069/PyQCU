#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
# Load your BiSheng/LLVM OpenMP toolchain environment before running this script.
"${CC:-clang}" -O3 -fopenmp -march=armv9-a+sme+sme-f64f64 \
  -msve-vector-bits=scalable stream_probe.c -o stream_probe
