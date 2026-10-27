"""Contract tests for KUPL benchmark stdout parsing."""

import numpy as np

from pyqcu.testing.simple.kupl_bistabcg import prebind_kupl_inputs
from pyqcu.testing.simple.benchmark_kupl import (
    EXPECTED_BENCHMARK_KEYS,
    parse_output,
    validate_records,
)


class _RecordingQcu:
    def __init__(self) -> None:
        self.events = []

    def applyInitQcu(self, set_ptrs, params, argv) -> None:
        self.events.append("init")
        set_ptrs.array[params.array[15]] = 0x1234

    def applyEndQcu(self, set_ptrs, params) -> None:
        self.events.append("end")
        set_ptrs.array[params.array[15]] = 0


def test_empty_benchmark_output_is_incomplete() -> None:
    missing, incomplete = validate_records(parse_output("no records\n"))
    assert missing == sorted(EXPECTED_BENCHMARK_KEYS)
    assert incomplete == []


def test_partial_benchmark_record_is_incomplete() -> None:
    records = parse_output(
        "precision=c64 operator=wilson ranks=1 grid=1,1,1,1 local=2,2,2,4 "
        "threads_per_rank=1 seconds=0.1\n"
    )
    missing, incomplete = validate_records(records)
    assert ("c64", "wilson") not in missing
    assert ("c64", "wilson") in incomplete


def test_prebind_inputs_runs_a_complete_private_lifecycle() -> None:
    qcu = _RecordingQcu()
    prebind_kupl_inputs(
        qcu, (4, 4, 4, 8), 0.05, 1e-6, 1.0, np.dtype(np.complex64),
        100, (1, 1, 1, 1), 0, 1)
    assert qcu.events == ["init", "end"]
