from qcu import Pointer


def testWilsonDslashQcu(fermion_out: Pointer, fermion_in: Pointer,
                        gauge: Pointer, params: Pointer, argv: Pointer) -> None: ...


def applyWilsonDslashQcu(fermion_out: Pointer, fermion_in: Pointer,
                         gauge: Pointer, params: Pointer, argv: Pointer) -> None: ...


def testCloverDslashQcu(fermion_out: Pointer, fermion_in: Pointer,
                        gauge: Pointer, params: Pointer, argv: Pointer) -> None: ...


def applyCloverDslashQcu(fermion_out: Pointer, fermion_in: Pointer,
                         gauge: Pointer, params: Pointer, argv: Pointer) -> None: ...


def applyBistabCgQcu(fermion_out: Pointer, fermion_in: Pointer,
                     gauge: Pointer, params: Pointer, argv: Pointer) -> None: ...


def applyCgQcu(fermion_out: Pointer, fermion_in: Pointer,
               gauge: Pointer, params: Pointer, argv: Pointer) -> None: ...


def applyGmresIrQcu(fermion_out: Pointer, fermion_in: Pointer,
                    gauge: Pointer, params: Pointer, argv: Pointer) -> None: ...
