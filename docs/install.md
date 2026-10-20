```bash
bash ./install.sh
```

## KUPL CPU backend

The independent MPI+OpenMP backend does not require CUDA:

```bash
cmake -S cpp/kupl/qcu -B cpp/kupl/qcu/build \
  -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/kupl/qcu/build -j
ctest --test-dir cpp/kupl/qcu/build --output-on-failure
```

Python uses the same flat ABI through `pyqcu.kupl.qcu`:

```bash
python -m pytest -q pyqcu/testing/kupl
```

Set `PYQCU_KUPL_LIB=/absolute/path/libqcu.so` when the library is installed
outside the default `cpp/kupl/qcu/build/` directory.

## QUDA/PyQUDA example runtime

The simple PyQUDA examples preload the repository-local
`data/quda-sm70-install/lib/libquda.so` when present. `env.sh` also places that
installation before stale global QUDA libraries. Set `QUDA_INSTALL` explicitly
to select another architecture or precision build:

```bash
export QUDA_INSTALL=/absolute/path/to/quda-install
source ./env.sh
python pyqcu/testing/simple/quda_bistabcg.py --help
```

The selected library must contain a CUDA architecture compatible with the
active GPU, and its QUDA ABI must match the installed PyQUDA extension.
