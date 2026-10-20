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
python -m pytest -q pyqcu/testing/kupl/test_kupl_python.py
```

Set `PYQCU_KUPL_LIB=/absolute/path/libqcu.so` when the library is installed
outside the default `cpp/kupl/qcu/build/` directory.
