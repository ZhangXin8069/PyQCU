first the `Python >= 3.10`
run this:
```bash
pip install -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple pytest mpi4py h5py tilelang --resume-retries 10
```
then the **pytorch** and **tilelang** will be installed, that are all we needed.

and do not forget to `export PYTHONPATH=/home/path/to/pyqcu:${PYTHONPATH}`