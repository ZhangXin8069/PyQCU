from pyqcu.testing import test_multigrid_multithread


def main():
    test_multigrid_multithread(nthreads=2, lat_size=[8, 8, 8, 8], mass=0.05, tol=1e-5)


if __name__ == "__main__":
    main()
