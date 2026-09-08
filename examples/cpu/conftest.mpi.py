def main():
    from pyqcu.testing import test_solver
    import torch

    test_solver(kind='clover', method='multigrid', dtype=torch.complex64,
                lat_size=[8, 8, 16, 16], device=torch.device('cpu'), support_parity=True)


if __name__ == "__main__":
    main()
