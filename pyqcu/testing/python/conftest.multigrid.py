def main():
    import torch
    from pyqcu.testing import test_solver

    mass = 0.05
    kappa = 1 / (2 * mass + 8)
    test_solver(method='multigrid', dtype=torch.complex64, device=torch.device('cuda'), kappa=torch.Tensor([kappa]),
                lat_size=[8, 8, 8, 16], max_level=2, support_parity=True)


if __name__ == '__main__':
    main()
