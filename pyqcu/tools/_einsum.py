
import torch


def Eexyzt_exyzt2Exyzt(Eexyzt: torch.Tensor, exyzt: torch.Tensor) -> torch.Tensor:
    return torch.einsum(
        'Eexyzt,exyzt->Exyzt', Eexyzt, exyzt)
