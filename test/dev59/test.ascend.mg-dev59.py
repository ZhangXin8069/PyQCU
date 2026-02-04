from pyqcu.torch import qcu
import torch
lat_x, lat_y, lat_z, lat_t = 16, 16, 8, 8
_qcu = qcu(lat_size=[lat_x, lat_y, lat_z, lat_t], dtype_list=[torch.complex128, torch.complex64, torch.complex64, torch.complex32], device_list=[
           torch.device('cpu'), torch.device('cpu'), torch.device('cpu'), torch.device('cpu')], dslash='clover', solver='mg', verbose=False)
_qcu.load()
_qcu.init()
_qcu.solve()
_qcu.test()
_qcu.mg.plot(
    save_path=f"test.ascend.{_qcu.solver}.{_qcu.device_list[0].type}-np{_qcu.size}-dev59.png")
_qcu.save()