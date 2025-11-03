from pyqcu.ascend import qcu
import torch
lat_x, lat_y, lat_z, lat_t = 16, 16, 16, 16
_qcu = qcu(lat_size=[lat_x, lat_y, lat_z, lat_t], dtype_list=[torch.complex64, torch.complex64, torch.complex64, torch.complex64], device_list=[
           torch.device('npu'), torch.device('npu'), torch.device('npu'), torch.device('npu')], dslash='clover', solver='bistabcg', verbose=False)
_qcu.load()
_qcu.init()
_qcu.solve()
_qcu.test()
_qcu.mg.plot(
    save_path=f"test.ascend.{_qcu.solver}.{_qcu.device_list[0].type}-np{_qcu.size}-dev59.png")
_qcu.save()
