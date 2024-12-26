import os
os.environ['LD_LIBRARY_PATH'] = '/root/pyqcu/lib' + ':' + os.environ.get('LD_LIBRARY_PATH', '')
print("LD_LIBRARY_PATH:", os.environ['LD_LIBRARY_PATH'])