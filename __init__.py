import os
os.environ['LD_LIBRARY_PATH'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib') + ':' + os.environ.get('LD_LIBRARY_PATH', '')
print("LD_LIBRARY_PATH:", os.environ['LD_LIBRARY_PATH'])