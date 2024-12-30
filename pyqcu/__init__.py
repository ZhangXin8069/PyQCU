print(
    '''
    @@@@@@######QCU NOTES START######@@@@@@@
    1. The libqcu.so was compiled when pyqcu setup in download_path/PyQCU/lib, please add this path to your LD_LIBRARY_PATH.
    2. The QCU splite grid by x->y->z->t, lattice by x->y->z->t->p->d->c->c or x->y->z->t->c->s(->p) and x->y->z->t->c->s->c->s(->p).
    3. The QCU input params in numpy array(dtype=np.int), argv in  numpy array(dtype=np.float32 or float64) array, other in cupy array(dtype=cp.complex64 or complex128).
     @@@@@@######QCU NOTES END######@@@@@@@
    ''')
