# quantum-mg
A set of routines for 2D multigrid. Depends on quantum-linalg.

The lattice layout is an even-odd spatial partitioning (all even then all odd). The file lattice/lattice.h gives a lot of functions that convert between indexing and coordinates. We define conversion functions for various lattice types, with the tensor ordering from outermost to innermost:

* LatticeComplex "volume", (eo, y, x)
* LatticeColorVector "size\_cv", (eo, y, x, c)
* LatticeColorMatrix "size\_cm", (eo, y, x, c1, c2) [c1 row, c2 column]
* LatticeGauge "size\_gauge", (mu, eo, y, x, c1, c2) [mu: +x, +y]
* LatticeHopping "size\_hopping", (mu, eo, y, x, c1, c2) [mu: +x, +y, -x, -y]
* LatticeCorner "size\_corner", (munu, eo, y, x, c1, c2) [munu: +x+y, -x+y, -x-y, +x-y]

