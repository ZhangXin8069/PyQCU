// Copyright (c) 2017 Evan S Weinberg
// Header file for a lattice object, which contains various info on
// the lattice volume, degrees of freedom per site (think ColorVector,
// then we can form products of it for ColorMatrix types), and functions
// to convert between indexing schemes. 

#ifndef QMG_LATTICE_HEADER
#define QMG_LATTICE_HEADER

// Hard coded class for 2D. Will eventually create an n-D class. I already
// have reference code, just not for even-odd indexing.
class Lattice2D
{
private:
  const int nd = 2; 
  int dims[2];
  int nc;
  
  int volume; // spatial volume
  int size_cv; // spatial volume * nc
  int size_cm; // spatial volume * nc * nc
  int size_gauge; // spatial volume * nc * nc * nd, for gauge field.
  int size_hopping; // spatial volume * nc * nc * 2 * nd
                    //    for hopping, two-link stencil terms. (+x, +y, -x, -y)
  int size_corner;  // spatial volume * nc * nc * 2 * nd (for 2d)
                    //    for corner stencil terms. 
  
  
public:
  
  Lattice2D(int xlen, int ylen, int my_nc)
    : nc(my_nc)
  {
    dims[0] = xlen;
    dims[1] = ylen;
    volume = dims[0]*dims[1];
    size_cv = volume*nc;
    size_cm = size_cv*nc;
    size_gauge = size_cm*nd;
    size_hopping = size_gauge*2;
    size_corner = size_gauge*2; 
  }
  
  Lattice2D(const Lattice2D& copy)
    : nc(copy.nc)
  {
    dims[0] = copy.dims[0];
    dims[1] = copy.dims[1];
    volume = dims[0]*dims[1];
    size_cv = volume*nc;
    size_cm = size_cv*nc;
    size_gauge = size_cm*nd;
    size_hopping = size_gauge*2;
    size_corner = size_gauge*2; 
  }
  
  ~Lattice2D()
  {  }

  void update_nc(int my_nc)
  {
    nc = my_nc;
    size_cv = volume*nc;
    size_cm = size_cv*nc;
    size_gauge = size_cm*nd;
    size_hopping = size_gauge*2;
    size_corner = size_gauge*2; 
  }
  
  /////////////////////////////////////////////////
  // FUNCTIONS TO GO FROM COORDINATES TO INDICES //
  /////////////////////////////////////////////////
  
  // Switch from an (x,y) coordinate to an even-odd partitioned index.
  inline int coord_to_index(int x, int y)
  {
    if (volume == 1) return 0;
    int parity = (x+y)%2;
    int i = (y+parity*dims[1])*dims[0]/2; // find y coord piece.
    return i + (x/2)%(dims[0]/2); // add x coord piece.
  }
  
  inline int coord_to_index(int* coord)
  {
    return coord_to_index(coord[0], coord[1]);
  }

  // Switch from an (x,y,dof) coordinate to an even-odd partitioned index.
  inline int dof_coord_to_index(int total_dof, int x, int y, int dof)
  {
    return total_dof*coord_to_index(x, y) + dof;
  }
  
  inline int dof_coord_to_index(int total_dof, int* coord, int dof)
  {
    return dof_coord_to_index(total_dof, coord[0], coord[1], dof);
  }

  inline int dof_coord_to_index(int total_dof, int i, int dof)
  {
    return total_dof*i + dof;
  }
  
  // Switch from an (x,y,nc) coordinate to an even-odd partitioned index.
  inline int cv_coord_to_index(int x, int y, int c)
  {
    return nc*coord_to_index(x, y) + c;
  }
  
  inline int cv_coord_to_index(int* coord, int c)
  {
    return cv_coord_to_index(coord[0], coord[1], c);
  }

  inline int cv_coord_to_index(int i, int c)
  {
    return nc*i + c;
  }
  
  // Switch from an (x,y,nc,nc) coordinate to an even-odd partitioned index.
  inline int cm_coord_to_index(int x, int y, int c1, int c2) // c1 = row, c2 = col
  {
    return nc*cv_coord_to_index(x,y,c1) + c2;
  }
  
  inline int cm_coord_to_index(int* coord, int c1, int c2)
  {
    return cm_coord_to_index(coord[0], coord[1], c1, c2);
  }

  inline int cm_coord_to_index(int i, int c1, int c2) // c1 = row, c2 = col
  {
    return nc*cv_coord_to_index(i,c1) + c2;
  }
  
  // Switch from an (x,y,nc,nc,mu) coordinate to an even-odd partitioned index
  inline int gauge_coord_to_index(int x, int y, int c1, int c2, int mu)
  {
    return mu*size_cm + cm_coord_to_index(x,y,c1,c2);
  }
  
  inline int gauge_coord_to_index(int* coord, int c1, int c2, int mu)
  {
    return gauge_coord_to_index(coord[0], coord[1], c1, c2, mu);
  }

  inline int gauge_coord_to_index(int i, int c1, int c2, int mu)
  {
    return mu*size_cm + cm_coord_to_index(i,c1,c2);
  }
  
  // Switch from an (x,y,nc,nc,\pm mu) coordinate to an even-odd partition index.
  inline int hopping_coord_to_index(int x, int y, int c1, int c2, int mu)
  {
    return mu*size_cm + cm_coord_to_index(x,y,c1,c2);
  }
  
  inline int hopping_coord_to_index(int* coord, int c1, int c2, int mu)
  {
    return hopping_coord_to_index(coord[0], coord[1], c1, c2, mu);
  }

  inline int hopping_coord_to_index(int i, int c1, int c2, int mu)
  {
    return mu*size_cm + cm_coord_to_index(i,c1,c2);
  }
  
  // Switch from an (x,y,nc,nc,munu) coordinate to an even-odd partition index.
  inline int corner_coord_to_index(int x, int y, int c1, int c2, int munu)
  {
    return munu*size_cm + cm_coord_to_index(x,y,c1,c2);
  }
  
  inline int corner_coord_to_index(int* coord, int c1, int c2, int munu)
  {
    return corner_coord_to_index(coord[0], coord[1], c1, c2, munu);
  }

  inline int corner_coord_to_index(int i, int c1, int c2, int munu)
  {
    return munu*size_cm + cm_coord_to_index(i,c1,c2);
  }

  ///////////////////////////////////////////////////////////////
  // FUNCTIONS TO GO FROM ONE INDEX TO ANOTHER WITH EXTRA INFO //
  ///////////////////////////////////////////////////////////////

  // Switch from a volume index + dof coordinate to cv index
  inline int vol_index_dof_to_cv_index(int vol_index, int c)
  {
    return nc*vol_index + c; 
  }
  
  /////////////////////////////////////////////////
  // FUNCTIONS TO GO FROM INDICES TO COORDINATES //
  /////////////////////////////////////////////////
  
  // Switch from an even-odd partitioned index to an x,y coordinate.
  inline void index_to_coord(int i, int& x, int& y)
  {
    if (volume == 1) { x = y = 0; return; } // cover special case.
    int parity = (i / (volume/2)); // 0 if even, 1 if odd.
    y = i/(dims[0]/2) - parity*dims[1];
    x = 2*(i % (dims[0]/2)) + (y%2 + parity)%2;
  }
  
  inline void index_to_coord(int i, int* coord)
  {
    index_to_coord(i, coord[0], coord[1]);
  }

  // Switch from an arbitrary dof index to an x,y,dof coordinate.
  inline void dof_index_to_coord(int i, int total_dof, int &x, int& y, int& dof)
  {
    index_to_coord(i / total_dof, x, y);
    dof = i % total_dof;
  }
  
  inline void dof_index_to_coord(int i, int total_dof, int* coord, int& dof)
  {
    dof_index_to_coord(i, total_dof, coord[0], coord[1], dof);
  }
  
  // Switch from a color vector index to an x,y,c coordinate.
  inline void cv_index_to_coord(int i, int &x, int& y, int& c)
  {
    index_to_coord(i / nc, x, y);
    c = i % nc;
  }
  
  inline void cv_index_to_coord(int i, int* coord, int& c)
  {
    cv_index_to_coord(i, coord[0], coord[1], c);
  }
  
  // Switch from a color matrix index to an x,y,c1,c2 coordinate.
  inline void cm_index_to_coord(int i, int &x, int& y, int& c1, int& c2)
  {
    cv_index_to_coord(i / nc, x, y, c1);
    c2 = i % nc;
  }
  
  inline void cm_index_to_coord(int i, int* coord, int& c1, int& c2)
  {
    cm_index_to_coord(i, coord[0], coord[1], c1, c2);
  }
  
  // Switch from a gauge index to an x,y,c1,c2,mu coordinate.
  inline void gauge_index_to_coord(int i, int &x, int& y, int& c1, int& c2, int& mu)
  {
    mu = i / size_cm; 
    cm_index_to_coord(i - mu*size_cm, x, y, c1, c2);
  }
  
  inline void gauge_index_to_coord(int i, int* coord, int& c1, int& c2, int& mu)
  {
    gauge_index_to_coord(i, coord[0], coord[1], c1, c2, mu);
  }
  
  // Switch from a hopping index to an x,y,c1,c2,mu coordinate.
  inline void hopping_index_to_coord(int i, int &x, int& y, int& c1, int& c2, int& mu)
  {
    mu = i / size_cm;
    cm_index_to_coord(i - mu*size_cm, x, y, c1, c2);
  }
  
  inline void hopping_index_to_coord(int i, int* coord, int& c1, int& c2, int& mu)
  {
    hopping_index_to_coord(i, coord[0], coord[1], c1, c2, mu);
  }
  
  // Switch from a corner index to an x,y,c1,c2,munu coordinate.
  inline void corner_index_to_coord(int i, int &x, int& y, int& c1, int& c2, int& munu)
  {
    munu = i / size_cm;
    cm_index_to_coord(i - munu*size_cm, x, y, c1, c2);
  }
  
  inline void corner_index_to_coord(int i, int* coord, int& c1, int& c2, int& munu)
  {
    corner_index_to_coord(i, coord[0], coord[1], c1, c2, munu);
  }
  
  /////////////////////////////////////////////
  // FUNCTIONS TO FIND OUT IF A SITE IS EVEN //
  /////////////////////////////////////////////
  
  inline bool index_is_even(int i)
  {
    return i > (volume / 2);
  }
  
  inline bool cv_index_is_even(int i)
  {
    return i > (size_cv / 2);
  }
  
  inline bool cm_index_is_even(int i)
  {
    return i > (size_cm / 2);
  }
  
  inline bool gauge_index_is_even(int i)
  {
    return i > (size_gauge / 2);
  }
  
  inline bool hopping_index_is_even(int i)
  {
    return i > (size_hopping / 2);
  }
  
  inline bool corner_index_is_even(int i)
  {
    return i > (size_corner / 2);
  }
  
  inline bool coord_is_even(int x, int y)
  {
    return (x+y)%2 == 0;
  }
  
  ////////////////////////////////////////////////
  // FUNCTIONS TO ACCESS INFO ABOUT THE LATTICE //
  ////////////////////////////////////////////////
  
  inline void get_dim(int* dims)
  {
    dims[0] = this->dims[0];
    dims[1] = this->dims[1];
  }
  
  // Return the length of dimension mu.
  inline int get_dim_mu(int mu)
  {
    if (mu >= 0 && mu < nd)
    {
      return dims[mu];
    }
    else
    {
      return -1;
    }
  }
  
  inline int get_nd()
  {
    return nd;
  }
  
  inline int get_nc()
  {
    return nc;
  }

  inline int get_nc_nc()
  {
    return nc*nc;
  }
  
  inline int get_volume()
  {
    return volume;
  }
  
  inline int get_size_dof(int total_dof)
  {
    return volume*total_dof; 
  }

  inline int get_size_cv()
  {
    return size_cv;
  }
  
  inline int get_size_cm()
  {
    return size_cm;
  }
  
  inline int get_size_gauge()
  {
    return size_gauge;
  }
  
  inline int get_size_hopping()
  {
    return size_hopping;
  }
  
  inline int get_size_corner()
  {
    return size_corner;
  }
  
};

#endif // QMG_LATTICE_HEADER
