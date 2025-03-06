// Copyright (c) 2017 Evan S Weinberg
// Test header for a real and complex operator.

#ifndef SQUARE_LAPLACE_HEADER
#define SQUARE_LAPLACE_HEADER

struct laplace_struct
{
  int length;
  double m_sq;
};

struct laplace_gauged_struct
{
  int length;
  double m_sq;
  std::complex<double>* gauge_links; // size 2*length*length
};

void square_laplacian(double* lhs, double* rhs, void* extra_data)
{
  // Declare variables.
  int i;
  int x,y;
  laplace_struct* lapstr = (laplace_struct*)extra_data;

  int length = lapstr->length;
  int volume = lapstr->length*lapstr->length;
  double m_sq = lapstr->m_sq;
  
  // For a 2D square lattice, the stencil is:
  //     |  0 -1  0 |
  //     | -1 +4 -1 |
  //     |  0 -1  0 |
  //
  // e2 = yhat
  // ^
  // | 
  // |-> e1 = xhat

  // Apply the stencil.
  for (i = 0; i < volume; i++)
  {
    lhs[i] = 0.0;
    x = i%length; // integer mod.
    y = i/length; // integer divide.

    // + e1.
    lhs[i] = lhs[i]-rhs[y*length+((x+1)%length)];

    // - e1.
    lhs[i] = lhs[i]-rhs[y*length+((x+length-1)%length)];

    // + e2.
    lhs[i] = lhs[i]-rhs[((y+1)%length)*length+x];

    // - e2.
    lhs[i] = lhs[i]-rhs[((y+length-1)%length)*length+x];

    // 0
    // Added mass term here.
    lhs[i] = lhs[i]+(4+m_sq)*rhs[i];
  }

}

void square_laplacian_gauged(complex<double>* lhs, complex<double>* rhs, void* extra_data)
{
  // Declare variables.
  int i;
  int x,y;
  laplace_gauged_struct* lapstr = (laplace_gauged_struct*)extra_data;

  int length = lapstr->length;
  int volume = lapstr->length*lapstr->length;
  double m_sq = lapstr->m_sq;
  complex<double>* gauge_links = lapstr->gauge_links; 
  
  // For a 2D square lattice, the stencil is:
  //     |  0 -1  0 |
  //     | -1 +4 -1 |
  //     |  0 -1  0 |
  //
  // e2 = yhat
  // ^
  // | 
  // |-> e1 = xhat

  // Apply the stencil.
  for (i = 0; i < volume; i++)
  {
    lhs[i] = 0.0;
    x = i%length; // integer mod.
    y = i/length; // integer divide.

    // + e1.
    lhs[i] = lhs[i]-gauge_links[y*length*2+x*2]*rhs[y*length+((x+1)%length)];

    // - e1.
    lhs[i] = lhs[i]-conj(gauge_links[y*length*2+((x+length-1)%length)*2])*rhs[y*length+((x+length-1)%length)];

    // + e2.
    lhs[i] = lhs[i]-gauge_links[y*length*2+x*2+1]*rhs[((y+1)%length)*length+x];

    // - e2.
    lhs[i] = lhs[i]-conj(gauge_links[((y+length-1)%length)*length*2+x*2+1])*rhs[((y+length-1)%length)*length+x];

    // 0
    // Added mass term here.
    lhs[i] = lhs[i]+(4+m_sq)*rhs[i];
  }
}

void square_staggered(double* lhs, double* rhs, void* extra_data)
{
  // Declare variables.
  int i;
  int x,y;
  double eta1;

  laplace_struct* lapstr = (laplace_struct*)extra_data;

  int length = lapstr->length;
  int volume = lapstr->length*lapstr->length;
  double mass = lapstr->m_sq;

  // For a 2D square lattice, the stencil is:
  //   1 |  0 -eta1  0 |
  //   - | +1    0  -1 |  , where eta1 = (-1)^x
  //   2 |  0 +eta1  0 |
  //
  // e2 = yhat
  // ^
  // | 
  // |-> e1 = xhat

  // Apply the stencil.
  for (i = 0; i < volume; i++)
  {
      lhs[i] = 0.0;
      x = i%length; // integer mod.
      y = i/length; // integer divide.
      eta1 = 1 - 2*(x%2);

      // + e1.
      lhs[i] = lhs[i]-rhs[y*length+((x+1)%length)];

      // - e1.
      lhs[i] = lhs[i]+ rhs[y*length+((x+length-1)%length)]; // The extra +N is because of the % sign convention.

      // + e2.
      lhs[i] = lhs[i]- eta1*rhs[((y+1)%length)*length+x];

      // - e2.
      lhs[i] = lhs[i]+ eta1*rhs[((y+length-1)%length)*length+x];

      // Normalization.
      lhs[i] = 0.5*lhs[i];

      // 0
      // Added mass term here.
      lhs[i] = lhs[i]+ mass*rhs[i];
  }

}

void square_wilson_gauged(complex<double>* lhs, complex<double>* rhs, void* extra_data)
{
  // Declare variables.
  int i;
  int x,y,c;
  laplace_gauged_struct* lapstr = (laplace_gauged_struct*)extra_data;

  int length = lapstr->length;
  int volume = lapstr->length*lapstr->length;
  double mass = lapstr->m_sq;
  complex<double>* gauge_links = lapstr->gauge_links; 
  complex<double> cplxI(0,1);
  
  int lattice_size = 2*volume;

  // Apply the stencil.
  for (i = 0; i < lattice_size; i++)
  {
    lhs[i] = 0.0;
    
    c = i%2;
    x = ((i-c)/2)%length; // integer mod.
    y = ((i-c)/2)/length; // integer divide.

    // + e1. 1/2*(\sigma_x - 1)
    lhs[i] += gauge_links[y*length*2+x*2]*
                (rhs[2*(y*length+((x+1)%length)) + (1-c)]
                 - rhs[2*(y*length+((x+1)%length)) + c]);

    // - e1. 1/2*(-\sigma_x - 1)
    lhs[i] += conj(gauge_links[y*length*2+((x+length-1)%length)*2])*
                (- rhs[2*(y*length+((x+length-1)%length)) + (1-c)]
                 - rhs[2*(y*length+((x+length-1)%length)) + c]);

    // + e2.
    lhs[i] += gauge_links[y*length*2+x*2+1]*
                ((c == 0 ? -1.0 : 1.0)*cplxI*rhs[2*(((y+1)%length)*length+x) + (1-c)]
                 - rhs[2*(((y+1)%length)*length+x) + c]);

    // - e2.
    lhs[i] += conj(gauge_links[((y+length-1)%length)*length*2+x*2+1])*
                (- (c == 0 ? -1.0 : 1.0)*cplxI*rhs[2*(((y+length-1)%length)*length+x) + (1-c)]
                 - rhs[2*(((y+length-1)%length)*length+x) + c]);

    // Normalization.
    lhs[i] = 0.5*lhs[i];

    // 0
    // Added mass term plus 2 from laplace term.
    lhs[i] = lhs[i]+ (mass+2.0)*rhs[i];

  }

}


#endif // SQUARE_LAPLACE_HEADER