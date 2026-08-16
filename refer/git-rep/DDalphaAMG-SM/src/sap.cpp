#include "sap.h"

void SAP_C::SchwarzBlocks(){
    int count, block;
    int x0, t0, x1, t1;
    for (int x = 0; x < Block_x; x++) {
        for (int t = 0; t < Block_t; t++) {
            x0 = x * (x_elements) + 1; t0 = t * (t_elements) + 1;
            x1 = (x + 1) * (x_elements); t1 = (t + 1) * (t_elements);
            block = x * Block_t + t;   //Vectorized block index.
            //Filling the block with the coordinates of the lattice points. No halos considered here.
            count = 0;
            for(int x = x0; x <= x1; x++) {
                for (int t = t0; t <= t1; t++) {
                    count += 1;
                    int m = (x+1-x0) * (t_elements+2) + (t+1-t0); //Coordinates in the block with halo included
                    Blocks[block][m] = x * (Nt+2)+ t;             //Coordinates in the original lattice with halo included
                    //Each block also considers all the degrees of freedom per lattice site, 
                    //so we only reference the lattice coordinates here.
                }
            }
            if (count != Ntot_no_halo) {
                std::cout << "Block " << block << " has " << count << " lattice points" << std::endl;
                std::cout << "Expected " << Ntot_no_halo << std::endl;
                exit(1);
            }
            //Red-black decomposition for the blocks.
            if (Block_t % 2 == 0) {
                if  (x%2 ==0){
                    (block % 2 == 0) ? RedBlocks[block / 2] = block:BlackBlocks[block / 2] = block; 
                }
                else{
                    (block % 2 == 0) ? BlackBlocks[block / 2] = block:RedBlocks[block / 2] = block; 
                }
            } 
            else {
                (block % 2 == 0) ? RedBlocks[block / 2] = block:BlackBlocks[block / 2] = block;                
            }
        }
    }
}



//A_B = I_B * D_B^-1 * I_B^T v --> Extrapolation of D_B^-1 to the original lattice.
//dim(v) = 2 * Ntot, dim(x) = 2 Ntot
//v: input, x: output
void SAP_C::I_D_B_1_It(const spinor& v, spinor& x,const int& block){
    bool print_message = false; //for testing GMRES in local blocks   
    spinor restriced_v(Nvars_w_halo); 
    spinor temp(Nvars_w_halo);
    //restriced_v = I_B^T v
    int m;
    for(int mx = 1; mx <= x_elements; mx++){
    for(int mt = 1; mt <= t_elements; mt++){
        m = mx * (t_elements + 2) + mt; //Lattice coordinate for the block
    for(int dof = 0; dof<dofs; dof++){
        restriced_v.val[dofs*m+dof] = v.val[dofs*Blocks[block][m]+dof];
    }
    }
    }

    current_block = block; //Set the block index for the GMRES_D_B operator

    gmres_DB.fgmres(restriced_v,restriced_v,temp, print_message); //temp = D_B^-1 restricted_v


    //Extrapolate the solution to the original lattice
    for(int mx = 1; mx <= x_elements; mx++){
    for(int mt = 1; mt <= t_elements; mt++){
        m = mx * (t_elements + 2) + mt; 
    for(int dof = 0; dof<dofs; dof++){
        x.val[dofs*Blocks[block][m]+dof] = temp.val[dofs*m+dof];
    }
    }
    }
}



int SAP_C::SAP(const spinor& v,spinor &x,const int& nu, const double&tol, const bool print){  
    double err;
    double v_norm = sqrt(std::real(dot(v, v))); //norm of the right hand side
    localFLOPS += dsq;

    spinor DB_1_r(dofs*Ntot_original_w_halo);  //D_B^-1 r
    spinor r(dofs*Ntot_original_w_halo); //residual
    spinor Dphi(dofs*Ntot_original_w_halo); //D x
    funcGlobal(x,Dphi);

    int n, m;
    for(int nx=1;nx<=Nx;nx++){
    for(int nt=1;nt<=Nt;nt++){
    for(int dof=0;dof<dofs;dof++){
        n = nx*(Nt+2)+nt;
        r.val[dofs*n+dof] = v.val[dofs*n+dof] - Dphi.val[dofs*n+dof];  //r = v - D x
        localFLOPS += ca;
    }
    }
    }


    for (int i = 0; i< nu; i++){
        for (auto block : RedBlocks){
            I_D_B_1_It(r,DB_1_r,block); //getting D_B^-1 r
            for(int mx = 1; mx <= x_elements; mx++){
            for(int mt = 1; mt <= t_elements; mt++){
                m = mx * (t_elements + 2) + mt; 
            for(int dof = 0; dof<dofs; dof++){
                x.val[dofs*Blocks[block][m]+dof] += DB_1_r.val[dofs*Blocks[block][m]+dof]; //x = x + D_B^-1 r
                localFLOPS += ca;
            }
            }
            }
        }
        
        funcGlobal(x,Dphi); //D x
        for(int nx=1;nx<=Nx;nx++){
        for(int nt=1;nt<=Nt;nt++){
        for(int dof=0;dof<dofs;dof++){
                n = nx*(Nt+2)+nt;
                r.val[dofs*n+dof] = v.val[dofs*n+dof] - Dphi.val[dofs*n+dof];  //r = v - D x
                localFLOPS += ca;
        }
        }
        }

        for (auto block : BlackBlocks){
            I_D_B_1_It(r,DB_1_r,block);  //getting D_B^-1 r
            for(int mx = 1; mx <= x_elements; mx++){
            for(int mt = 1; mt <= t_elements; mt++){
                m = mx * (t_elements + 2) + mt; 
            for(int dof = 0; dof<dofs; dof++){
                x.val[dofs*Blocks[block][m]+dof] += DB_1_r.val[dofs*Blocks[block][m]+dof]; //x = x + D_B^-1 r
                localFLOPS += ca;
            }
            }
            }
        }

        funcGlobal(x,Dphi); //D x
        for(int nx=1;nx<=Nx;nx++){
        for(int nt=1;nt<=Nt;nt++){
        for(int dof=0;dof<dofs;dof++){
                n = nx*(Nt+2)+nt;
                r.val[dofs*n+dof] = v.val[dofs*n+dof] - Dphi.val[dofs*n+dof];  //r = v - D x
                localFLOPS += ca;
        }
        }
        }
        //This dot function is a virtual function from SAP_C
        err = sqrt(std::real(dot(r, r))); 
        localFLOPS += dsq;
        if (err < tol * v_norm) {
            if (print == true && mpi::rank2d == 0)
                std::cout << "SAP converged in " << i << " iterations, error: " << err << std::endl;
            return i;
        }
    }

    if (print == true && mpi::rank2d == 0)
        std::cout << "SAP didn't converge in " << nu << " iterations, error: " << err << std::endl;

    return nu; 
}



void SAP_fine_level::D_B(const spinor& U, const spinor& v, spinor& x, const double& m0,const int& block){
    //v and x are both of dimensions (dofs*(x_elements+2)*(t_elements+2)) (they are padded with zeros to account for the boundaries)

    int rpb_0; //Right periodic boundary in the 0-direction
    int rpb_1; //Right periodic boundary in the 1-direction
    int lpb_0; //Left periodic boundary in the 0-direction
    int lpb_1; //Left periodic boundary in the 1-direction
    int xp;
    int xm;
    int tp;
    int tm;

    int n, m;
    for(int mx = 1; mx <= x_elements; mx++){
    for(int mt = 1; mt <= t_elements; mt++){
        m = mx * (t_elements + 2) + mt; //Lattice coordinate for the block
        n = Blocks[block][m];           //Lattice coordinate of m in the original lattice. Needed to call the gauge field.

        //Neighbor coordinates inside the block
        xp = mx+1;
        xm = mx-1;
        tp = mt+1;
        tm = mt-1;
        rpb_0   = mx*(t_elements+2) + tp;   //Right
        rpb_1   = xp*(t_elements+2) + mt;   //Down
        lpb_0   = mx*(t_elements+2) + tm;   //Left
        lpb_1   = xm*(t_elements+2) + mt;   //Up

        //When the local neighbors touch the halo, the value of v.val is zero, effectively removing that contribution.

       //mu = 0
		x.val[2*m] = (m0 + 2) * v.val[2*m] - 0.5 * ( 
				U.val[2*n] 	 * rsign[2*n]   	* (v.val[2*rpb_0] - v.val[2*rpb_0+1])
			+	U.val[2*n+1] * rsign[2*n+1] 	* (v.val[2*rpb_1] + I_number * v.val[2*rpb_1+1])
			+ std::conj(U.val[2*lpb[2*n]]) 		* lsign[2*n] 	* (v.val[2*lpb_0] + v.val[2*lpb_0+1])
			+ std::conj(U.val[2*lpb[2*n+1]+1]) 	* lsign[2*n+1]  *  (v.val[2*lpb_1] - I_number*v.val[2*lpb_1+1])
		);
		//mu = 1
		x.val[2*m+1] = (m0 + 2) * v.val[2*m+1] - 0.5 * ( 
				U.val[2*n] 	 * rsign[2*n] 		* (-v.val[2*rpb_0] + v.val[2*rpb_0+1])
			+	U.val[2*n+1] * rsign[2*n+1] 	* (-I_number*v.val[2*rpb_1] + v.val[2*rpb_1+1])
			+ std::conj(U.val[2*lpb[2*n]]) 		* lsign[2*n] 	* (v.val[2*lpb_0] + v.val[2*lpb_0+1])
			+ std::conj(U.val[2*lpb[2*n+1]+1]) 	* lsign[2*n+1]  * (I_number*v.val[2*lpb_1] + v.val[2*lpb_1+1])
		);

    }
    }
    localFLOPS += 2*81*x_elements*t_elements;
    
}




