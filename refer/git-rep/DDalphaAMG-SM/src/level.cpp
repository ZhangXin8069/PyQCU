#include "level.h"

void Level::makeType(const int dofs, 
	MPI_Datatype& local_domain, MPI_Datatype& local_domain_resized,
	MPI_Datatype& coarse_domain, MPI_Datatype& coarse_domain_resized){
	//Data type for the elements of a spinor inside a rank. 
    //The datatype does not include the halo, but assumes the spinor to be sent has it
    MPI_Type_vector(Nx,Nt*dofs,dofs*(Nt+2),MPI_DOUBLE_COMPLEX, &local_domain);
    MPI_Type_commit(&local_domain);

    //The displacement of local_domain_resized is in units of std::complex<double>
    MPI_Type_create_resized(local_domain, 0, sizeof(std::complex<double>), &local_domain_resized);
    MPI_Type_commit(&local_domain_resized);

    // Gather inner domains from all ranks in the coarse communicator
    // Buffer has size (Nx_coarse_rank+2)*(Nt_coarse_rank+2)*DOF
    // Create a type that matches the global buffer layout (strided by full global row including halo)
    MPI_Type_vector(Nx,                 // number of rows to place per rank
        dofs * Nt,               		// elements per row (complex numbers)
        dofs * (Nt_coarse_rank + 2),  	// stride between rows in global buffer (complex elements) including halo
            MPI_DOUBLE_COMPLEX,
            &coarse_domain);
    MPI_Type_commit(&coarse_domain);

    // Resize type so displacements are specified in units of one complex element
    MPI_Type_create_resized(coarse_domain, 0, sizeof(std::complex<double>), &coarse_domain_resized);
    MPI_Type_commit(&coarse_domain_resized);
}

void Level::makeDatatypes(){
	//Data types for spinors
	makeType(DOF,local_domain_spinor,local_domain_spinor_resized,coarse_domain_spinor,coarse_domain_spinor_resized);    
	//Data types for coarse links
	makeType(DOF*DOF,local_domain_linkG1,local_domain_linkG1_resized,coarse_domain_linkG1,coarse_domain_linkG1_resized);  
	makeType(DOF*DOF*2,local_domain_linkG2G3,local_domain_linkG2G3_resized,coarse_domain_linkG2G3,coarse_domain_linkG2G3_resized);  
	// Prepare counts and displacements (displacements in complex-element units)
    for (int r = 0; r < mpi::size_c; r++) {
        counts_spinor[r] = 1; // one instance of recv_domain_resized per contributing rank
		counts_G1[r] = 1;
		counts_G2G3[r] = 1;
        int rx = r / mpi::ranks_t_c; // coarse-group x coordinate
        int rt = r % mpi::ranks_t_c; // coarse-group t coordinate
        // Global starting position inside the buffer including halo (halo at index 0)
        int global_x_start = rx * Nx + 1; // +1 to skip halo
        int global_t_start = rt * Nt + 1; // +1 to skip halo
        // Displacement in complex-element units into buffer.val (including halo padding)
        displs_spinor[r] 	= (global_x_start * (Nt_coarse_rank + 2) + global_t_start) * DOF;
		displs_G1[r] 		= (global_x_start * (Nt_coarse_rank + 2) + global_t_start) * DOF*DOF;
		displs_G2G3[r] 		= (global_x_start * (Nt_coarse_rank + 2) + global_t_start) * DOF*DOF*2;
    }
}

void Level::gather_to_coarse_rank(const spinor& local_spinor, spinor& coarse_spinor, const int dofs){
	int commID = mpi::rank_dictionary[mpi::rank2d];  
    int input_ini_local = dofs * (Nt + 2 + 1); // start of [1,1] in local input (complex elements)
    static int root_rank = 0;  //Root rank inside each communicator agglomerating ranks

	if (dofs == DOF)
    	MPI_Gatherv(&local_spinor.val[input_ini_local],1,local_domain_spinor,&coarse_spinor.val[0],
			counts_spinor,displs_spinor,coarse_domain_spinor_resized,root_rank, mpi::coarse_comm[commID]);
	else if (dofs == DOF*DOF)
    	MPI_Gatherv(&local_spinor.val[input_ini_local],1,local_domain_linkG1,&coarse_spinor.val[0],
			counts_G1,displs_G1,coarse_domain_linkG1_resized,root_rank, mpi::coarse_comm[commID]);
	else if (dofs == DOF*DOF*2)
    	MPI_Gatherv(&local_spinor.val[input_ini_local],1,local_domain_linkG2G3,&coarse_spinor.val[0],
			counts_G2G3,displs_G2G3,coarse_domain_linkG2G3_resized,root_rank, mpi::coarse_comm[commID]);
	else
		std::cout << "Give a valid number of DOFs in function gather_to_coarse_rank" << std::endl;
	

}
void Level::scatter_to_local_rank_from_coarse_rank(const spinor& coarse_spinor, spinor& local_spinor,const int dofs){
	int commID = mpi::rank_dictionary[mpi::rank2d];  
    int input_ini_local = dofs * (Nt + 2 + 1); // start of [1,1] in local input (complex elements)
    static int root_rank = 0;  //Root rank inside each communicator agglomerating ranks
	
	if (dofs == DOF)
    	MPI_Scatterv(&coarse_spinor.val[0], counts_spinor, displs_spinor, coarse_domain_spinor_resized, &local_spinor.val[input_ini_local],1,
        	local_domain_spinor_resized, root_rank, mpi::coarse_comm[commID]);
	else if (dofs == DOF*DOF)
		MPI_Scatterv(&coarse_spinor.val[0], counts_G1, displs_G1, coarse_domain_linkG1_resized, &local_spinor.val[input_ini_local],1,
        	local_domain_linkG1_resized, root_rank, mpi::coarse_comm[commID]);
	else if (dofs == DOF*DOF*2)
		MPI_Scatterv(&coarse_spinor.val[0], counts_G2G3, displs_G2G3, coarse_domain_linkG2G3_resized, &local_spinor.val[input_ini_local],1,
        	local_domain_linkG2G3_resized, root_rank, mpi::coarse_comm[commID]);
	else
		std::cout << "Give a valid number of DOFs in function scatter_to_local_rank_from_coarse_rank" << std::endl;
}


//Prolongator times a vector on the coarse grid
void Level::P_vc(const spinor& vc,spinor& out){
	for(int i = 0; i < (Nx+2)*(Nt+2)*colors*2; i++)
		out.val[i] = 0.0;

	//Botch vc and out include their corresponding halo 
	int bx, bt, bx_shifted, bt_shifted;
	//(bx,bt) are the block coordinates without halo
	//(bx_shifted ,bt_shifted) consider the halo
	int xini, tini, xfin, tfin;
	int idxout, idxv; //Vectorized index of out, v.
	
	std::vector<spinor>* tv = &tvec; 
	spinor* v = &out;
	if (ranks_per_block > 1){
		for(int cc=0;cc<Ntest;cc++)
			gather_to_coarse_rank(tvec[cc],gathered_tvec[cc],DOF);
		
		tv = &gathered_tvec;
		v  = &gathered_out;
		for(int i = 0; i<(Nt_coarse_rank+2)*(Nx_coarse_rank+2)*2*colors; i++)
			v->val[i] = 0;
	}

	for(int b = 0; b<blocks_per_coarse_rank; b++){
		bx = b / tblocks_per_coarse_rank;
		bt = b % tblocks_per_coarse_rank;
		bx_shifted = bx+1;
		bt_shifted = bt+1;
		//Coordinates inside the block (bx,bt) for a spinor with halo
		xini = x_elements*bx+1; xfin = xini + x_elements;
		tini = t_elements*bt+1; tfin = tini + t_elements;
		for(int cc = 0; cc < Ntest; cc++){
			for(int x=xini; x<xfin; x++){
			for(int t=tini; t<tfin; t++){	
			for(int c=0; c<colors; c++){
			for(int s=0; s<2; s++){
				idxout 	= x*(Nt_coarse_rank+2)*colors*2 					+ t*colors*2 		 + c*2 	+ s;
				idxv 	= bx_shifted*(tblocks_per_coarse_rank+2)*Ntest*2 	+ bt_shifted*Ntest*2 + cc*2 + s;
				v->val[idxout] += (*tv)[cc].val[idxout] * vc.val[idxv]; 
				localFLOPS += ca+cm;
			}
			}
			}
			}
		}
	}

	if (ranks_per_block > 1)
		scatter_to_local_rank_from_coarse_rank(gathered_out,out,DOF);
}
	

//Restriction operator times a spinor on the fine grid
void Level::Pdagg_v(const spinor& v,spinor& out) {
	//out lives on the coarse grid
	int bx, bt, bx_shifted, bt_shifted;
	int xini, tini, xfin, tfin;
	int idxout, idxv; //Vectorized index of out, v.

	for(int i = 0; i < (xblocks_per_coarse_rank+2)*(tblocks_per_coarse_rank+2)*2*Ntest; i++)
		out.val[i]= 0.0; //Initialize the output spinor


	std::vector<spinor>* tv = &tvec; 
	const spinor* vf = &v;

	if (ranks_per_block > 1){	
		for(int cc=0;cc<Ntest;cc++)
			gather_to_coarse_rank(tvec[cc],gathered_tvec[cc],DOF);
		gather_to_coarse_rank(v,gathered_v,DOF);

		vf = &gathered_v;
		tv = &gathered_tvec;
	}
	for (int b = 0; b<blocks_per_coarse_rank; b++) {	
		bx = b / tblocks_per_coarse_rank;
		bt = b % tblocks_per_coarse_rank; 
		bx_shifted = bx+1;
		bt_shifted = bt+1;
		xini = x_elements*bx+1; xfin = xini + x_elements;
		tini = t_elements*bt+1; tfin = tini + t_elements;
		for(int cc=0; cc<Ntest; cc++){
			for(int x=xini; x<xfin; x++){
			for(int t=tini; t<tfin; t++){	
			for(int c=0; c<colors; c++){
			for(int s=0; s<2; s++){
				idxout 	= bx_shifted*(tblocks_per_coarse_rank+2)*Ntest*2 	+ bt_shifted*Ntest*2 + cc*2 + s;
				idxv 	= x*(Nt_coarse_rank+2)*colors*2 					+ t*colors*2 + c*2 	+ s;
				out.val[idxout] += std::conj((*tv)[cc].val[idxv]) * vf->val[idxv];
				localFLOPS += ca+cm;
			}	
			}
			}
			}
		}
	}

}
	

//Local orthonormalization.
void Level::orthonormalize(){	
    //Given a set of test vectors, returns a local orthonormalization.
    //For the set of test vectors (v1^(1)|v2^(1)|...|v_Nv^(1)|...|v1^(Nagg)|...|v_Nv^(Nagg)), we
    //orthonormalize the sets {v1^{a}, ..., vn^(a)} for each aggregate.
    //This follows the steps from Section 3.1 of A. Frommer et al "Adaptive Aggregation-Based Domain Decomposition 
    //Multigrid for the Lattice Wilson-Dirac Operator", SIAM, 36 (2014).

	//Orthonormalization by applying Gram-Schmidt
	c_double proj; 
	c_double norm;

	int bx, bt, bx_shifted, bt_shifted, xini, xfin, tini, tfin;
	int indx;
	
	std::vector<spinor>* tv = &tvec;

	if (ranks_per_block>1){
		for(int cc=0;cc<Ntest;cc++)
			gather_to_coarse_rank(tvec[cc],gathered_tvec[cc],DOF);
		tv = &gathered_tvec;
	}

	for (int b = 0; b<blocks_per_coarse_rank; b++) {	
		bx = b / tblocks_per_coarse_rank;
		bt = b % tblocks_per_coarse_rank; 
		//----Coordinates of the elements inside block----//
		xini = x_elements*bx+1; xfin = xini + x_elements;
		tini = t_elements*bt+1; tfin = tini + t_elements; 
		//-----------------------------------------------//
		//Spin defining the aggregate for a particular block
		for (int s = 0; s < 2; s++) {
			//Looping over the test vectors
			for (int nt = 0; nt < Ntest; nt++) {
				for (int ntt = 0; ntt < nt; ntt++) {
					proj = 0;
					for(int x=xini; x<xfin; x++){
					for(int t=tini; t<tfin; t++){	
					for(int c=0; c<colors; c++){
						indx = x*(Nt_coarse_rank+2)*colors*2 + t*colors*2 + c*2 + s;
						proj += (*tv)[nt].val[indx] * std::conj((*tv)[ntt].val[indx]);
						localFLOPS += ca+cm;

					}
					}
					}
					for(int x=xini; x<xfin; x++){
					for(int t=tini; t<tfin; t++){	
					for(int c=0; c<colors; c++){
						indx = x*(Nt_coarse_rank+2)*colors*2 + t*colors*2 + c*2 + s;
						(*tv)[nt].val[indx] -= proj * (*tv)[ntt].val[indx];
						localFLOPS += ca+cm;

					}
					}
					}
				}
				//normalize
				norm = 0.0;
				for(int x=xini; x<xfin; x++){
				for(int t=tini; t<tfin; t++){	
				for(int c=0; c<colors; c++){
					indx = x*(Nt_coarse_rank+2)*colors*2 + t*colors*2 + c*2 + s;
					norm += (*tv)[nt].val[indx] * std::conj((*tv)[nt].val[indx]);
					localFLOPS += ca+cm;

				}
				}
				}
				norm = sqrt(std::real(norm)) + 0.0*I_number;
				localFLOPS += dsq;
				for(int x=xini; x<xfin; x++){
				for(int t=tini; t<tfin; t++){	
				for(int c=0; c<colors; c++){
					indx = x*(Nt_coarse_rank+2)*colors*2 + t*colors*2 + c*2 + s;
					if (std::abs(norm) > 1e-8){
						(*tv)[nt].val[indx] /= norm;	//For the ranks that don't gather the tv, norm is zero.
						localFLOPS += cd;
					}

				}
				}
				} 
				
			}
		} 	
	}
		
	if (ranks_per_block > 1){
		for(int cc=0;cc<Ntest;cc++)
			scatter_to_local_rank_from_coarse_rank(gathered_tvec[cc],tvec[cc],DOF);
	}
}

void Level::makeDirac(){
	c_double P[2][2][2], M[2][2][2]; 
	//P = 1 + sigma
	P[0][0][0] = 1.0; P[0][0][1] = 1.0;
	P[0][1][0] = 1.0; P[0][1][1] = 1.0; 

	P[1][0][0] = 1.0; P[1][0][1] = -I_number;
	P[1][1][0] = I_number; P[1][1][1] = 1.0; 

	//M = 1- sigma
	M[0][0][0] = 1.0; M[0][0][1] = -1.0;
	M[0][1][0] = -1.0; M[0][1][1] = 1.0; 

	M[1][0][0] = 1.0; M[1][0][1] = I_number;
	M[1][1][0] = -I_number; M[1][1][1] = 1.0; 

	int n;
	exchange_halo(U.val);

	for(int x = 1; x<=Nx; x++){
	for(int t = 1; t<=Nt; t++){
		n = x*(Nt+2)+t;
	for(int alf=0; alf<2;alf++){
	for(int bet=0; bet<2;bet++){
	for(int c = 0; c<colors; c++){
	for(int b = 0; b<colors; b++){
		G1.val[getG1index(n,alf,bet,c,b)] 	  = 0; //This coefficient is not used at level 0
		G2.val[getG2G3index(n,alf,bet,c,b,0)] = 0; G2.val[getG2G3index(n,alf,bet,c,b,1)] = 0;
		G3.val[getG2G3index(n,alf,bet,c,b,0)] = 0; G3.val[getG2G3index(n,alf,bet,c,b,1)] = 0;
		//For level = 0 
		for(int mu : {0,1}){
			G2.val[getG2G3index(n,alf,bet,c,b,mu)] = 0.5 * M[mu][alf][bet] * U.val[2*n+mu];
			G3.val[getG2G3index(n,alf,bet,c,b,mu)] = 0.5 * P[mu][alf][bet] * std::conj(U.val[2*lpb[2*n+mu]+mu]);
			localFLOPS += (dcm+cm)*2;
		}
		
	}
	}
	}
	}

	}
	}
	
		
}


//Exchange halo for spinor v among the working ranks at the current level. Level>=1. For level=0 we use the other
//halo exchange in the Dirac operator
void Level::exchange_halo_l(const spinor& v,const int& Nx, const int& Nt, const MPI_Datatype& column, const MPI_Comm& comm){
    int row_size = DOF * Nt;
	int send_start, recv_start;   
	int bot, top, left, right;
	int result;
	//I cannot use Comm_compare when comm = MPI_COMM_NULL
   
	//If we have the communicator of the fine grid
	if (comm != MPI_COMM_NULL){
		MPI_Comm_compare(comm, mpi::cart_comm, &result);
    	if (result == MPI_IDENT) {
			bot 	= mpi::bot;
			top 	= mpi::top;
			left 	= mpi::left;
			right 	= mpi::right;
    	}
		else{
			//In this case we call comm_coarse_level
			bot 	= mpi::bot_c;
			top 	= mpi::top_c;
			left 	= mpi::left_c;
			right 	= mpi::right_c;
		}

    	//Send top row to top rank. Receive top row from bot rank.
		//(x*(Nt+2)+t)*DOF +dof
	
		send_start = ((Nt+2) + 1)*DOF;   	//x=1, t=1
		recv_start = ((Nx+1)*(Nt+2)+1)*DOF;	//x=Nx+1, t=1
    	MPI_Sendrecv(&v.val[send_start], row_size, MPI_DOUBLE_COMPLEX, top, 0,
        	&v.val[recv_start], row_size, MPI_DOUBLE_COMPLEX, bot, 0,
        	comm, MPI_STATUS_IGNORE);

    	//Send bot row to bot rank. Receive bot row from top rank.
		send_start = (Nx*(Nt+2)+1)*DOF;	//x=Nx, t=1
		recv_start = 1*DOF;				//x=0,	t=1
    	MPI_Sendrecv(&v.val[send_start], row_size, MPI_DOUBLE_COMPLEX, bot, 1,
        	&v.val[recv_start], row_size, MPI_DOUBLE_COMPLEX, top, 1,
        	comm, MPI_STATUS_IGNORE);

    	//Send left column to left rank. Receive left column from right rank. 
		send_start = ((Nt+2) + 1)*DOF;		//x=1,	t=1
		recv_start = ((Nt+2)+(Nt+1))*DOF;	//x=1,	t=Nt+1
    	MPI_Sendrecv(&v.val[send_start], 1, column, left, 2,
    		&v.val[recv_start], 1, column, right, 2,
    		comm, MPI_STATUS_IGNORE);

    	//Send right column to right rank. Receive right column from left rank. 
		send_start = ((Nt+2)+Nt)*DOF;		//x=1,	t=Nt
		recv_start = (Nt+2)*DOF;			//x=1,	t=0
    	MPI_Sendrecv(&v.val[send_start], 1, column, right, 3,
    		&v.val[recv_start], 1, column, left, 3,
    		comm, MPI_STATUS_IGNORE);
	}
}


//Dirac operator at the current level
void Level::D_operator(const spinor& v, spinor& out){	
	exchange_halo_l(v,Nx,Nt,mpi::column_type[level],ranks_comm);
	int indx, indx1, indx2, n;
	//n only runs in the interior of the lattice domain
	for(int x = 1; x<=Nx; x++){
	for(int t = 1; t<=Nt; t++){
		n = x*(Nt+2)+t;
		for(int alf = 0; alf<2; alf++){
		for(int c = 0; c<colors; c++){
			indx = n*colors*2+c*2+alf;
			out.val[indx] = (mass::m0+2)*v.val[indx];
			localFLOPS += da+dcm;
		for(int bet = 0; bet<2; bet++){
		for(int b = 0; b<colors; b++){
			indx1 = n*colors*2+b*2+bet;
			out.val[indx] -= G1.val[getG1index(n,alf,bet,c,b)] * v.val[indx1];
			localFLOPS += ca+cm;
			for(int mu:{0,1}){
				indx1 = rpb_l(x,t,mu,Nx,Nt)*colors*2+b*2+bet;
				indx2 = lpb_l(x,t,mu,Nx,Nt)*colors*2+b*2+bet;
				out.val[indx] -= ( 	G2.val[getG2G3index(n,alf,bet,c,b,mu)] * rsign_l(t,mu) * v.val[indx1]
								+ 	G3.val[getG2G3index(n,alf,bet,c,b,mu)] * lsign_l(t,mu) * v.val[indx2] 
								);
				localFLOPS += ca +ca + 4*cm;
			}
		}
		}
		}
		}
	}
	}
	
}

/*
	Make coarse gauge links. They will be used in the next level as G1, G2 and G3.
*/
void Level::makeCoarseLinks(Level& next_level){
	//Make gauge links for level l
	std::vector<spinor>* w = &tvec;
	spinor* g1 = &G1;
	spinor* g2 = &G2;
	spinor* g3 = &G3;
	if (ranks_per_block > 1){
		for(int cc=0; cc<Ntest;cc++)
			gather_to_coarse_rank(tvec[cc],gathered_tvec[cc],DOF); //DOFs should be a parameter
		w = &gathered_tvec;

		gather_to_coarse_rank(G1,gathered_G1,DOF*DOF);
		gather_to_coarse_rank(G2,gathered_G2,2*DOF*DOF);
		gather_to_coarse_rank(G3,gathered_G3,2*DOF*DOF);
		g1 = &gathered_G1;
		g2 = &gathered_G2;
		g3 = &gathered_G3;
		for(int cc=0; cc<Ntest;cc++)
			exchange_halo_l((*w)[cc],Nx_coarse_rank,Nt_coarse_rank,coarse_column_type,mpi::comm_coarse_level);	//We exchange halos for the test vectors.
	}
	else{
		for(int cc=0; cc<Ntest;cc++)
			exchange_halo_l((*w)[cc],Nx,Nt,mpi::column_type[level],ranks_comm);
	}
	
	c_double wG2, wG3;
	spinor &A_coeff = next_level.G1; 
	spinor &B_coeff = next_level.G2;
	spinor &C_coeff = next_level.G3;
	int indxA; int indxBC[2]; //Indices for A, B and C coefficients
	int block_r;
	int block_l;
	//p and s are the colors at the coarse level
	//c and b are the colors at the current level
	int bx, bt, bx_shifted, bt_shifted, block;
	int xini, xfin, tini, tfin;
	int n; 
	int indx;
	int rn, ln; //right and left neighbors
	int rb, lb; //right and left blocks
	for(int b = 0; b<blocks_per_coarse_rank; b++){
		bx = b / tblocks_per_coarse_rank;
		bt = b % tblocks_per_coarse_rank;	
		bx_shifted = bx+1;
		bt_shifted = bt+1;
		block = (bx_shifted)*(tblocks_per_coarse_rank+2) + bt_shifted;//Indexing for a coarse spinors with halo
		//Coordinates inside the block (bx,bt) for a spinor with halo
		xini = x_elements*bx+1; xfin = xini + x_elements;
		tini = t_elements*bt+1; tfin = tini + t_elements;
	for(int alf=0; alf<2; alf++){
	for(int bet=0; bet<2; bet++){
	for(int p = 0; p<Ntest; p++){
	for(int s = 0; s<Ntest; s++){
		indxA 		= getAindex(block,alf,bet,p,s); 	//Indices for the next level
		indxBC[0] 	= getBCindex(block,alf,bet,p,s,0);
		indxBC[1] 	= getBCindex(block,alf,bet,p,s,1);
		A_coeff.val[indxA] = 0;
		B_coeff.val[indxBC[0]] = 0; B_coeff.val[indxBC[1]] = 0;
		C_coeff.val[indxBC[0]] = 0; C_coeff.val[indxBC[1]] = 0;
		//Elements inside the lattice blocks
		for(int x=xini;x<xfin;x++){
		for(int t=tini;t<tfin;t++){
			n = x*(Nt_coarse_rank+2)+t;
			for(int c = 0; c<colors; c++){
			for(int b = 0; b<colors; b++){
				//[w*_p^(block,alf)]_{c,alf}(x) [A(x)]^{alf,bet}_{c,b} [w_s^{block,bet}]_{b,bet}(x)
				indx = n*colors*2 + c*2 + alf;
				A_coeff.val[indxA] += std::conj((*w)[p].val[indx]) * g1->val[getG1index(n,alf,bet,c,b)] * (*w)[s].val[n*colors*2 + b*2 + bet];
				localFLOPS += ca+cm*2;
				for(int mu : {0,1}){
					rn = rpb_l(x,t,mu,Nx_coarse_rank,Nt_coarse_rank); //(x,t)+hat{mu}
					ln = lpb_l(x,t,mu,Nx_coarse_rank,Nt_coarse_rank); //(x,t)-hat{mu}
					rb = next_level.rpb_l(bx_shifted,bt_shifted,mu,next_level.Nx,next_level.Nt); //(bx,bt)+hat{mu}
					lb = next_level.lpb_l(bx_shifted,bt_shifted,mu,next_level.Nx,next_level.Nt); //(bx,bt)-hat{mu}
					getLatticeBlock(rn, block_r); //block_r: block where rn lives
					getLatticeBlock(ln, block_l); //block_l: block where ln lives

					wG2 = std::conj((*w)[p].val[indx]) * g2->val[getG2G3index(n,alf,bet,c,b,mu)]; 
					wG3 = std::conj((*w)[p].val[indx]) * g3->val[getG2G3index(n,alf,bet,c,b,mu)];
					localFLOPS += 2*cm;

					//Only diff from zero when n+hat{mu} in Block(x)
					if (block_r == block){
						A_coeff.val[indxA] 			+= wG2 * (*w)[s].val[rn*colors*2 + b*2 + bet];
						localFLOPS += ca+cm;
					}
					//Only diff from zero when n+hat{mu} in Block(x+hat{mu})
					else if (block_r == rb){
						B_coeff.val[indxBC[mu]] 	+= wG2 * (*w)[s].val[rn*colors*2 + b*2 + bet]; 
						localFLOPS += ca+cm;
					}
					//Only diff from zero when n-hat{mu} in Block(x)
					if (block_l == block){
						A_coeff.val[indxA] 			+= wG3 * (*w)[s].val[ln*colors*2 + b*2 + bet];
						localFLOPS += ca+cm;
					}
					//Only diff from zero when n-hat{mu} in Block(x-hat{mu})
					else if (block_l == lb){
						C_coeff.val[indxBC[mu]] 	+= wG3 * (*w)[s].val[ln*colors*2 + b*2 + bet];
						localFLOPS += ca+cm;
					}
				}//mu loop
			}//b loop
			}//c loop
		}//x inside block
		}//t inside block	
	} //s
	} //p
	} //bet
	} //alf
	} //block
}

void Level::checkOrthogonality() {
	int bx, bt, xini, xfin, tini, tfin;
	c_double dot_product;
	int indx;
	int coarse_rank=0;

	std::vector<spinor>* tv = &tvec;
	if (ranks_per_block > 1){
		//clean_gathered_tvec();
		for(int cc=0;cc<Ntest;cc++)
			gather_to_coarse_rank(tvec[cc],gathered_tvec[cc],DOF);
		tv = &gathered_tvec;
		int commID = mpi::rank_dictionary[mpi::rank2d];
		MPI_Comm_rank(mpi::coarse_comm[commID], &coarse_rank);
	}

	if ( (coarse_rank == 0 && ranks_per_block>1) || (ranks_per_block<1) ){
	for (int b = 0; b<blocks_per_coarse_rank; b++) {	
		bx = b / tblocks_per_coarse_rank;
		bt = b % tblocks_per_coarse_rank; 
		//----Coordinates of the elements inside block----//
		xini = x_elements*bx+1; xfin = xini + x_elements;
		tini = t_elements*bt+1; tfin = tini + t_elements; 
		//-----------------------------------------------//
		//Spin defining the aggregate for a particular block
		for (int s = 0; s < 2; s++) {
			//Looping over the test vectors
			for (int nt = 0; nt < Ntest; nt++) {
			for (int ntt = 0; ntt < Ntest; ntt++) {
				dot_product = 0.0;
				for(int x=xini; x<xfin; x++){
				for(int t=tini; t<tfin; t++){	
				for(int c=0; c<colors; c++){
					indx = x*(Nt_coarse_rank+2)*colors*2 + t*colors*2 + c*2 + s;
					dot_product += std::conj((*tv)[nt].val[indx]) * (*tv)[ntt].val[indx]; //v_nt . v_ntt
				}
				}
				}
				if (std::abs(dot_product) > 1e-8 && nt!=ntt) {
					if (mpi::rank2d == 0){
						std::cout << "Block " << b << " spin " << s << std::endl;
						std::cout << "Level " << level << std::endl;
						std::cout << "Test vectors " << nt << " and " << ntt << " are not orthogonal: " << dot_product << std::endl;
					}
					//exit(1);
				}
				else if(std::abs(dot_product-1.0) > 1e-8 && nt==ntt){
					if (mpi::rank2d == 0){
						std::cout << "Level " << level << std::endl;
						std::cout << "Test vector " << nt << " not normalized " << dot_product << std::endl;
					}
					//exit(1);
				}
			}
			}
		}
	}
	}
	if (mpi::rank2d == 0)
		std::cout << "Test vectors on level " << level << " are orthonormalized " << std::endl;
}

//Local Dc used for SAP
void Level::SAP_level_l::D_local(const spinor& in, spinor& out, const int& block){
    int xp, xm, tp, tm;
	int rpb_mu[2];
	int lpb_mu[2];
    int n, m;
	int indx, indx1, indx2;
	for(int mx = 1; mx <= x_elements; mx++){
    for(int mt = 1; mt <= t_elements; mt++){
		m = mx * (t_elements + 2) + mt; //Lattice coordinate for the block
        n = Blocks[block][m]; //Already inside the domain, no halos

		//Neighbor coordinates inside the block
        xp = mx+1;
        xm = mx-1;
        tp = mt+1;
        tm = mt-1;
        rpb_mu[0]   = mx*(t_elements+2) + tp;   //Right
        rpb_mu[1]   = xp*(t_elements+2) + mt;   //Down
        lpb_mu[0]   = mx*(t_elements+2) + tm;   //Left
        lpb_mu[1]   = xm*(t_elements+2) + mt;   //Up
		for(int alf = 0; alf<2; alf++){
		for(int c = 0; c<colors; c++){
			indx = m*colors*2+c*2+alf;
			out.val[indx] = (mass::m0+2)*in.val[indx];
			localFLOPS += da+dcm;
				for(int bet = 0; bet<2; bet++){
				for(int b = 0; b<colors; b++){
					indx1 = m*colors*2+b*2+bet;
					out.val[indx] -= parent->G1.val[parent->getG1index(n,alf,bet,c,b)] * in.val[indx1];
					localFLOPS += ca+cm;
					for(int mu:{0,1}){
						indx1 = rpb_mu[mu]*colors*2+b*2+bet;
						indx2 = lpb_mu[mu]*colors*2+b*2+bet;
						out.val[indx] -= 
							( parent->G2.val[parent->getG2G3index(n,alf,bet,c,b,mu)] * in.val[indx1]
							+ parent->G3.val[parent->getG2G3index(n,alf,bet,c,b,mu)] * in.val[indx2]
							);
						localFLOPS += ca+ca+2*cm;
					}

				}
				}
			}
			}
	}
	}
}