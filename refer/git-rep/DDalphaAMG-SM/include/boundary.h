#ifndef BOUNDARY_H
#define BOUNDARY_H
#include "variables.h"


/*
                 		t                    
    0  +------------------------------------+  Nt   
       |                                    |
       |           lpb[2*n+1]    		    |
       |                   				    |
    x  |    lpb[2*n+0]  n  rpb[2*n+0]       |   
       |                       			    |
       |           rpb[2*n+1]        	    |
       |                          			|
    Nx +------------------------------------+ Nt  
*/

inline void boundaries() {	
    using namespace LV; 
	using namespace mpi;
    for (int x = 1; x <= width_x; x++) {
        for (int t = 1; t <= width_t; t++) {
            int xp = x+1;
            int xm = x-1;
            int tp = t+1;
            int tm = t-1;
            int n = x * (width_t+2) + t;

            //Periodic boundary is already considered in the halo exchange
            //Neighbor coordinates
            rpb[2*n]    = x*(width_t+2) + tp;   //Right
            rpb[2*n+1]  = xp*(width_t+2) + t;   //Down

            lpb[2*n]    = x*(width_t+2)+tm;     //Left
            lpb[2*n+1]  = xm*(width_t+2)+t;    //Up

            rsign[2*n] = 1; rsign[2*n+1] = 1;
			lsign[2*n] = 1; lsign[2*n+1] = 1;

			if ((rank+1) % ranks_t == 0){
				rsign[2*n] = (t == width_t) ? -1 : 1;   //sign for the right boundary in time
			} 
			if (rank % ranks_t == 0){
				lsign[2*n] = (t == 1) ? -1 : 1;         //sign for the left boundary in time	
			}
        }
    }


   
}

#endif