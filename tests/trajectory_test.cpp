#include <iostream>

// matrix multiplication
#include "../he_ar/matrix_he_ar.hpp"

#include "parameters.hpp"
#include "file.hpp"

#include "trajectory.hpp"

void syst (REAL t, REAL *y, REAL *f)
{
  	(void)(t); // avoid unused parameter warning 

	double *out = new double[4];
	rhs( out, y[0], y[1], y[2], y[3] );

	f[0] = out[0]; // \dot{R} 
	f[1] = out[1]; // \dot{p_R}
	f[2] = out[2]; // \dot{\theta}
	f[3] = out[3]; // \dot{p_\theta}

	delete [] out;
}

int main()
{
	Parameters parameters;
	FileReader filReader( "../parameters_equilibrium_mean.in", &parameters );
	
	Trajectory trajectory( parameters );
	trajectory.run_trajectory( syst );


	return 0;
}
