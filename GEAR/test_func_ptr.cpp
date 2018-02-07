#include "basis.hpp"
#include "vmblock.hpp"
#include "gear.hpp"

#include <iostream>
#include "../he_ar/matrix_he_ar.hpp"


class Trajectory
{
public:
	Trajectory() { } 
void run_trajectory(dglsysfnk func)
	{
	REAL epsabs;    //  absolute error bound
	REAL epsrel;    //  relative error bound    
	REAL t0;        // left edge of integration interval
	REAL *y0;       // [0..n-1]-vector: initial value, approxim. 
	REAL h;         // initial, final step size
	REAL xend;      // right edge of integration interval 

	long fmax;      // maximal number of calls of right side in gear4()
	long aufrufe;   // actual number of function calls
	int  N;         // number of DEs in system
	int  fehler;    // error code from umleiten(), gear4()

	void *vmblock;  // List of dynamically allocated vectors
		
	N = 4;
	vmblock = vminit();
	y0 = (REAL*) vmalloc(vmblock, VEKTOR, N, 0);

	// accuracy of trajectory
	epsabs = 1E-13;
	epsrel = 1E-13;

	fmax = 1e8;  		  // maximal number of calls 
	t0 = 0.0;

	h = 0.1;  // initial step size
	xend = 100; 
		
	fehler = gear4(&t0, xend, N, func, y0, epsabs, epsrel, &h, fmax, &aufrufe);
}


private:
	REAL x;	
	int* params;

};

void syst (REAL t, REAL *y, REAL *f)
{
  	(void)(t); // avoid unused parameter warning 

	double *out = new double[4];

	rhs( out, y[0], y[1], y[2], y[3] );

	//cout << "out[0]: " << out[0] << endl;
	//cout << "out[1]: " << out[1] << endl;
	//cout << "out[2]: " << out[2] << endl;
	//cout << "out[3]: " << out[3] << endl;

	f[0] = out[0]; // \dot{R} 
	f[1] = out[1]; // \dot{p_R}
	f[2] = out[2]; // \dot{\theta}
	f[3] = out[3]; // \dot{p_\theta}

	delete [] out;
}


int main()
{
	Trajectory trajectory;

	trajectory.run_trajectory( syst );

	return 0;
}
