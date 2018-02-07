#pragma once

#include <mpi.h>
#include <iostream>
#include <cstring>

#include "tags.hpp"

#include "matrix_he_ar.hpp"
#include "parameters.hpp"

#include "basis.hpp"
#include "vmblock.hpp"
#include "gear.hpp"

class Trajectory
{
public:
	Trajectory( Parameters & parameters )
		: parameters(parameters)
	{
		std::cout << "(trajectory) constructor" << std::endl;

		vmblock = vminit();
	
		N = parameters.DIM;

		y0 = (REAL*) vmalloc( vmblock, VEKTOR, N, 0 );
		y0_copy = (REAL*) vmalloc( vmblock, VEKTOR, N, 0 );

		if ( ! vmcomplete(vmblock) )
		{
			vmfree( vmblock ); // free memory in list
		}
	}

	~Trajectory( )
	{
		std::cout << "(trajectory) destructor" << std::endl;

		vmfree( vmblock );
	}

	bool receive_initial_conditions( void ); 

	void reverse_initial_conditions( void );
	void show_initial_conditions( void );
	void dump_dipoles( void );	

	int report_trajectory_status( void );

	void run_trajectory( dglsysfnk syst );
	
	std::vector<double> & get_dipx( void ) { return dipx; }
	std::vector<double> & get_dipy( void ) { return dipy; }
	std::vector<double> & get_dipz( void ) { return dipz; }	

	int get_trajectory_counter( void ) const { return trajectory_counter; }
private:
	Parameters parameters;

	REAL *y0 = 0; // pointer to [0..n-1]-vecotor; initial value
	REAL *y0_copy = 0; // copy of initial condition

	int N; 
	int trajectory_counter; // counter of current trajectory

	int cut_trajectory = 0;

	void * vmblock; // start of the vector/matrix list
	
	// vectors storing components of dipole moment
	std::vector<double> dipx;
	std::vector<double> dipy;
	std::vector<double> dipz;
};

