#include <mpi.h>

#include <iostream>
#include <iomanip>
#include <random>
#include <ctime>
#include <functional>
#include <algorithm>

#include <Eigen/Dense>

// should be included BEFORE Gear header files
// due to namespace overlap
#include <vector>

// matrix multiplication
#include "matrix_he_ar.hpp"
// FileReader class
#include "file.hpp"
// Parameters class
#include "parameters.hpp"
// SpectrumInfo class
#include "spectrum_info.hpp"
// MCMC_generator class
#include "mcmc_generator.hpp"
// Integrand class
#include "integrand.hpp"
// Integrator class
#include "integrator.hpp"
// miscellaneous functions
#include "fft.hpp"
// physical constants
#include "constants.hpp"
// library for FFT
#include <fftw3.h>

// Gear header files
#include "basis.hpp"
#include "vmblock.hpp"
#include "gear.hpp"

using namespace std;
using namespace std::placeholders;
using Eigen::VectorXd;

// ############################################
// Exit tag for killing slave
const int EXIT_TAG = 42;
const int TRAJECTORY_CUT_TAG = 43;
const int TRAJECTORY_FINISHED_TAG = 44;
// ############################################

// ############################################
const double DALTON_UNIT = 1.660539040 * 1e-27;

const double HE_MASS = 4.00260325413;
const double AR_MASS = 39.9623831237; 
const double MU_SI = HE_MASS * AR_MASS / ( HE_MASS + AR_MASS ) * DALTON_UNIT; 
const double MU = MU_SI / constants::AMU;

static const double MYPI = 3.141592653589793; 
const double TWO_PI = 2 * MYPI; 
// ############################################

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

// x = [ R, pR, pT ]
double numerator_integrand_( VectorXd x, const double& temperature )
{
	double R = x( 0 );
	double pR = x( 1 );
	double pT = x( 2 );

	double h = pow(pR, 2) / (2 * MU) + pow(pT, 2) / (2 * MU * R * R);
	return exp( -h * constants::HTOJ / (constants::BOLTZCONST * temperature )); 
}

// x = [ R, pR, pT ]
double denumerator_integrand_( VectorXd x, const double& temperature )
{
	double R = x( 0 );
	double pR = x( 1 );
	double pT = x( 2 );	
	
	double h = pow(pR, 2) / (2 * MU) + pow(pT, 2) / (2 * MU * R * R) + ar_he_pot( R );
	
	if ( h < 0 )
	{
		// cout << "R: " << R << "; h: " << h << endl;
	}
	//cout << "R: " << R << "; U: " << ar_he_pot( R ) << endl;
	return exp( -h * constants::HTOJ / (constants::BOLTZCONST * temperature )); 
}

vector<double> create_frequencies_vector( Parameters& parameters )
{
	double FREQ_STEP = 1.0 / (parameters.sampling_time * constants::ATU) / constants::LIGHTSPEED_CM / parameters.MaxTrajectoryLength; // cm^-1
	//cout << "FREQ_STEP: " << FREQ_STEP << endl;
	int FREQ_SIZE = (int) parameters.FREQ_MAX / FREQ_STEP + 1;

	vector<double> freqs( FREQ_SIZE );
	for(int k = 0; k <  FREQ_SIZE; k++) 
	{
		freqs[k] = k * FREQ_STEP;
	}

	return freqs;
}

void master_code( int world_size )
{
	MPI_Status status;
	int source;

	Parameters parameters;
	FileReader fileReader( "parameters_equilibrium_mean.in", &parameters ); 
	//parameters.show_parameters();

	int sent = 0;	
	int received = 0;

	// status of calculation
	bool is_finished = false;

	vector<double> freqs = create_frequencies_vector( parameters );
	int FREQ_SIZE = freqs.size();

	// creating objects to hold spectrum info
	SpectrumInfo classical( FREQ_SIZE, "classical" );

	function<double(VectorXd)> numerator_integrand = bind( numerator_integrand_, _1, parameters.Temperature );

	// creating MCMC_generator object
	MCMC_generator generator( numerator_integrand, parameters );
	
	// initializing initial point to start burnin from
	VectorXd initial_point = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>( parameters.initial_point.data(), parameters.initial_point.size());	
	generator.burnin( initial_point, 10000 );	
	
	generator.set_point_limits()->add_limit(0, 0.0, 40.0)
								->add_limit(1, -50.0, 50.0)
								->add_limit(2, -250.0, 250.0);
	
	// allocating histograms to store variables
	generator.set_histogram_limits()->add_limit(0, 0.0, parameters.RDIST)
									->add_limit(1, -50.0, 50.0)
								    ->add_limit(2, -250.0, 250.0); 
	vector<string> names { "R", "pR", "pT" };
	generator.allocate_histograms( names );	

	VectorXd p;
	// sending first trajectory	
	for ( int i = 1; i < world_size; i++ )
	{
		p = generator.generate_free_state_point( );
		//generator.show_current_point();

		MPI_Send( p.data(), parameters.DIM, MPI_DOUBLE, i, 0, MPI_COMM_WORLD );
		MPI_Send( &sent, 1, MPI_INT, i, 0, MPI_COMM_WORLD );

		sent++;
	}

	clock_t start = clock();

	while( true )
	{
		if ( is_finished )	
		{
			for ( int i = 1; i < world_size; i++ )
			{	
				MPI_Send( &is_finished, 1, MPI_INT, i, EXIT_TAG, MPI_COMM_WORLD );
			}

			break;
		}

		int msg;
		MPI_Recv( &msg, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status ); 
		if ( status.MPI_TAG == TRAJECTORY_CUT_TAG )
		{
			cout << "(0) Received cutting trajectory tag!" << endl;
			if ( sent < parameters.NPOINTS )
			{
				source = status.MPI_SOURCE;
				p = generator.generate_free_state_point( );
				MPI_Send( p.data(), parameters.DIM, MPI_DOUBLE, source, 0, MPI_COMM_WORLD );
				MPI_Send( &sent, 1, MPI_INT, source, 0, MPI_COMM_WORLD );
				sent++;

				continue;
			}
		}

		// ############################################################
		// Receiving data
		classical.receive( source, true );
		received++;
		// ############################################################

		classical.add_package_to_total();

		string name = "temp";
		stringstream ss;
		if ( received % 500 == 0 )
		{
			double multiplier = 1.0 / parameters.NPOINTS;
			classical.multiply_total( multiplier );
			
			ss << received;
			classical.saving_procedure( parameters, freqs, name + ss.str() + ".txt", "total" );
			classical.zero_out_total();
		}

		if ( received == parameters.NPOINTS )
		{
			double multiplier = 1.0 / parameters.NPOINTS; 
			classical.multiply_total( multiplier );

			cout << ">> Saving spectrum" << endl << endl;
			classical.saving_procedure( parameters, freqs ); 

			is_finished = true;
		}		

		if ( sent < parameters.NPOINTS )
		{
			p = generator.generate_free_state_point( ); 
			generator.show_current_point( );	

			MPI_Send( p.data(), parameters.DIM, MPI_DOUBLE, source, 0, MPI_COMM_WORLD );
			MPI_Send( &sent, 1, MPI_INT, source, 0, MPI_COMM_WORLD );

			sent++;
		}
	}
}

void merge_vectors( vector<double>& merged, vector<double>& forward, vector<double>& backward )
{
	reverse( forward.begin(), forward.end() );
	for ( size_t k = 0; k < forward.size(); k++ )
	{
		merged.push_back( forward[k] );
	}

	for ( size_t k = 0; k < backward.size(); k++ )
	{
		merged.push_back( backward[k] );
	}
}

void slave_code( int world_rank )
{
	// it would be easier for slave to read parameters file by himself, 
	// rather than sending him parameters object (or just several parameters)
	Parameters parameters;
	FileReader fileReader( "parameters_equilibrium_mean.in", &parameters ); 

	MPI_Status status;

	// #####################################################
	// initializing special fourier class
	Fourier fourier( parameters.MaxTrajectoryLength );

	vector<double> freqs = create_frequencies_vector( parameters );
	int FREQ_SIZE = freqs.size();
	double FREQ_STEP = freqs[1] - freqs[0];

	double specfunc_coeff = 1.0/(4.0*M_PI)/constants::EPSILON0 * pow(parameters.sampling_time * constants::ATU, 2)/2.0/M_PI * pow(constants::ADIPMOMU, 2);

	double spectrum_coeff = 8.0*M_PI*M_PI*M_PI/3.0/constants::PLANCKCONST/constants::LIGHTSPEED * specfunc_coeff * pow(constants::LOSHMIDT_CONSTANT, 2);

	// j -> erg; m -> cm
	double SPECFUNC_POWERS_OF_TEN = 1e19;
	// m^-1 -> cm^-1
	double SPECTRUM_POWERS_OF_TEN = 1e-2;

	double kT = constants::BOLTZCONST * parameters.Temperature;
	// #####################################################
	
	//// #####################################################
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

	fmax = 1e9;  		  // maximal number of calls 
	// #####################################################

	// creating objects to hold spectal info
	SpectrumInfo classical;

	vector<double> p( parameters.DIM );
	int traj_counter = 0;
	bool cut_trajectory = false;

	while ( true )
	{
		MPI_Recv( &p[0], parameters.DIM, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status ); 
		
		if ( status.MPI_TAG == EXIT_TAG )
		{
			cout << "Received exit tag." << endl;
			break;
		}
		MPI_Recv( &traj_counter, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status );

		y0[0] = p[0]; 
		y0[1] = p[1];
		y0[2] = 0.0; 
		y0[3] = p[2]; 

		int counter = 0;
		double R_end_value = parameters.RDIST; 

		// dipole moment in laboratory frame
		vector<double> temp( 3 );
		vector<double> dipx_forward;
		vector<double> dipy_forward;
		vector<double> dipz_forward;

		// #####################################################
		t0 = 0.0;

		h = 0.1;  // initial step size
		xend = parameters.sampling_time; // initial right bound of integration
		// #####################################################

		clock_t start = clock();

		cut_trajectory = false;
		// #####################################################
		while ( y0[0] < R_end_value ) 
		{
			if ( counter == parameters.MaxTrajectoryLength )
			{
				cout << "Forward trajectory cut!" << endl;
				cut_trajectory = true;
				break;
			}

			fehler = gear4(&t0, xend, N, syst, y0, epsabs, epsrel, &h, fmax, &aufrufe);

			if ( fehler != 0 ) 
			{
				cout << "Gear4: error n = " << 10 + fehler << endl;
				break;
			}

			transform_dipole( temp, y0[0], y0[2] );

			dipx_forward.push_back( temp[0] );
			dipy_forward.push_back( temp[1] );
			dipz_forward.push_back( temp[2] );

			xend = parameters.sampling_time * (counter + 2);

			aufrufe = 0;  // actual number of calls

			counter++;
		}

		//cout << "Finished forward trajectory part!" << endl;
		// #####################################################
		t0 = 0.0;

		h = 0.1;  // initial step size
		xend = parameters.sampling_time; // initial right bound of integration
		
		y0[0] = p[0]; 
		y0[1] = - p[1];
		y0[2] = 0.0; 
		y0[3] = - p[2]; 
		
		vector<double> dipx_backward;
		vector<double> dipy_backward;
		vector<double> dipz_backward;
		
		while ( y0[0] < R_end_value ) 
		{
			if ( counter == parameters.MaxTrajectoryLength ) 
			{
				cout << "Backward trajectory cut!" << endl;
				cut_trajectory = true;
				break;
			}

			fehler = gear4(&t0, xend, N, syst, y0, epsabs, epsrel, &h, fmax, &aufrufe);

			if ( fehler != 0 ) 
			{
				cout << "Gear4: error n = " << 10 + fehler << endl;
				break;
			}

			transform_dipole( temp, y0[0], y0[2] );

			dipx_backward.push_back( temp[0] );
			dipy_backward.push_back( temp[1] );
			dipz_backward.push_back( temp[2] );

			xend = parameters.sampling_time * (counter + 2);

			aufrufe = 0;  // actual number of calls

			counter++;
		}
		//cout << "Finished backward trajectory part!" << endl;
		
		int msg = 0;
		if ( cut_trajectory )
		{
			cout << "Sending cutting trajectory signal!" << endl;
			MPI_Send( &msg, 1, MPI_INT, 0, TRAJECTORY_CUT_TAG, MPI_COMM_WORLD );
			continue;
		}
		else
		{
			MPI_Send( &msg, 1, MPI_INT, 0, TRAJECTORY_FINISHED_TAG, MPI_COMM_WORLD );
		}

		vector<double> dipx;
		vector<double> dipy;
		vector<double> dipz;
		
		merge_vectors( dipx, dipx_forward, dipx_backward );
		merge_vectors( dipy, dipy_forward, dipy_backward );
		merge_vectors( dipz, dipz_forward, dipz_backward );
		
		if ( dipz.size() == 0 )
		{
			cout << "Starting point: " << p[0] << " " << p[1] << " " << p[2] << endl;
		}

		// #####################################################
		// length of dipole vector = number of samples
		int npoints = dipz.size();

		// zeroing input arrays
		fourier.zero_out_input( );

		fourier.copy_into_fourier_array( dipx, "x" );
		fourier.copy_into_fourier_array( dipy, "y" );
		fourier.copy_into_fourier_array( dipz, "z" );

		// executing fourier transform
		fourier.do_fourier( );

		double omega, dipfft;

		double ReFx, ReFy, ReFz;
		double ImFx, ImFy, ImFz;

		double specfunc_value_classical;
		double spectrum_value_classical;

		for ( int k = 0; k < FREQ_SIZE; k++ )
		{
			// Hz 
			omega = 2.0 * M_PI * constants::LIGHTSPEED_CM * freqs[k];

			ReFx = fourier.outx[k][0];
			ReFy = fourier.outy[k][0];
			ReFz = fourier.outz[k][0];
			ImFx = fourier.outx[k][1];
			ImFy = fourier.outy[k][1];
			ImFz = fourier.outz[k][1];

			dipfft = ReFx * ReFx + ReFy * ReFy + ReFz * ReFz +
 					 ImFx * ImFx + ImFy * ImFy + ImFz * ImFz;
			//cout << "dipfft[" << k << "] = " << dipfft * constants::ADIPMOMU * constants::ADIPMOMU << endl; 

			specfunc_value_classical = SPECFUNC_POWERS_OF_TEN * specfunc_coeff * dipfft;
			classical.specfunc_package.push_back( specfunc_value_classical );

			spectrum_value_classical = SPECTRUM_POWERS_OF_TEN * spectrum_coeff * omega *  ( 1.0 - exp( - constants::PLANCKCONST_REDUCED * omega / kT ) ) * dipfft;
			
			classical.spectrum_package.push_back( spectrum_value_classical );

			classical.m2_package += spectrum_value_classical * FREQ_STEP; 
		}

		cout << "(" << world_rank << ") Processing " << traj_counter << " trajectory. npoints = " << npoints << "; time = " << (clock() - start) / (double) CLOCKS_PER_SEC << "s" << endl;

		//// #################################################
		// Sending data
		classical.send();
		classical.clear_package();
		// #################################################
	}
}

int main( int argc, char* argv[] )
{
	//Initialize the MPI environment
	MPI_Init( &argc, &argv );

	//getting id of the current process
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	//getting number of running processes
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size); 

	if ( world_rank == 0 ) 
	{
		clock_t start = clock();

		master_code( world_size );

		cout << "Time elapsed: " << (clock() - start) / (double) CLOCKS_PER_SEC << "s" << endl;
	}	
	else
	{
		slave_code( world_rank );
	}

	MPI_Finalize();

	return 0;
}
