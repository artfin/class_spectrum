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
// tags for mpi messages
#include "tags.hpp"
// physical constants
#include "constants.hpp"
// Trajectory class
#include "trajectory.hpp"
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
const double HE_MASS = 4.00260325413;
const double AR_MASS = 39.9623831237; 
const double MU_SI = HE_MASS * AR_MASS / ( HE_MASS + AR_MASS ) * constants::DALTON_UNIT; 
const double MU = MU_SI / constants::AMU;
// ############################################

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

// x = [ R, pR, theta, pT ]
double hamiltonian( VectorXd x )
{
	double R = x( 0 );
	double theta = x( 1 );
	double pR = x( 2 );
	double pT = x( 3 );

	return pow(pR, 2) / (2 * MU) + pow(pT, 2) / (2 * MU * R * R);
}

// x = [ R, theta, pR, pT ]
double numerator_integrand_( VectorXd x, const double& temperature )
{
	double h = hamiltonian( x );
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
	generator.burnin( initial_point, 2 );	

	generator.set_point_limits()->add_limit(0, 4.0, 40.0)
								->add_limit(1, 0.0, 2 * M_PI)
								->add_limit(2, -50.0, 50.0)
								->add_limit(3, -250.0, 250.0);

	// allocating histograms to store variables
	generator.set_histogram_limits()->add_limit(0, 0.0, 40.0)
		   							->add_limit(1, 0.0, 2 * M_PI)	
									->add_limit(2, -50.0, 50.0)
								    ->add_limit(3, -250.0, 250.0);

	vector<string> names { "R", "theta", "pR", "pT" };
	generator.allocate_histograms( names );	

	VectorXd p;
	// sending first trajectory	
	for ( int i = 1; i < world_size; i++ )
	{
		p = generator.generate_free_state_point( hamiltonian );
		generator.show_current_point();

		MPI_Send( p.data(), parameters.DIM, MPI_DOUBLE, i, 0, MPI_COMM_WORLD );
		//cout << "(master) sent point " << endl;

		MPI_Send( &sent, 1, MPI_INT, i, 0, MPI_COMM_WORLD );
		//cout << "(master) sent number of trajectory" << endl;

		sent++;
	}

	clock_t start = clock();

	while( true )
	{
		if ( is_finished )	
		{
			for ( int i = 1; i < world_size; i++ )
				MPI_Send( &is_finished, 1, MPI_INT, i, tags::EXIT_TAG, MPI_COMM_WORLD );

			break;
		}

		int msg;
		MPI_Recv( &msg, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status );
	   	source = status.MPI_SOURCE;	

		if ( status.MPI_TAG == tags::TRAJECTORY_CUT_TAG )
		{
			cout << "(master) Received cutting trajectory tag!" << endl;
			if ( sent < parameters.NPOINTS )
			{
				p = generator.generate_free_state_point( hamiltonian );
				generator.show_current_point();

				cout << "(master) Sending new point!" << endl;
				MPI_Send( p.data(), parameters.DIM, MPI_DOUBLE, source, 0, MPI_COMM_WORLD );
				MPI_Send( &sent, 1, MPI_INT, source, 0, MPI_COMM_WORLD );
				sent++;
				cout << "(master) Sent new point" << endl;

				continue;
			}
		}
	
		//if ( status.MPI_TAG == tags::TRAJECTORY_FINISHED_TAG )
		//	cout << "(master) Trajectory is not cut!" << endl;

		// ############################################################
		// Receiving data
		classical.receive( source, false );
		received++;
		// ############################################################

		classical.add_package_to_total();

		string name = "temp";
		stringstream ss;

		int block_trajectory_size = 1; 
		if ( received % block_trajectory_size == 0 )
		{
			double multiplier = 1.0 / block_trajectory_size; 
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
			p = generator.generate_free_state_point( hamiltonian ); 
			generator.show_current_point( );	

			MPI_Send( p.data(), parameters.DIM, MPI_DOUBLE, source, 0, MPI_COMM_WORLD );
			MPI_Send( &sent, 1, MPI_INT, source, 0, MPI_COMM_WORLD );

			sent++;
		}
	}
}

void merge_vectors( vector<double> & merged, vector<double> & forward, vector<double> & backward )
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
	
	// creating objects to hold spectal info
	SpectrumInfo classical;

	int cut_trajectory = 0;
	bool exit_status = false;
	Trajectory trajectory( parameters );

	stringstream ss;

	vector<double> dipx, dipx_forward, dipx_backward;
	vector<double> dipy, dipy_forward, dipy_backward;
	vector<double> dipz, dipz_forward, dipz_backward;

	while ( true )
	{
		cut_trajectory = 0;
		trajectory.set_cut_trajectory( 0 );	

		clock_t start = clock();

		exit_status = trajectory.receive_initial_conditions( );
		if ( exit_status ) 
			break;
		
		trajectory.run_trajectory( syst );

		//ss << trajectory.get_trajectory_counter();
		//trajectory.save_trajectory( "test_trajectories2/forw_trajectory_" + ss.str() + ".txt" );

		dipx_forward = trajectory.get_dipx();
		dipy_forward = trajectory.get_dipy();
		dipz_forward = trajectory.get_dipz();
		trajectory.dump_dipoles( );

		trajectory.reverse_initial_conditions( );

		trajectory.run_trajectory( syst );
	
		//trajectory.save_trajectory( "test_trajectories2/back_trajectory_" + ss.str() + ".txt" );
		//ss.str("");

		dipx_backward = trajectory.get_dipx();
		dipy_backward = trajectory.get_dipy();
		dipz_backward = trajectory.get_dipz();
		trajectory.dump_dipoles();

		cut_trajectory = trajectory.report_trajectory_status( );
			
		if ( cut_trajectory == 1 )
			continue;
		
		merge_vectors( dipx, dipx_forward, dipx_backward );
		merge_vectors( dipy, dipy_forward, dipy_backward );
		merge_vectors( dipz, dipz_forward, dipz_backward );

		// #####################################################
		// length of dipole vector = number of samples
		int npoints = dipz.size();

		// zeroing input arrays
		fourier.zero_out_input( );

		fourier.copy_into_fourier_array( dipx, "x" );
		fourier.copy_into_fourier_array( dipy, "y" );
		fourier.copy_into_fourier_array( dipz, "z" );
		dipx.clear();
		dipy.clear();
		dipz.clear();

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

			/*			
			if ( freqs[k] > 300 && spectrum_value_classical > 1e4 )
			{
				cout << "freqs[" << k << "] = " << freqs[k] << "; spectrum_value_classical: " << spectrum_value_classical << endl;
			}
			*/
		}

		cout << "(" << world_rank << ") Processing " << trajectory.get_trajectory_counter() << " trajectory. npoints = " << npoints << "; time = " << (clock() - start) / (double) CLOCKS_PER_SEC << "s" << endl;

		/// #################################################
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

	cout << "(before) MPI_Finalize()" << endl;

	MPI_Finalize();

	return 0;
}
