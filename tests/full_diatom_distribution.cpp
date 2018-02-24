#include <iostream>
#include <functional>

#include "../MISC/parameters.hpp"
#include "../MISC/mcmc_generator.hpp"
#include "../MISC/file.hpp"
#include "../MISC/constants.hpp"
#include "../POT/ar_he_pes.hpp"

#include <Eigen/Dense>

using namespace std;
using namespace Eigen;
using namespace std::placeholders;

const double MU = 6632.04; 

// x = [ R, pR, theta, pT ]
double hamiltonian( VectorXd x )
{
	double R = x( 0 );
	double theta = x( 1 );
	double pR = x( 2 );
	double pT = x( 3 );

	return pow(pR, 2) / (2 * MU) + pow(pT, 2) / (2 * MU * R * R) + ar_he_pot( R );
}

// x = [ R, theta, pR, pT ]
double numerator_integrand_( VectorXd x, const double& temperature )
{
	double h = hamiltonian( x );
	return exp( -h * constants::HTOJ / (constants::BOLTZCONST * temperature )); 
}

int main()
{
	Parameters parameters;
	FileReader fileReader( "../parameters_equilibrium_mean.in", &parameters );
	parameters.output_directory = "./";

	parameters.alpha = 2.0;
	
	function<double(VectorXd)> numerator_integrand = bind( numerator_integrand_, _1, parameters.Temperature );

	MCMC_generator generator( numerator_integrand, parameters ); 
	
	// initializing initial point to start burnin from
	VectorXd initial_point = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>( parameters.initial_point.data(), parameters.initial_point.size());	
	generator.burnin( initial_point, 2 );	
	
	generator.set_point_limits()->add_limit(0, 4.0, 40.0)
								->add_limit(1, 0.0, 2 * M_PI)
								->add_limit(2, -50.0, 50.0)
								->add_limit(3, -250.0, 250.0);

	// allocating histograms to store variables
	generator.set_histogram_limits()->add_limit(0, 3.0, 40.0)
		   							->add_limit(1, 0.0, 2 * M_PI)	
									->add_limit(2, -50.0, 50.0)
								    ->add_limit(3, -250.0, 250.0);

	vector<string> names { "R_pot2", "theta_pot2", "pR_pot2", "pT_pot2" };
	generator.allocate_histograms( names );	

	const int chain_length = 1e5;

	VectorXd p;

	clock_t start = clock();
	for ( size_t k = 0; k < chain_length; k++ )
	{
		if ( k % 1000 == 0 ) 
		{
			cout << "k: " << k << endl;
			cout << "Elapsed: " << (double) (clock() - start) / CLOCKS_PER_SEC << " s" << endl << endl;
		}

		p = generator.generate_free_state_point( hamiltonian );
	}

	return 0;
}
