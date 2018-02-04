#include <iostream>
#include <functional>

#include "../MISC/parameters.hpp"
#include "../MISC/mcmc_generator.hpp"
#include "../MISC/file.hpp"
#include "../MISC/constants.hpp"

#include <Eigen/Dense>

using namespace std;
using namespace Eigen;
using namespace std::placeholders;

const double MU = 6632.04; 

// x = [ R, pR, pT ]
double numerator_integrand_( VectorXd x, const double& temperature )
{
	double R = x( 0 );
	double pR = x( 1 );
	double pT = x( 2 );

	double h = pow(pR, 2) / (2 * MU) + pow(pT, 2) / (2 * MU * R * R);
	return exp( -h * constants::HTOJ / (constants::BOLTZCONST * temperature )); 
}	

int main()
{
	Parameters parameters;
	FileReader fileReader( "../parameters_equilibrium_mean.in", &parameters );
	parameters.output_directory = "./";

	function<double(VectorXd)> numerator_integrand = bind( numerator_integrand_, _1, parameters.Temperature );

	MCMC_generator generator( numerator_integrand, parameters ); 
	
	// initializing initial point to start burnin from
	VectorXd initial_point = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>( parameters.initial_point.data(), parameters.initial_point.size());	
	generator.burnin( initial_point, 10000 );	
	
	// allocating histograms to store variables
	/*
	generator.set_histogram_limits()->add_limit(0, 0.0, 2 * parameters.RDIST) 
									->add_limit(1, -50.0, 50.0)
								    ->add_limit(2, -250.0, 250.0); 
	vector<string> names { "R", "pR", "pT" };
	generator.allocate_histograms( names );	
	*/
	
	const int chain_length = 10000000;

	VectorXd p;
	for ( size_t k = 0; k < chain_length; k++ )
	{
		if ( k % 100000 == 0 )
		{
			cout << "k = " << k << endl;
		}

		p = generator.generate_free_state_point( );
	}

	return 0;
}
