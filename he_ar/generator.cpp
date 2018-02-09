#include <iostream>
#include <functional>
#include <vector>
#include <string>

#include <Eigen/Dense>
#include "../MISC/file.hpp"
#include "../MISC/mcmc_generator.hpp"
#include "../MISC/constants.hpp"

using namespace std;
using namespace std::placeholders;

using Eigen::VectorXd;

// ############################################
const double HE_MASS = 4.00260325413;
const double AR_MASS = 39.9623831237; 
const double MU_SI = HE_MASS * AR_MASS / ( HE_MASS + AR_MASS ) * constants::DALTON_UNIT; 
const double MU = MU_SI / constants::AMU;
// ############################################

// x = [R, pR, pT]
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

	function<double(VectorXd)> numerator_integrand = bind( numerator_integrand_, _1, parameters.Temperature );

	MCMC_generator generator( numerator_integrand, parameters );	

	VectorXd initial_point = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>( parameters.initial_point.data(), parameters.initial_point.size() );
	generator.burnin( initial_point, 10000 );

	generator.set_point_limits()->add_limit(0, 3.0, parameters.RDIST)
								->add_limit(1, -100.0, 100.0)
								->add_limit(2, 0.0, 2 * M_PI )
								->add_limit(3, -300.0, 300.0);

	generator.set_histogram_limits()->add_limit(0, 3.0, parameters.RDIST)
									->add_limit(1, -100.0, 100.0)
									->add_limit(2, 0.0, 2 * M_PI)
									->add_limit(3, -300.0, 300.0);

	vector<string> names{ "R", "pR", "theta", "pTheta" };
	generator.allocate_histograms( names );

	int chain_length = 10000;
	for ( size_t k = 0; k != chain_length; k++ )
		generator.generate_free_state_point( );

	return 0;
}
