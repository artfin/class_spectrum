#include <iostream>
#include <Eigen/Dense>

#include "integrand.hpp"
#include "integrator.hpp"
#include <functional>
#include "constants.hpp"
#include "../POT/ar_he_pes.hpp"

using namespace std;
using namespace std::placeholders;

// x = [ R, pR, pT ]
double denumerator_integrand_( VectorXd x, const double& temperature )
{
	double R = x( 0 );
	double pR = x( 1 );
	double pT = x( 2 );	
	
	double h = pow(pR, 2) / 2 + pow(pT, 2) / (2 * R * R) + ar_he_pot( R );
	cout << "R: " << R << "; U: " << ar_he_pot( R ) << endl;
	return exp( -h * constants::HTOJ / (constants::BOLTZCONST * temperature )); 
}

int main( )
{
	MPI_Init( NULL, NULL ); 

	function<double(VectorXd)> denumerator_integrand = bind( denumerator_integrand_, _1, 200.0 );  
	Integrand denum_integrand( denumerator_integrand, 3 );
	denum_integrand.set_limits()->add_limit( 0, 4.0, "+inf")
						  		->add_limit( 1, "-inf", "+inf" )
						  		->add_limit( 2, "-inf", "+inf" );
	
	Integrator denumerator_integrator( denum_integrand );
	double denumerator_value = denumerator_integrator.run_integration( 10, 1000 );

	MPI_Finalize();

	return 0;	
}
