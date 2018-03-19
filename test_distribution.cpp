#include <iostream>
#include "ar_he_pes.hpp"
#include "ar_he_dip_buryak_fit.hpp"
#include "constants.hpp"

#include "mcmc_generator.hpp"

#include <Eigen/Dense>

#include "parameters.hpp"
#include "file.hpp"

#include <vector>
#include <cstddef>
#include "hep/mc.hpp"

using namespace std;
using namespace std::placeholders;
using Eigen::VectorXd;

// ############################################
const double HE_MASS = 4.00260325413;
const double AR_MASS = 39.9623831237; 
const double MU_SI = HE_MASS * AR_MASS / ( HE_MASS + AR_MASS ) * constants::DALTON_UNIT; 
const double MU = MU_SI / constants::AMU;
// ############################################

const int DIM = 4;

// значения R в интеграле в числителе
const double R0_num = 3.0;
const double Rmax_num = 40.0;
// значения R в интеграле в знаменателе
const double R0_den = 3.0;
const double Rmax_den = 40.0;

double integrand( hep::mc_point<double> const & x, double Temperature, bool is_numerator )
{
	double pR_new = x.point()[1];
	//double Theta_new = x.point()[2];
	double pTheta_new = x.point()[3];

	//double Theta = M_PI * Theta_new;
	double pR = tan(M_PI * (pR_new - 0.5));
	double pTheta = tan(M_PI * (pTheta_new - 0.5));
	
	double jacpR = M_PI * (1 + pow(pR, 2));
	double jacTheta = M_PI;
	double jacpTheta = M_PI * (1 + pow(pTheta, 2));
	
	double jacTotal = jacpR * jacTheta * jacpTheta;

	// рассчитываем знаменатель
	if ( !is_numerator )
	{
		double R = x.point()[0] * (Rmax_den - R0_den) + R0_den;
		double jacR = Rmax_den - R0_den;
		jacTotal *= jacR;
		
		double h = pow(pR, 2) / (2 * MU) + pow(pTheta, 2) / (2 * MU * R * R) + ar_he_pot( R );

		if ( h > 0 )
			return exp( -h * constants::HTOJ / constants::BOLTZCONST / Temperature ) * jacTotal;
		else 
			return 0.0;
	}

	// рассчитываем числитель
	if ( is_numerator ) 
	{
		double R  = x.point()[0] * (Rmax_num - R0_num) + R0_num;
		double jacR = Rmax_num - R0_num;
		jacTotal *= jacR;
		
		double h = pow(pR, 2) / (2 * MU) + pow(pTheta, 2) / (2 * MU * R * R) + ar_he_pot( R );

		if ( h > 0 )
			return ar_he_dip_buryak_fit(R) * ar_he_dip_buryak_fit(R) * exp( -h * constants::HTOJ / constants::BOLTZCONST / Temperature) * jacTotal;
		else
			return 0.0;
	}
}

double integrate( double Temperature )
{
	int niter = 10;
	int npoints = 1e5;

	auto dipole_integrand = bind( integrand, _1, Temperature, true );

	hep::vegas_callback<double>(hep::vegas_verbose_callback<double>);

	auto results = hep::vegas(
		hep::make_integrand<double>(dipole_integrand, DIM), 
		vector<size_t>(niter, npoints)
			);

	auto result = hep::cumulative_result0( results.begin() + 1, results.end());
	double dipole_integral = result.value();

	auto exponent_integrand = bind( integrand, _1, Temperature, false );

	results = hep::vegas(
			hep::make_integrand<double>(exponent_integrand, DIM),
			vector<size_t>(niter, npoints)
		);

	result = hep::cumulative_result0( results.begin() + 1, results.end() );

	double exponent_integral = result.value();	

	return dipole_integral / exponent_integral; 
}

double hamiltonian( VectorXd x )
{
	double R = x(0);
	double pR = x(1);
	//double Theta = x(2);
	double pTheta = x(3);
	
	return pow(pR, 2) / (2 * MU) + pow(pTheta, 2) / (2 * MU * R * R) + ar_he_pot( R );
}

double mcmc_distribution_function_( VectorXd x, double Temperature )
{
	double h = hamiltonian( x );
	return exp( - h * constants::HTOJ / constants::BOLTZCONST / Temperature );
}

double MCMC( int npoints, double Temperature )
{
	Parameters parameters;
	FileReader fileReader( "parameters_physical_correlation.in", &parameters );
	
	function<double(VectorXd)> mcmc_distribution_function = bind( mcmc_distribution_function_, _1, Temperature );

	MCMC_generator generator( mcmc_distribution_function, parameters );

	// initializing initial point to start burnin from
	VectorXd initial_point = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>( parameters.initial_point.data(), parameters.initial_point.size());	
	generator.burnin( initial_point, 5 );	

	// [ R, pR, theta, pT ]
	generator.set_point_limits()->add_limit(0, 3.0, parameters.RDIST)
								->add_limit(1, -100.0, 100.0)
								->add_limit(2, 0.0, 2 * M_PI)
								->add_limit(3, -1000.0, 1000.0);
	
	double summ = 0;
	for ( int i = 0; i < npoints; i++ )
	{
		if ( i % 100000 == 0 )
			cout << "(MCMC) i = " << i << endl;

		VectorXd p = generator.generate_free_state_point( hamiltonian ); 
		summ = summ + ar_he_dip_buryak_fit(p(0)) * ar_he_dip_buryak_fit(p(0));
	}

	return summ / npoints; 
}

int main()
{
	const double Temperature = 295.0;

	double integral_value = integrate( Temperature ); 
	cout << "integral value: " << integral_value << endl;

	double distribution_summ = MCMC( 5e6, Temperature );
	cout << "Distribution value: " << distribution_summ << endl;
		
	return 0;
}

