#pragma once

#include "hep/mc-mpi.hpp"
#include "integrand.hpp"

#include <iostream>

using std::cout;
using std::endl;

class Integrator
{
public:
	Integrand& integrand;

	Integrator( Integrand& integrand ) : integrand(integrand)
	{
	}

	double run_integration( const int& niter, const int& ndots )
	{
		hep::vegas_callback<double>(hep::vegas_verbose_callback<double>);
		
		auto results = hep::mpi_vegas(
			MPI_COMM_WORLD,
			hep::make_integrand<double>( integrand, integrand.DIM ),
			std::vector<std::size_t>( niter, ndots )
		);

		auto result = hep::cumulative_result0( results.begin() + 1, results.end() );

		return result.value();
	}
};
