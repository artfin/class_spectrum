#include "mcmc_generator.hpp"

MCMC_generator::MCMC_generator( std::function<double(VectorXd)> f,
			   					Parameters& parameters,	
								vector<pair<int, double>>& to_wrap ) : 
		f(f),
		parameters(parameters),
		to_wrap(to_wrap)
{
}

MCMC_generator::~MCMC_generator()
{
	string filename;

	if ( set_histograms )
	{
		for ( size_t i = 0; i < parameters.DIM; i++ )
		{
			gsl_histogram_normalize( histograms[i] );

			filename = parameters.output_directory + names[i] + ".txt";
			save_histogram( histograms[i], filename ); 
			
			gsl_histogram_free( histograms[i] );
		}
	}
}

void MCMC_generator::gsl_histogram_normalize( gsl_histogram* h )
{
	double max = gsl_histogram_max( h );
	double min = gsl_histogram_min( h );
	double step = (max - min) / NBINS;

	double sum = gsl_histogram_sum( h ) * step;
	gsl_histogram_scale( h, 1.0 / sum );
}

void MCMC_generator::save_histogram( gsl_histogram* h, const string& filename )
{
	std::ofstream file( filename );

	double lower_bound, higher_bound, bin_content;
	for ( size_t counter = 0; counter < NBINS; counter++ )
	{
		gsl_histogram_get_range( h, counter, &lower_bound, &higher_bound );
		bin_content = gsl_histogram_get( h, counter );
		file << lower_bound << " " << higher_bound << " " << bin_content << endl;
	}

	file.close();
}

double MCMC_generator::nextDouble( const double& min, const double& max )
{
	std::uniform_real_distribution<double> distributuion( min, max );
	return distributuion( this->generator );
}

void MCMC_generator::nextGaussianVec( VectorXd& v, VectorXd& mean )
{
	for ( size_t i = 0; i < parameters.DIM; i++ )
	{
		std::normal_distribution<double> d( mean(i), parameters.alpha );
		v(i) = d( generator );
	}
}

double MCMC_generator::wrapMax( const double& x, const double& max )
{
	return fmod( max + fmod(x, max), max );
}

void MCMC_generator::allocate_histograms( vector<string>& names )
{
	this->names = names;

	for ( size_t i = 0; i < limits.limits.size(); i++ )
	{
		gsl_histogram* h = gsl_histogram_alloc( NBINS );
		gsl_histogram_set_ranges_uniform( h, limits.limits[i].lb, limits.limits[i].ub );
		histograms.push_back( h );
	}
}

VectorXd MCMC_generator::metro_step( VectorXd& x )
{
	VectorXd prop( parameters.DIM );
	nextGaussianVec( prop, x );

	if ( nextDouble(0.0, 1.0) < std::min( 1.0, f(prop) / f(x) ))
	{
		return prop;
	}

	return x;
}

void MCMC_generator::burnin( VectorXd initial_point, const int& burnin_length )
{
	VectorXd x = metro_step( initial_point );

	for ( size_t i = 0; i < burnin_length; i++ )
	{
		x = metro_step( x );
	}

	current_point = x;
	burnin_done = true;
}

void MCMC_generator::show_current_point( void )
{
	cout << "Generated p: ";
	for ( size_t i = 0; i < parameters.DIM; i++ )
		cout << "p(" << i << ") = " << current_point(i) << "; ";
	cout << endl;
}

VectorXd MCMC_generator::generate_point( ) 
{
	if ( burnin_done == false )
	{
		std::cout << "Burnin is not done!" << std::endl;
		exit( 1 );
	}

	int moves = 0;
	VectorXd x = current_point;
	VectorXd xnew;

	double gunsight;
	bool point_found = false;

	while ( !point_found )
	{
		xnew = metro_step( x );

		if ( to_wrap != DEFAULT_VECTOR )
		{
			int curr_var; // number of current variable
			double curr_max; // maximum of current variable

			for ( size_t i = 0; i < to_wrap.size(); i++ )
			{
				curr_var = to_wrap[i].first;
				curr_max = to_wrap[i].second;

				xnew(curr_var) = wrapMax( xnew(curr_var), curr_max );
			}
		}

		if ( xnew != x )
		{
			x = xnew;
			moves++;
		}

		if ( moves > parameters.subchain_length )
		{
			if ( set_plimits )
			{
				//cout << ">> generated point: ";
			
				point_found = true;	
				for ( size_t i = 0; i < parameters.DIM; i++ )
				{
					if ( x(i) > plimits.limits[i].ub || x(i) < plimits.limits[i].lb )
					{
						point_found = false;
					}

					//cout << "x(" << i << ") = " << x(i)
						 //<< " (lb: " << plimits.limits[i].lb << "; ub: " << plimits.limits[i].ub << ");";
				}	

				//cout << endl;	
			}

		}
	}

	if ( set_histograms )
	{
		for ( size_t i = 0; i < parameters.DIM; i++ )
			gsl_histogram_increment( histograms[i], x(i) );
	}	

	current_point = x;
	return x;
}

