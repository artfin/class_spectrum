#include <iostream>
#include <random>
#include <vector>
#include <cassert>
#include <fstream>
#include <cmath>

#include <gsl/gsl_histogram.h>

using namespace std;

const int NBINS = 100;
mt19937 generator{time(0)};
const int DIM = 1;

// функция, задающая распределение
double f( vector<double> const & x )
{
	assert( x.size() == DIM );
	return exp( - x[0] * x[0]); 
}

// функция, которую мы интегрируем с распределением f
double g ( vector<double> const & x )
{
	assert( x.size() == DIM );
	return x[0] * x[0]; 
}	

double nextDouble( const double min, const double max )
{
	std::uniform_real_distribution<double> distribution( min, max );
	return distribution( generator );
}

void gsl_histogram_normalize( gsl_histogram* h )
{
	double max = gsl_histogram_max( h );
	double min = gsl_histogram_min( h );
	double step = (max - min) / NBINS;

	double sum = gsl_histogram_sum( h ) * step;
	gsl_histogram_scale( h, 1.0 / sum );
}

void save_histogram( gsl_histogram * h, string const & filename )
{
	ofstream file( filename );
	
	double lower_bound, higher_bound, bin_content;
	for ( size_t counter = 0; counter < NBINS; counter++ )
	{
		gsl_histogram_get_range( h, counter, &lower_bound, &higher_bound );
		bin_content = gsl_histogram_get( h, counter );
		file << lower_bound << " " << higher_bound << " " << bin_content << endl;
	}

	file.close();
}

vector<double> gibbs_step( vector<double> const & x, vector<double> const & radii )
{
	assert ( x.size() == radii.size() );

	vector<double> xnew(x.size());
	for ( size_t k = 0; k < x.size(); k++ )
	{
		normal_distribution<double> d( x[k], radii[k] );
	   	xnew[k] = d( generator );	
	}

	if ( nextDouble(0.0, 1.0) < min( 1.0, f(xnew) / f(x) ))
		return xnew;
	else
		return x;
}

int main()
{
	cout << "Dimension of function is " << DIM << endl;
	cout << "Enter the radii vector: " << endl;
	vector<double> radii(DIM);
	for ( int k = 0; k < DIM; k++ )
		cin >> radii[k];

	cout << "Enter a number of points to generate: " << endl;
	int npoints;
	cin >> npoints;

	cout << "Enter intial guess: " << endl;
	vector<double> initial_guess(DIM);
	for ( int k = 0; k < DIM; k++ )
		cin >> initial_guess[k];
	
	int NRUNS;
	cout << "Enter number of runs: " << endl;
	cin >> NRUNS;

	const int burnin = 1000;
	cout << "Running burn-in -- " << burnin << " points" << endl;
	
	vector<double> x = gibbs_step( initial_guess, radii );
	for ( int k = 0; k < burnin; k++ )
		x = gibbs_step( x, radii );

	//gsl_histogram * h = gsl_histogram_alloc( NBINS ); 
	//gsl_histogram_set_ranges_uniform( h, -3.0, 3.0 );

	// по мере поступления точек будем считать в них значение
	// функции g(x) и накапливать ее значения в переменной integral_value

	for ( int run_counter = 0; run_counter < NRUNS; run_counter++ )
	{
		double integral_value = 0.0;

		//ofstream file("sample.dat");
		for ( int k = 0; k < npoints; k++ )
		{
			x = gibbs_step( x, radii );
			integral_value += g( x );

			//gsl_histogram_increment( h, x[0] );

			//for ( size_t i = 0; i < x.size(); i++ )
				//file << x[i] << " ";
			//file << endl;
		}

		// делим накопленное значение суммы на количество точек
		integral_value = integral_value / npoints;
		integral_value = integral_value * sqrt(M_PI);

		//gsl_histogram_normalize( h );
		//save_histogram( h, "function_histogram.dat" );
		//gsl_histogram_free( h );

		//cout << "Cumulative sum = " << integral_value << endl; 
	
		double exact_value = sqrt(M_PI) / 2;
		//cout << "Exact value is " << exact_value << endl; 
		//cout << "abs difference = " << abs(integral_value - exact_value) << endl;
		//cout << "rel difference = " << (integral_value - exact_value) / exact_value * 100 << "%" << endl;
		double rel_difference = (integral_value - exact_value) / exact_value;
		cout << rel_difference << endl;
	}

	return 0;
}

