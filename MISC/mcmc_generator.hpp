#pragma once

#include <iostream>
#include <random>
#include <functional>
#include <vector>
#include <chrono>
#include <fstream>

#include "parameters.hpp"

#include <Eigen/Dense>
#include <gsl/gsl_histogram.h>

#include "limits.hpp"

using std::cout;
using std::endl;
using std::function;
using std::vector;
using std::pair;
using std::string;

using Eigen::VectorXd;

static vector<pair<int,double>> DEFAULT_VECTOR;

class MCMC_generator 
{
public:
	vector<pair<int, double>> to_wrap;
	Parameters parameters;
	
	Limits plimits;
	bool set_plimits = false;
	Limits* set_point_limits( void ) 
	{ 
		set_plimits = true;
		return &plimits; 
	}	

	vector<gsl_histogram*> histograms;
	vector<string> names;
	const int NBINS = 100;
	void gsl_histogram_normalize( gsl_histogram* h );
	void save_histogram( gsl_histogram* h, const string& filename );

	bool set_histograms = false;
	Limits limits;
	Limits* set_histogram_limits( void ) 
	{ 
		set_histograms = true;
		return &limits; 
	}  
	void allocate_histograms( vector<string>& names );

	bool burnin_done = false;

	VectorXd current_point{ parameters.DIM };

	function<double(VectorXd)> f;

	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::mt19937 generator{ seed }; 
	
	void set_initial_point( std::vector<double> ip );
	void show_current_point( void );

	double nextDouble( const double& min, const double& max );
	void nextGaussianVec( VectorXd &v, VectorXd &mean );
	double wrapMax( const double& x, const double& max );

	void burnin( VectorXd initial_point, const int& burnin_length );
	VectorXd metro_step( VectorXd& x );

	int pR_index;
	int Jx_index;

	VectorXd generate_point( ); 

	MCMC_generator( 
		function<double(VectorXd)> f,
	   	Parameters& parameters,	
		vector<pair<int, double>>& to_wrap = DEFAULT_VECTOR 
			);
	~MCMC_generator();
};
