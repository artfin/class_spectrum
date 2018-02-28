#pragma once

#include <iostream>
#include <fftw3.h>
#include <algorithm>
#include <cmath>
#include <fstream>

class CorrelationFourierTransform
{
public:
	CorrelationFourierTransform( int MaxTrajectoryLength) 
		: MaxTrajectoryLength(MaxTrajectoryLength)
	{
		int outlen = MaxTrajectoryLength / 2 + 1;

		in = (double*) fftw_malloc( sizeof(double) * MaxTrajectoryLength );
		out = (fftw_complex*) fftw_malloc( sizeof(fftw_complex) * outlen );
		for ( size_t k = 0; k < outlen; k++ )
		{
			out[k][0] = 0.0;
			out[k][1] = 0.0;
		}
		

		p = fftw_plan_dft_r2c_1d( MaxTrajectoryLength, in, out, FFTW_ESTIMATE );
		
		zero_out_input();
	}

	~CorrelationFourierTransform()
	{
		fftw_destroy_plan( p );
		fftw_free( in );
		fftw_free( out );
	}

	void calculate_physical_correlation( vector<double> & dipx, vector<double> & dipy, vector<double> & dipz, vector<double> & initial_dip )
	{
		assert( dipx.size() == dipy.size() );
		assert( dipx.size() == dipz.size() );
		assert( dipy.size() == dipz.size() );

		for ( size_t k = 0; k < dipx.size(); k++ )
		{
			physical_correlation.push_back(
				dipx[k] * initial_dip[0] + dipy[k] * initial_dip[1] + dipz[k] * initial_dip[2]
			);
		}
	}

	void zero_out_input( void )
	{
		for ( size_t i = 0; i < MaxTrajectoryLength; i++ )
			in[i] = 0.0;

		physical_correlation.clear();
		//cout << "physical_correlation.size(): " << physical_correlation.size() << endl;
		//cout << "physical_correlation.capacity(): " << physical_correlation.capacity() << endl;
	}

	// create an array of type:
	// [0, ...0, C(delta_t * N), ... , C(delta_t), C(0), C(delta_t), ..., C(delta_t * N), 0, ... 0]
	//void copy_into_fourier_array( void )
	//{
		//double * start_pos = in + (MaxTrajectoryLength - 2 * physical_correlation.size()) / 2 + 1;
		//double * center = in + MaxTrajectoryLength / 2; 

		//std::reverse_copy( physical_correlation.begin(), physical_correlation.end(), start_pos );
		//std::copy( physical_correlation.begin(), physical_correlation.end(), center );
	//}	

	// create an array of type:
	// [C(0), C(delta_t), ..., C(delta_t * N), 0 ... 0]
	void copy_into_fourier_array( void )
	{
		std::copy( physical_correlation.begin(), physical_correlation.end(), in );
	}

	void do_fourier( void )
	{
		fftw_execute( p );
	}

	void show_in( void )
	{
		for ( size_t i = 0; i < MaxTrajectoryLength; i++ )
		{
			std::cout << in[i] << std::endl;
		}
	}

	void show_out( void )
	{
		for ( size_t i = 0; i < MaxTrajectoryLength / 2 + 1; i++ )
		{
			std::cout << out[i][0] << " " << out[i][1] << std::endl;
		}
	}

	double * get_in() const { return in; }
	fftw_complex * get_out() const { return out; }

private:
	int MaxTrajectoryLength;

	vector<double> physical_correlation;

	double * in;
	fftw_complex * out;

	fftw_plan p;
};
