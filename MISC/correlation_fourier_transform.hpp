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
	}

	void copy_into_fourier_array( void )
	{
		double * start_pos = in + (MaxTrajectoryLength - 2 * physical_correlation.size()) / 2;
		double * center = in + MaxTrajectoryLength / 2 + 1;

		std::reverse_copy( physical_correlation.begin(), physical_correlation.end(), start_pos );
		std::copy( physical_correlation.begin(), physical_correlation.end(), center );
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

private:
	int MaxTrajectoryLength;

	vector<double> physical_correlation;

	double * in;
	fftw_complex * out;

	fftw_plan p;
};
