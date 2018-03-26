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
		in.reserve( MaxTrajectoryLength );
		out.reserve( MaxTrajectoryLength );

		p = fftw_plan_r2r_1d( MaxTrajectoryLength, &in[0], &out[0], FFTW_REDFT10, FFTW_ESTIMATE ); 
		
		zero_out_input();
		//in = (double*) fftw_malloc( sizeof(double) * MaxTrajectoryLength );
		//out = (fftw_complex*) fftw_malloc( sizeof(fftw_complex) * outlen );
		//for ( size_t k = 0; k < outlen; k++ )
		//{
			//out[k][0] = 0.0;
			//out[k][1] = 0.0;
		//}
		

		//p = fftw_plan_dft_r2c_1d( MaxTrajectoryLength, in, out, FFTW_ESTIMATE );
	}

	~CorrelationFourierTransform()
	{
		fftw_destroy_plan( p );
		//fftw_free( in );
		//fftw_free( out );
	}

	void calculate_physical_correlation( vector<double> & dipx, vector<double> & dipy, vector<double> & dipz )
	{
		int size = dipx.size();
		assert( dipy.size() == size ); 
		assert( dipz.size() == size ); 

		physical_correlation.resize( size );
		
		double res = 0;
		for ( size_t n = 0; n < size; n++ )
		{
			res = 0;
			for ( size_t i = 0, j = n; j < size; i++, j++ )
			{
				res += dipx[i] * dipx[j];
				res += dipy[i] * dipy[j];
				res += dipz[i] * dipz[j];
			}

			physical_correlation[n] = res / (size - n);
		}
	}
	
	void symmetrize( ) 
	{
		int size = physical_correlation.size();

		for ( size_t k = 0; k < size / 2; k++ )
			physical_correlation[k] = 0.5 * ( physical_correlation[k] + physical_correlation[ size - k - 1] );
		for ( size_t k = size / 2; k < size; k++ )
			physical_correlation[k] = physical_correlation[size - k - 1];
	}

	void zero_out_input( void )
	{
		for ( size_t i = 0; i < MaxTrajectoryLength; i++ )
			in[i] = 0.0;

		physical_correlation.clear();
	}

	//void copy_into_fourier_array( void )
	//{
		//double * start_pos = in + (MaxTrajectoryLength - 2 * physical_correlation.size()) / 2 + 1;
		//double * center = in + MaxTrajectoryLength / 2; 

		//std::reverse_copy( physical_correlation.begin(), physical_correlation.end(), start_pos );
		//std::copy( physical_correlation.begin(), physical_correlation.end(), center );
	//}	

	/*
	void copy_into_fourier_array( void )
	{
		size_t i = 0;
		size_t center_pos = MaxTrajectoryLength / 2;
		for ( std::vector<double>::iterator it = physical_correlation.begin(); it != physical_correlation.end(); ++it, ++i )
			in[center_pos + i] = *it;

		i = 0;
		size_t start_pos = center_pos - physical_correlation.size();	
		for ( std::vector<double>::reverse_iterator rit = physical_correlation.rbegin(); rit != physical_correlation.rend(); ++rit, ++i )
			in[start_pos + i] = *rit;
	}
	*/

	void copy_into_fourier_array( void )
	{
		std::copy( physical_correlation.begin(), physical_correlation.end(), &in[0] );
	}

	void do_fourier( double dt )
	{
		fftw_execute( p );

		double denominator = 2 * MaxTrajectoryLength * dt * MaxTrajectoryLength * M_PI;
		for ( int k = 0; k < MaxTrajectoryLength; k++ )
			out[k] /= denominator; 
	}

	//void show_in( void )
	//{
		//for ( size_t i = 0; i < MaxTrajectoryLength; i++ )
		//{
			//std::cout << in[i] << std::endl;
		//}
	//}

	//void show_out( void )
	//{
		//for ( size_t i = 0; i < MaxTrajectoryLength / 2 + 1; i++ )
		//{
			//std::cout << out[i][0] << " " << out[i][1] << std::endl;
		//}
	//}

	double * get_in() { return &in[0]; } 
	double * get_out() { return &out[0]; }  

	//double * get_in() const { return in; }
	//fftw_complex * get_out() const { return out; }
	
	vector<double> physical_correlation;

private:
	int MaxTrajectoryLength;

	vector<double> in;
	vector<double> out;

	//double * in;
	//fftw_complex * out;

	fftw_plan p;
};
