#pragma once

#include <fftw3.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <vector>
#include <iostream>

#define REALPART 0
#define IMAGPART 1

class Fourier
{
	public:
		int MaxTrajectoryLength;

		double *inx;
		double *iny;
		double *inz;

		fftw_complex *outx;
		fftw_complex *outy;
		fftw_complex *outz;

		fftw_plan px;
		fftw_plan py;
		fftw_plan pz;

		void copy_into_fourier_array( std::vector<double>& v, std::string type );		
		void zero_out_input( void );
		void do_fourier( void );
		
		Fourier( int MaxTrajectoryLength );
		~Fourier();
};

// copy from vector to fftw_complex*
void copy_to( std::vector<double> &v, fftw_complex* arr );
void copy_to( std::vector<double> &v, double* arr );

std::vector<double> linspace( const double min, const double max, const int npoints );

std::vector<double> fft_one_side( std::vector<double> signal );
std::vector<double> fft_two_side( std::vector<double> &signal );

void fftfreq( std::vector<double> &freqs, const int len, const double d );
void fftshift( std::vector<double> &arr, const int len );
void ifftshift( std::vector<double> &arr, const int len );

void fft_positive( std::vector<double> &signal );

