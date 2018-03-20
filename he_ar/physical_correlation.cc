#include <mpi.h>

#include <iostream>
#include <iomanip>
#include <random>
#include <ctime>
#include <functional>
#include <algorithm>
#include <cassert>

#include <Eigen/Dense>

// should be included BEFORE Gear header files
// due to namespace overlap
#include <vector>

// matrix multiplication
#include "matrix_he_ar.hpp"
#include "ar_he_pes.hpp"
// FileReader class
#include "file.hpp"
// Parameters class
#include "parameters.hpp"
// SpectrumInfo class
#include "spectrum_info.hpp"
// MCMC_generator class
#include "mcmc_generator.hpp"
// Integrand class
#include "integrand.hpp"
// Integrator class
#include "integrator.hpp"
// miscellaneous functions
#include "correlation_fourier_transform.hpp"
// tags for mpi messages
#include "tags.hpp"
// physical constants
#include "constants.hpp"
// Trajectory class
#include "trajectory.hpp"
// library for FFT
#include <fftw3.h>

// Gear header files
#include "basis.hpp"
#include "vmblock.hpp"
#include "gear.hpp"

using namespace std;
using namespace std::placeholders;
using Eigen::VectorXd;

// ############################################
const double HE_MASS = 4.00260325413;
const double AR_MASS = 39.9623831237; 
const double MU_SI = HE_MASS * AR_MASS / ( HE_MASS + AR_MASS ) * constants::DALTON_UNIT; 
const double MU = MU_SI / constants::AMU;
// ############################################


// "интерфейсная" функция. она передается методу GEAR, который
// осуществляет решение системы дифуров, правая часть которой, задается этой
// функцией. Внутри мы осуществляем вызов функции, которая лежит в файле
// matrix_he_ar.cpp, там осуществлен матричный расчет правой части дифуров.
// Эта функция сделана для удобства, устроена таким образом, чтобы было 
// совместимо с gear4.
void syst (REAL t, REAL *y, REAL *f)
{
	// параметр t мы не используем, от него у нас ничего не зависит
  	(void)(t); // avoid unused parameter warning 
	double *out = new double[4];

	// вызываем функцию, вычисляющую правые части
	rhs( out, y[0], y[1], y[2], y[3] );

	// засовываем в нужном порядке производные
	f[0] = out[0]; // \dot{R} 
	f[1] = out[1]; // \dot{p_R}
	f[2] = out[2]; // \dot{\theta}
	f[3] = out[3]; // \dot{p_\theta}

	delete [] out;
}

// вычисляем гамильтониан двухатомной системы
// используем стандартый порядок аргументов: координата -- импульс
// x = [ R, pR, theta, pT ]
double hamiltonian( VectorXd x )
{
	double R = x( 0 );
	double pR = x( 1 );
	double theta = x( 2 );
	double pT = x( 3 );

	return pow(pR, 2) / (2 * MU) + pow(pT, 2) / (2 * MU * R * R) + ar_he_pot( R );
}

// вычисляем exp(-H/kT); принимаем Eigen::VectorXd координат и температуру
// далее в коде строится функция с bind-ом температуры на ту, которая задана
// в инпут-файле
// x = [ R, pR, theta, pT ]
double numerator_integrand_( VectorXd x, const double& temperature )
{
	double h = hamiltonian( x );
	return exp( -h * constants::HTOJ / (constants::BOLTZCONST * temperature )); 
}

vector<double> create_frequencies_vector( Parameters& parameters )
{
	double FREQ_STEP = 1.0 / (parameters.sampling_time * constants::ATU) / constants::LIGHTSPEED_CM / parameters.MaxTrajectoryLength; // cm^-1
	//cout << "FREQ_STEP: " << FREQ_STEP << endl;
	int FREQ_SIZE = (int) parameters.FREQ_MAX / FREQ_STEP + 1;

	vector<double> freqs( FREQ_SIZE );
	for(int k = 0; k <  FREQ_SIZE; k++) 
	{
		freqs[k] = k * FREQ_STEP;
	}

	return freqs;
}

void master_code( int world_size )
{
	MPI_Status status;
	int source;

	Parameters parameters;
	FileReader fileReader( "parameters_physical_correlation.in", &parameters ); 
	//parameters.show_parameters();

	int sent = 0;	
	int received = 0;

	// status of calculation
	bool is_finished = false;

	vector<double> freqs = create_frequencies_vector( parameters );
	int FREQ_SIZE = freqs.size();

	// creating objects to hold spectrum info
	SpectrumInfo classical( FREQ_SIZE, "classical" );

	function<double(VectorXd)> numerator_integrand = bind( numerator_integrand_, _1, parameters.Temperature );

	// creating MCMC_generator object
	MCMC_generator generator( numerator_integrand, parameters );
	
	// initializing initial point to start burnin from
	VectorXd initial_point = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>( parameters.initial_point.data(), parameters.initial_point.size());	
	generator.burnin( initial_point, 2 );	

	// [ R, pR, theta, pT ]
	generator.set_point_limits()->add_limit(0, 4.0, parameters.RDIST)
								->add_limit(1, -50.0, 50.0)
								->add_limit(2, 0.0, 2 * M_PI)
								->add_limit(3, -250.0, 250.0);

	// allocating histograms to store variables
	generator.set_histogram_limits()->add_limit(0, 4.0, parameters.RDIST)
									->add_limit(1, -50.0, 50.0)
		   							->add_limit(2, 0.0, 2 * M_PI)	
								    ->add_limit(3, -250.0, 250.0);

	vector<string> names { "R", "pR", "theta", "pT" };
	generator.allocate_histograms( names );	

	ofstream distribution_points("distribution_points.txt");

	VectorXd p;
	// sending first trajectory	
	for ( int i = 1; i < world_size; i++ )
	{
		p = generator.generate_free_state_point( hamiltonian );
		generator.show_current_point();

		MPI_Send( p.data(), parameters.DIM, MPI_DOUBLE, i, 0, MPI_COMM_WORLD );
		//cout << "(master) sent point " << endl;

		MPI_Send( &sent, 1, MPI_INT, i, 0, MPI_COMM_WORLD );
		//cout << "(master) sent number of trajectory" << endl;
		
		distribution_points << p[0] << " " << p[1] << " " << p[2] << " " << p[3] << endl;

		sent++;
	}

	clock_t start = clock();

	double * correlation_total = new double [parameters.MaxTrajectoryLength];
	double * correlation_package = new double [parameters.MaxTrajectoryLength];

	for ( size_t k = 0; k < parameters.MaxTrajectoryLength; k++ )
		correlation_total[k] = 0.0;

	while( true )
	{
		if ( is_finished )	
		{
			for ( int i = 1; i < world_size; i++ )
				MPI_Send( &is_finished, 1, MPI_INT, i, tags::EXIT_TAG, MPI_COMM_WORLD );

			break;
		}

		int msg;
		MPI_Recv( &msg, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status );
	   	source = status.MPI_SOURCE;	
		if ( status.MPI_TAG == tags::TRAJECTORY_CUT_TAG )
		{
			cout << "(master) Received cutting trajectory tag!" << endl;
			if ( sent <= parameters.NPOINTS )
			{
				p = generator.generate_free_state_point( hamiltonian );
				generator.show_current_point();

				cout << "(master) Sending new point!" << endl;
				MPI_Send( p.data(), parameters.DIM, MPI_DOUBLE, source, 0, MPI_COMM_WORLD );
				MPI_Send( &sent, 1, MPI_INT, source, 0, MPI_COMM_WORLD );
				cout << "(master) Sent new point" << endl;

				continue;
			}
		}
	
		if ( status.MPI_TAG == tags::TRAJECTORY_FINISHED_TAG )
			cout << "(master) Trajectory is not cut!" << endl;
	
		for ( size_t k = 0; k < parameters.MaxTrajectoryLength; k++ )
			correlation_package[k] = 0.0;

		MPI_Recv( &correlation_package[0], parameters.MaxTrajectoryLength, MPI_DOUBLE, source, MPI_ANY_TAG, MPI_COMM_WORLD, &status );
	
		if ( received > 1 )
		{
			for ( size_t i = 0; i < parameters.MaxTrajectoryLength; i++ )
			{
				// размерность корреляции дипольного момента -- квадрат диполя
				correlation_total[i] += correlation_package[i] * constants::ADIPMOMU * constants::ADIPMOMU;
				correlation_total[i] *= (double) (received - 1) / received;
			}
		}
		else
		{
			for ( size_t i = 0; i < parameters.MaxTrajectoryLength; i++ )
				correlation_total[i] += correlation_package[i] * constants::ADIPMOMU * constants::ADIPMOMU;
		}

		received++;
		cout << "(master) after all MPI_Recv; received = " << received << endl;

		if ( received == parameters.NPOINTS )
		{
			ofstream file( "equilibrium_correlation.txt" );
			for ( size_t k = 0; k < parameters.MaxTrajectoryLength; k++ )
				file << correlation_total[k] << endl;
			file.close();

			is_finished = true;
		}

		if ( sent < parameters.NPOINTS )
		{
			p = generator.generate_free_state_point( hamiltonian );
			generator.show_current_point();

			MPI_Send( p.data(), parameters.DIM, MPI_DOUBLE, source, 0, MPI_COMM_WORLD );
			MPI_Send( &sent, 1, MPI_INT, source, 0, MPI_COMM_WORLD );
		
			distribution_points << p[0] << " " << p[1] << " " << p[2] << " " << p[3] << endl;
			
			sent++;
		}
	}

	delete [] correlation_total;
	delete [] correlation_package;
}

// использует move semantics или Named Return Value Optimization, до конца не разобрался
// но в любом случае здесь мы сливаем два вектора в один, копирования результата НЕ происходит 
vector<double> merge( vector<double> & backward, vector<double> & forward )
{
	vector<double> merged;
	merged.reserve( backward.size() + forward.size() );

	for ( vector<double>::reverse_iterator rit = backward.rbegin(); rit != backward.rend(); ++rit )
		merged.push_back( *rit );		

	for ( vector<double>::iterator it = forward.begin(); it != forward.end(); ++it )
		merged.push_back( *it );

	return merged;
}

void slave_code( int world_rank )
{
	// it would be easier for slave to read parameters file by himself, 
	// rather than sending him parameters object (or just several parameters)
	Parameters parameters;
	FileReader fileReader( "parameters_physical_correlation.in", &parameters ); 

	// #####################################################
	vector<double> freqs = create_frequencies_vector( parameters );
	int FREQ_SIZE = freqs.size();
	double FREQ_STEP = freqs[1] - freqs[0];

	// creating objects to hold spectal info
	SpectrumInfo classical;

	int cut_trajectory = 0;
	bool exit_status = false;
	Trajectory trajectory( parameters );

	CorrelationFourierTransform correlationFT( parameters.MaxTrajectoryLength );

	vector<double> initial_dip;
	vector<double> dipx, dipx_forward, dipx_backward;
	vector<double> dipy, dipy_forward, dipy_backward;
	vector<double> dipz, dipz_forward, dipz_backward;

	vector<double> specfunc_package(FREQ_SIZE);
	vector<double> spectrum_package(FREQ_SIZE);

	while ( true )
	{
		// переменная cut_trajectory играет роль переменной типа bool
		// ( это сделано для того, чтобы переменная могла быть переслана при помощи MPI_Send, 
		// там нет встроенного типа MPI_BOOL. )
		//
		// если траектория оказывается обрублена (длиннее чем parameters.MaxTrajectoryLength точек), 
		// то в классе Trajectory статус траектории становится cut и при помощи метода
		// .report_trajectory_status() получаем статус текущей траектории.
		// В начале каждой итерации мы поэтому должны занулить этот статус здесь и внутри объекта класса Trajectory
		cut_trajectory = 0;
		trajectory.set_cut_trajectory( 0 );	

		clock_t start = clock();

		// реализуем MPI получение начальных условий и запись их в private элементы объекта класса Trajectory
		// возвращаем статус полученного сообщения. Если статус полученного сообщения таков, что больше траекторий
		// обсчитывать не надо, то exit_status = true и, соответственно, текущий процесс выходит из бесконечного цикла
		// и прекращает свою работу.
		exit_status = trajectory.receive_initial_conditions( );
		if ( exit_status )
		{	
			cout << "(" << world_rank << ") exit message is received!" << endl;
			break;
		}

		// тут получаем вектор диполя, соответствующий начальному положению: mu(0) = (mu_x(0), mu_y(0), mu_z(0)); 
		initial_dip = trajectory.get_initial_dip();

		// начинаем траекторию из полученного начального положения в фазовом пространстве
		trajectory.run_trajectory( syst );

		// если мы прошли предыдущий блок кода, значит траектория имеет допустимую длину.
		// копируем компоненты дипольного момента вдоль траектории в виде структуры данных vector<double>  
		dipx_forward = trajectory.get_dipx();
		dipy_forward = trajectory.get_dipy();
		dipz_forward = trajectory.get_dipz();
		// после копирования освобождаем эти вектора внутри объекта trajectory	
		trajectory.dump_dipoles( );
		
		// обращаем импульсы в полученных начальных условиях
		// trajectory.reverse_initial_conditions();

		// запускаем траекторию теперь уже обращенную во времени 
		// ( если теперь пройти по этой траектории некоторое время, остановится, снова обратить импульсы 
		// и пойти по ней, то мы вернемся в исходную точку и пойдем по траектории, которая уже была пройдена -- обозначена как forward).
		// trajectory.run_trajectory( syst );
		
		// если траектория оказывается длиннее, чем максимально допустимая длина, то объект класса Trajectory посылает
		// запрос мастер-процессу на получение новой начальной точки фазового пространства. Текущий процесс зачищает 
		// накопленные на текущей траектории компоненты дипольного момента методом .dump_dipoles(), и переходим в начало
		// текущего цикла, готовимся получать новый набор начальных условий
		// 
		// Не будем отсылать сообщение мастеру, если оборвется forward часть траектории, потому что это заставит мастера
		// ждать расчета второй части траектории. В это время он не может обрабатывать запросы других рабов. Время мастера
		// дороже времени раба -- холостой прогон backward траектории в масштабах программы дешевле.
		cut_trajectory = trajectory.report_trajectory_status( );
		if ( cut_trajectory == 1 )
		{
			trajectory.dump_dipoles();
			continue;
		}

		// если мы прошли проверку, то траектория имеет допустимую длину.
		// копируем компоненты дипольного момента вдоль backward-траектории в виде vector<double>
		/*
		dipx_backward = trajectory.get_dipx();
		dipy_backward = trajectory.get_dipy();
		dipz_backward = trajectory.get_dipz();
		*/
		// после копирования освобождаем эти вектора внутри объекта trajectory
		//trajectory.dump_dipoles();

		// удаляем первый элемент каждого массива, т.к. он относится к начальной точке и уже находится
		// в массивах dipx_forward, dipy_forward, dipz_forward 
		/*
		dipx_backward.erase( dipx_backward.begin() );
		dipy_backward.erase( dipy_backward.begin() );
		dipz_backward.erase( dipz_backward.begin() );
		*/

		// объединяем пары векторов dipx_backward и dipx_forward, предварительно обратив первый
		// результат записываем в dipx. Аналогично с другими компонентами.
		//
		// внутри функции merge размер dipx изначально резервируется при помощи метода reserve()
		// после того, как мы закончим работу с dipx, dipy, dipz в рамках текущей итерации надо не забыть
		// освободить место внутри них при помощи метода clear() 
		/*
		dipx = merge( dipx_backward, dipx_forward );	
		dipy = merge( dipy_backward, dipy_forward );
		dipz = merge( dipz_backward, dipz_forward );
		*/

		// прежде чем продолжать, добьемся того, чтобы вектора dipx, dipy и dipz были четной длины
		// это важно для симметризации, чтобы точка симметрии лежала между двумя центральными элементами
		// если длина массивов нечетна, то в таком случае просто избавимся от последнего элемента.
		//
		// .end() - 1 указывает на последний элемент, потому что в STL принято, что .end() указывает на 
		// элемент следующий за последним!
		/*
		if ( dipx.size() % 2 != 0 )
		{
			dipx.erase( dipx.end() - 1);
			dipy.erase( dipy.end() - 1);
			dipz.erase( dipz.end() - 1);
		}
		*/

		// рассчитываем корреляцию векторов дипольного момента с начальной ориентацией
		// результат сохраняется в зарезервированном vector<double> внутри объекта correlationFT
		correlationFT.calculate_physical_correlation( dipx_forward, dipy_forward, dipz_forward );

		cout << "(" << world_rank << ") Processing " << trajectory.get_trajectory_counter() << " trajectory. npoints = " << dipz.size() << "; time = " << (clock() - start) / (double) CLOCKS_PER_SEC << "s" << endl;
		
		// теперь освобождаем вектора, хранящие дипольной момент траектории, рассчитываемой на текущей итерации
		//dipx.clear();
		//dipy.clear();
		//dipz.clear();

		// симметризуем массив корреляций
		// реализуется как сложение исходного массива с обращенным исходным массивом
		// затем деление всех элементов пополам  
 		// correlationFT.symmetrize( );

		// копируем вектор корреляций дипольного момента в подготовленный массив длиной 2^n
		correlationFT.copy_into_fourier_array( );

		// Отправляем собранный массив корреляций мастер-процессу
		MPI_Send( &correlationFT.get_in()[0], parameters.MaxTrajectoryLength, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD );		
		correlationFT.zero_out_input();
	}
}

int main( int argc, char* argv[] )
{
	//Initialize the MPI environment
	MPI_Init( &argc, &argv );

	//getting id of the current process
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	//getting number of running processes
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size); 

	if ( world_rank == 0 ) 
	{
		clock_t start = clock();

		master_code( world_size );

		cout << "Time elapsed: " << (clock() - start) / (double) CLOCKS_PER_SEC << "s" << endl;
	}	
	else
	{
		slave_code( world_rank );
	}

	MPI_Finalize();

	return 0;
}
