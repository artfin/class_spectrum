#pragma once

#include <string>
#include <iostream>
#include <stdexcept>

struct Limit
{
	int var_number;
	std::string lb_str; 
	double ub;
	double lb;
	std::string ub_str;
	
	
	// change of variables 
	enum chvarTypes { INFINF, FINITEINF, FINITE } chvarType;
	enum limitTypes { DOUBLE, STRING } ubType, lbType;
	
	void show_limit( void )
	{
		if ( lbType == limitTypes::DOUBLE && ubType == limitTypes::DOUBLE )
			std::cout << "(limit " << var_number << "): (DOUBLE) " << lb << " -- (DOUBLE) " << ub << std::endl;
		if ( lbType == limitTypes::DOUBLE && ubType == limitTypes::STRING )
			std::cout << "(limit " << var_number << "): (DOUBLE) " << lb << " -- (STRING) " << ub_str << std::endl;
		if ( lbType == limitTypes::STRING && ubType == limitTypes::DOUBLE )
			std::cout << "(limit " << var_number << "): (STRING) " << lb_str << " -- (DOUBLE) " << ub << std::endl;
		if ( lbType == limitTypes::STRING && ubType == limitTypes::STRING )
			std::cout << "(limit " << var_number << "): (STRING) " << lb_str << " -- (STRING) " << ub_str << std::endl;
	}

	Limit( int var_number, double lb, double ub ) : 
		var_number(var_number), lb(lb), ub(ub) 
	{
		ubType = limitTypes::DOUBLE;
		lbType = limitTypes::DOUBLE;

		chvarType = chvarTypes::FINITE;
	}
	
	Limit( int var_number, std::string lb_str, double ub ) : 
		var_number(var_number), lb_str(lb_str), ub(ub)
	{
		lbType = limitTypes::STRING;
		ubType = limitTypes::DOUBLE;
	}

	Limit( int var_number, double lb, std::string ub_str ) :
		var_number(var_number), lb(lb), ub_str(ub_str)
	
	{
		lbType = limitTypes::DOUBLE;
		ubType = limitTypes::STRING;

		chvarType = chvarTypes::FINITEINF;
		cout << "(lb_str): " << lb << "; (ub_str): " << ub_str << endl;
	}

	Limit( int var_number, std::string lb_str, std::string ub_str ) :
		var_number(var_number), lb_str(lb_str), ub_str(ub_str)
	{
		lbType = limitTypes::STRING;
		ubType = limitTypes::STRING;

		if ( lb_str == "-inf" && ub_str == "+inf" ) chvarType = chvarTypes::INFINF;
		else throw std::invalid_argument( "Unknown bounds!" );
	}
};


