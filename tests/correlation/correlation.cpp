#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>

using namespace std;

void readFromFile( const string & filename, vector<string> & content )
{
    ifstream dipfile( filename );

    string line;
    while( getline(dipfile, line) )
        content.push_back(line);


    dipfile.close();
}

void parseFile(vector<string> & content, vector<double> & input)
{
    double value;
    for ( size_t k = 0; k < content.size(); k++ )
    {
        stringstream(content[k]) >> value;
        input.push_back(value);
    }
}

void calculateCorrelation( vector<double> & input, vector<double> & output )
{
    double res = 0;

    for ( size_t n = 0; n < input.size(); n++ )
    {
        res = 0;
        for ( size_t i = 0, j = n; j < input.size(); i++, j++ )
            res = res + input[i] * input[j];

        output.push_back( res / (input.size() - n) );
    }
}

int main()
{
    vector<double> input{1.0, 1.0, 1.0};

    vector<double> output;
    calculateCorrelation( input, output );

    cout << "output: ";
    for ( size_t i = 0; i < output.size(); i++ )
        cout << output[i] << " ";
    cout << endl;

    return 0;
}
