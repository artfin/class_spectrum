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

        output.push_back( res / (input.size() - n + 1) );
    }
}

int main()
{
    vector<string> content;
    readFromFile("../input.dat", content);

    vector<double> input;
    parseFile(content, input);

    for ( size_t k = 0; k < input.size(); k++ )
        cout << input[k] << " ";
    cout << endl;

    vector<double> output;
    calculateCorrelation( input, output );

    for ( size_t k = 0; k < output.size(); k++ )
        cout << output[k] << endl;


    return 0;
}
