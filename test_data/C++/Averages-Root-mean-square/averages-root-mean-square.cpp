#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>

int main( ) {
  std::vector<int> numbers ;
  for ( int i = 1 ; i < 11 ; i++ )
    numbers.push_back( i ) ;
  double meansquare = sqrt( ( std::inner_product( numbers.begin(), numbers.end(), numbers.begin(), 0 ) ) / static_cast<double>( numbers.size() ) );
  std::cout << "The quadratic mean of the numbers 1 .. " << numbers.size() << " is " << meansquare << " !\n" ;
  return 0 ;
}
