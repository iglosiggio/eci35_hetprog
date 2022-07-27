#include <iostream>
#include <random>
#include <vector>
#include <algorithm>
#include <functional>

#include <add_array.hh>
using namespace add_array;

struct Poly {
	float t = 0;
	float dt, a, b, c;
	Poly(float _dt, float _a, float _b, float _c): dt(_dt), a(_a), b(_b), c(_c) {}
	float operator()() {
		float result = a * t * t + b * t * c;
		t += dt;
		return result;
	}
};

int main() {
	std::vector<float> A(n), B(n), C(n);
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dis(0.0f, 1.0f);

	// Initialize the array with random values
	A[0] = dis(rd);
	B[0] = dis(rd);

	// Initialize the rest of the array with a polynomial
	std::generate(A.begin()+1, A.end(), Poly(0.5,    2, 1.3, 6));
	std::generate(B.begin()+1, B.end(), Poly(  7, 0.02,   1, 0));

	// Add the two vectors
	std::transform(A.begin(), A.end(), B.begin(), C.begin(), [](auto a, auto b) {
		//return a*a  + b*b + a*b;
		return a + b;
	});

	// Print the first 8 elements
	std::for_each(C.begin(), C.begin()+8, [](float v) {
		std::cout << v << std::endl;
	});
}
