#include <iostream>
#include <random>
#include <vector>
#include <algorithm>
#include <functional>

#include <CL/sycl.hpp>
using namespace sycl;


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

	// Choose the set of devices to work with
	auto device_selector = default_selector{};
	//auto device_selector = gpu_selector{};
	//auto device_selector = accelerator_selector{};
	//auto device_selector = host_selector{};
	//auto device_selector = cpu_selector{};

	// Add the two vectors
	{
		buffer bufA {A}, bufB {B}, bufC {C};
		queue q(device_selector);
		q.submit([&](handler &h) {
			auto A = bufA.get_access(h, read_only);
			auto B = bufB.get_access(h, read_only);
			auto C = bufC.get_access(h, write_only);
			h.parallel_for(n, [=](auto i) [[intel::kernel_args_restrict]] {
				auto a = A[i];
				auto b = B[i];
				auto& c = C[i];
				//c = a*a + b*b + a*b;
				c = a + b;
			});
		});
	}

	// Print the first 8 elements
	std::for_each(C.begin(), C.begin()+8, [](float v) {
		std::cout << v << std::endl;
	});

	// Print the devices that were available
	std::cout << "Devices:" << std::endl;
	for (auto device : sycl::device::get_devices()) {
		auto score = device_selector(device);
		std::cout << " - "
			  << "(score=" << score << ")\t"
		          << device.get_info<sycl::info::device::name>()
			  << std::endl;
	}
}
