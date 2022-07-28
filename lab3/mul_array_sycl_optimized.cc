#include <iostream>
#include <random>
#include <vector>

#include <CL/sycl.hpp>

namespace {
  constexpr size_t n = 1024;
};

class optimized;

void multiply_reduce(const std::vector<float>& A, std::vector<float>& res) {
	sycl::default_selector device_selector;
	sycl::queue q(device_selector);

	size_t length = A.size();
	cl::sycl::buffer to_reduce {A}, to_store_result {res};
	q.submit([&](auto& h) {
		auto A = to_reduce.get_access(h, cl::sycl::read_only);
		auto res = to_store_result.get_access(h, cl::sycl::write_only);
		h.single_task([=] {
			constexpr size_t M = 5;

			float result_copies[M];
			#pragma unroll
			for (size_t i = 0; i < M; i++) {
				result_copies[i] = 1.0;
			}

			for (size_t i = 0; i < length; i++) {
				float cur = result_copies[0] * A[i];

				#pragma unroll
				for (size_t i = 0; i < M-1; i++) {
					result_copies[i] = result_copies[i+1];
				}
				result_copies[M-1] = cur;
			}

			float result = 1.0;
			#pragma unroll
			for (size_t i = 0; i < M; i++) {
				result *= result_copies[i];
			}
			res[0] = result;
		});
	});
}


int main()
{
    std::vector<float> A, res(1);

    A.reserve(n);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 0.25f);

    float value = dis(gen);
    A.push_back(value);

    for(size_t i = 1; i < n; ++i) {
	A.push_back(A[0] + static_cast<float>(i)/static_cast<float>(n));
    }

    multiply_reduce(A, res);

    std::cout << "res[0] = " << res[0] << std::endl;
}
