#include <iostream>
#include <random>
#include <vector>

#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

using namespace sycl;

const size_t pipe_entries = 16; // ensure 512 bit burst

template <typename dst_pipe_id, typename kernel_name>
auto read_from_buffer(queue& q, sycl::buffer<float, 1>& buf_src) {
	using output = pipe<dst_pipe_id, float, pipe_entries>;
	return q.submit([&](auto& h) {
		auto src = buf_src.get_access(h, read_only);
		h.template single_task<kernel_name>([=] {
			size_t size = src.size();
			for (size_t i = 0; i < size; i++) {
				output::write(src[i]);
			}
		});
	});
}

template <typename lhs_pipe_id, typename rhs_pipe_id, typename dst_pipe_id, typename kernel_name>
auto add_pipes(queue& q, size_t work_size) {
	using lhs_input = pipe<lhs_pipe_id, float, pipe_entries>;
	using rhs_input = pipe<rhs_pipe_id, float, pipe_entries>;
	using output = pipe<dst_pipe_id, float, pipe_entries>;
	return q.submit([&](auto& h) {
		h.template single_task<kernel_name>([=] {
			for (size_t i = 0; i < work_size; i++) {
				output::write(lhs_input::read() + rhs_input::read());
			}
		});
	});
}

template <typename src_pipe_id, typename kernel_name>
auto write_to_buffer(queue& q, sycl::buffer<float, 1>& buf_dst) {
	using input = pipe<src_pipe_id, float, pipe_entries>;
	return q.submit([&](auto& h) {
		auto dst = buf_dst.get_access(h, write_only);
		h.template single_task<kernel_name>([=] {
			size_t size = dst.size();
			for (size_t i = 0; i < size; i++) {
				dst[i] = input::read();
			}
		});
	});
}

int main() {
	const size_t n = 32;

	std::vector<float> A, B, C(n);

	A.reserve(n);
	B.reserve(n);

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dis(0.0f, 1.0f);

	float value = dis(gen);
	A.push_back(value);
	B.push_back(value - 1.0f);

	for(size_t i = 1; i < n; ++i) {
		A.push_back(A[0]+i);
		B.push_back(B[0]+i);
	}

#if defined(FPGA_EMULATOR)
	sycl::ext::intel::fpga_emulator_selector device_selector;
#else
	sycl::ext::intel::fpga_selector device_selector;
#endif

	// property list to enable SYCL profiling for the device queue
	// auto props = property_list{property::queue::enable_profiling()};

	sycl::queue q(device_selector);
	sycl::buffer buf_A {A}, buf_B {B}, buf_C {C};
	auto read_A = read_from_buffer<class stream_A, class read_from_A>(q, buf_A);
	auto read_B = read_from_buffer<class stream_B, class read_from_B>(q, buf_B);
	auto work = add_pipes<class stream_A, class stream_B, class stream_C, class add_streams>(q, n);
	auto write_C = write_to_buffer<class stream_C, class write_to_C>(q, buf_C);
	write_C.wait();

	for (int i = 0; i < 8; i++) {
		std::cout << "C[" << i << "] = " << C[i] << std::endl;
	}
}
