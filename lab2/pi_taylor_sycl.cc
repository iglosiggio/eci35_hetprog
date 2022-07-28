#include <CL/sycl.hpp>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using my_float = float;
using tid_time = std::pair<std::thread::id, double>;

template<typename F, typename... Args>
tid_time
time_ms(F f, Args&&... args)
{
	auto start = std::chrono::steady_clock::now();
	f(std::forward<Args>(args)...);
	auto stop = std::chrono::steady_clock::now();

	auto tid = std::this_thread::get_id();
	double ex_time = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start).count();
	return std::make_pair(tid, ex_time);
}

auto pi_taylor_dumb(size_t steps, sycl::queue& q) -> my_float {
	std::vector<my_float> to_reduce(steps);
	{
		cl::sycl::buffer buf_to_reduce {to_reduce};
		q.submit([&](auto& h) {
			auto to_reduce = buf_to_reduce.get_access(h, cl::sycl::write_only);
			h.parallel_for(steps, [=](auto i) {
				my_float n = i;
				my_float sign = (i % 2 == 0) ? 1.0 : -1.0;
				to_reduce[i] = sign / (2.0f * n + 1.0);
			});
		}).wait();
	}
	return 4.0 * std::reduce(to_reduce.begin(), to_reduce.end());
}

auto pi_taylor_reduce_jumping(size_t steps, sycl::queue& q) -> my_float {
	auto work_group_size = q.get_device().get_info<cl::sycl::info::device::max_work_group_size>();
	auto compute_units = q.get_device().get_info<cl::sycl::info::device::max_compute_units>();
	auto parallel_work_items = work_group_size * compute_units;

	std::cout << parallel_work_items << " " << work_group_size << "x" << compute_units << std::endl;

	std::vector<my_float> to_reduce(parallel_work_items);
	{
		cl::sycl::buffer buf_to_reduce {to_reduce};
		q.submit([&](auto& h) {
			auto to_reduce = buf_to_reduce.get_access(h, cl::sycl::write_only);
			h.parallel_for(parallel_work_items, [=](auto work_id) {
				float result = 0.0;
				for (size_t i = work_id; i < steps; i += parallel_work_items) {
					my_float n = i;
					my_float sign = (i % 2 == 0) ? 1.0 : -1.0;
					result += sign / (2.0f * n + 1.0);
				}
				to_reduce[work_id] = result;
			});
		}).wait();
	}
	return 4.0 * std::reduce(to_reduce.begin(), to_reduce.end());
}

auto pi_taylor_reduce_block(size_t steps, sycl::queue& q) -> my_float {
	auto work_group_size = q.get_device().get_info<cl::sycl::info::device::max_work_group_size>();
	auto compute_units = q.get_device().get_info<cl::sycl::info::device::max_compute_units>();
	auto parallel_work_items = work_group_size * compute_units;
	auto work_item_size = (steps + (parallel_work_items - 1)) / parallel_work_items;

	std::cout << parallel_work_items << " " << work_group_size << "x" << compute_units << std::endl;

	std::vector<my_float> to_reduce(parallel_work_items);
	{
		cl::sycl::buffer buf_to_reduce {to_reduce};
		q.submit([&](auto& h) {
			auto to_reduce = buf_to_reduce.get_access(h, cl::sycl::write_only);
			h.parallel_for(parallel_work_items, [=](auto work_id) {
				float result = 0.0;
				size_t start_i = work_id * work_item_size;
				for (size_t i = start_i; i < start_i + work_item_size; i++) {
					my_float n = i;
					my_float sign = (i % 2 == 0) ? 1.0 : -1.0;
					result += sign / (2.0f * n + 1.0);
				}
				to_reduce[work_id] = result;
			});
		}).wait();
	}
	return 4.0 * std::reduce(to_reduce.begin(), to_reduce.end());
}

int
main(int argc, const char *argv[])
{
	// read the number of steps from the command line
	if (argc != 2) {
		std::cerr << "Invalid syntax: pi_taylor <steps>" << std::endl;
		exit(1);
	}

	auto steps = std::stoll(argv[1]);

	sycl::default_selector device_selector;
	sycl::queue q(device_selector);

	auto work_group_size = q.get_device().get_info<cl::sycl::info::device::max_work_group_size>();

	if (steps < work_group_size ) {
		std::cerr << "The number of steps should be larger than " << work_group_size << std::endl;
		exit(1);

	}

	my_float pi = pi_taylor_reduce_block(steps, q);

	std::cout << "For " << steps << " steps, pi value: "
		<< std::setprecision(std::numeric_limits<long double>::digits10 + 1)
		<< pi << std::endl;
}

