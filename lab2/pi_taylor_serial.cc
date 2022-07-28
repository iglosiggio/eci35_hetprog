#include <iomanip>
#include <iostream>
#include <limits>

using my_float = float;

auto pi_taylor(size_t steps) -> my_float {
	my_float result = 0;
	my_float sign = 1;
	for (size_t i = 0; i < steps; i++) {
		my_float n = i;
		result += sign / (2.0 * n + 1.0);
		sign *= -1.0;
	}
	return 4.0 * result;
}

int main(int argc, const char *argv[]) {

    // read the number of steps from the command line
    if (argc != 2) {
        std::cerr << "Invalid syntax: pi_taylor <steps>" << std::endl;
        exit(1);

    }

    size_t steps = std::stoll(argv[1]);
    auto pi = pi_taylor(steps);

    std::cout << "For " << steps << ", pi value: "
        << std::setprecision(std::numeric_limits<long double>::digits10 + 1)
        << pi << std::endl;
}
