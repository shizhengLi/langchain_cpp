#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

// Configure Catch2
namespace Catch {
    std::ostream& operator<<(std::ostream& os, const Approx& approx) {
        return os << approx.toString();
    }
}