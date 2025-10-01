# Add Catch2 testing framework
include(FetchContent)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.4.0
)

FetchContent_MakeAvailable(Catch2)

# Note: nlohmann/json removed due to CMake version compatibility issues
# Will implement simple JSON parsing for OpenAI API responses