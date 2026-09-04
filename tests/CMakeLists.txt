find_package(Catch2 2 REQUIRED)
include(Catch)

add_library(acn_test_main OBJECT test_main.cpp)
target_link_libraries(acn_test_main PUBLIC acn::acn Catch2::Catch2)

function(acn_add_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE acn::acn Catch2::Catch2 acn_test_main)
    target_compile_options(${name} PRIVATE -Wall -Wextra -Wpedantic -Wshadow)
    catch_discover_tests(${name})
endfunction()

acn_add_test(test_int_backend)
acn_add_test(test_ring)
acn_add_test(test_canonical)
acn_add_test(test_static)
acn_add_test(test_dynamic)
acn_add_test(test_convert)
acn_add_test(test_divide)
acn_add_test(test_format)
acn_add_test(test_interval)
acn_add_test(test_matrix)
acn_add_test(test_gates)
acn_add_test(test_circuits)
acn_add_test(test_echelon)

if(ACN_WITH_GMP)
    acn_add_test(test_bigint)
    acn_add_test(test_overflow)

    # tests/reference: PLAN.md §9.5's independent oracle. BigInt-based
    # throughout, so it only exists when GMP is available, same as the
    # rest of the BigInt-gated tests.
    add_library(acn_reference_oracle OBJECT reference/reference.cpp)
    target_link_libraries(acn_reference_oracle PUBLIC acn::acn)
    target_compile_options(acn_reference_oracle PRIVATE -Wall -Wextra -Wpedantic -Wshadow)

    add_executable(test_reference_oracle test_reference_oracle.cpp)
    target_link_libraries(test_reference_oracle PRIVATE acn::acn Catch2::Catch2 acn_test_main acn_reference_oracle)
    target_include_directories(test_reference_oracle PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/reference)
    target_compile_options(test_reference_oracle PRIVATE -Wall -Wextra -Wpedantic -Wshadow)
    catch_discover_tests(test_reference_oracle)
endif()

if(ACN_STATS)
    acn_add_test(test_stats)
endif()
