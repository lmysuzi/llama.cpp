set(TARGET llama-powerinfer-compare)
add_executable(${TARGET} compare.cpp)
target_link_libraries(${TARGET} PRIVATE llama ggml ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/vendor)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

add_executable(test-relullama-outprod test-outprod.cpp)
target_link_libraries(test-relullama-outprod PRIVATE ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(test-relullama-outprod PRIVATE cxx_std_17)
