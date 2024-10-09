# only support ubuntu >= 18.04. which has cmake 3.10
# the reason why we need it is because FindCUDA/find_package(CUDA)
# is depricated since cmake 3.10.
# refer to https://cmake.org/cmake/help/v3.25/module/FindCUDA.html
cmake_minimum_required(VERSION 3.10)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(TASK_BENCH)

enable_language(CUDA)
find_package(CUDA)
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -fpic -fpie -fpermissive -std=c++11 -pthread -W -Wall")

# Set the CUDA architecture to compile for  -arch=sm_70; -arch=sm_72; -arch=sm_75; 
set(CUDA_ARCH "-arch=sm_80" CACHE STRING "CUDA architecture")

# Add the CUDA compiler flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${CUDA_ARCH})

include_directories("${CMAKE_SOURCE_DIR}/src/")
include_directories("${CMAKE_SOURCE_DIR}/src/customtasks/")

link_directories(/usr/local/cuda/lib64) 
include_directories(/usr/local/cuda/include)   

link_directories(/usr/lib/wsl/lib) # /usr/lib/wsl/lib/libcuda.so

# link_directories(/usr/local/cudnn/lib/x86_64-linux-gnu)
# include_directories(/usr/local/cudnn/include/x86_64-linux-gnu)   
# link_directories(/usr/local/tensorrt/lib/x86_64-linux-gnu)
# include_directories(/usr/local/tensorrt/include/x86_64-linux-gnu)  

## cutask
FILE(GLOB_RECURSE CU_SRC src/*.cu)

cuda_add_library(cutask SHARED ${CU_SRC}) 
target_link_libraries(cutask cuda)

## main
FILE(GLOB_RECURSE CC_SRC src/*.cpp)
add_executable(task_bench ${CC_SRC})
target_link_libraries(task_bench cudart cutask)
