#!/bin/bash 
echo "begin compile ..." 

common_include=../common
bin_file=$(basename "$PWD")

rm ./$bin_file

g++ -O2  -std=c++11 -Wall src/main.cpp $common_include/logger.cpp -I$common_include -lpthread -o $bin_file  

echo "compile done" 

ls -lh $bin_file

./$bin_file
