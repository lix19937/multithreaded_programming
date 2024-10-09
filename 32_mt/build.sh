#!/bin/bash 
echo "begin compile ..." 

common_include=../common
bin_file=$(basename "$PWD")

cd build  

cmake ../  && make -j12
