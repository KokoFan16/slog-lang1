set -e

# go to this scripts location to run tests.
cd $(dirname $(realpath $0))

rm -rf ./output
mkdir ./output
echo "generating c++ test files ..."
racket ./compile-test-file-generator.rkt

echo "test 1: builtins-tests-generated.cpp"
echo "compiling ..."
g++ -g ./output/builtins-tests-generated.cpp -o ./output/builtins-tests -std=c++1z
echo "running test ..."
./output/builtins-tests

echo "====================================================="
echo "test 2: builtins-tests2-generated.cpp"
echo "compiling ..."
# g++ ./output/builtins-tests2-generated.cpp -o ./output/builtins-tests2 -std=c++1z -O3 -finline-limit=100000000 -flto --param max-inline-recursive-depth-auto=100 \
#  --param max-inline-insns-recursive-auto=10000000 -Winline
clang++ -g ./output/builtins-tests2-generated.cpp -o ./output/builtins-tests2 -std=c++1z -O3
echo "running test ..."
./output/builtins-tests2

# aggregators have changed ... TODO: update the aggregators test
# echo "====================================================="
# echo "test 3: aggregators-tests-generated.cpp"
# echo "compiling ..."
# clang++ ./output/aggregators-tests-generated.cpp -o ./output/aggregators-tests -std=c++1z -O3
# echo "running test ..."
# ./output/aggregators-tests

echo " "
echo "done"
