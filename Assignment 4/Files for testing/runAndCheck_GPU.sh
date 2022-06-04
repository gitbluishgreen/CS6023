#!/bin/bash

numTestCases=5

echo "You are in the $PWD directory now"
echo "You have decided to run this script $numTestCases testcases"

echo "Generating outputs for a4.cu"
nvcc /content/src/a4_v1.cu 
x=1
while [ $x -le $numTestCases ]
do
    ./a.out A4_TestCases_Big/Input/input${x}.txt >  A4_TestCases_Big/MyOutput/output${x}.txt
    echo "Generated output for testcase: $x"
    x=$(( $x + 1 ))
done

echo "Done with generating all the testcases for a4_v1.cu"

echo "Now will start comparing the outputs and also report the average time taken"

echo "" > data.txt
x=1
while [ $x -le $numTestCases ]
do
	python checker.py A4_TestCases_Big/Output/output${x}.txt A4_TestCases_Big/MyOutput/output${x}.txt $x
	echo "Compared output for testcase: $x"
	x=$(( $x + 1 ))
done

echo "Now will find out the average time over all testcases"

python calculate.py data.txt $numTestCases




