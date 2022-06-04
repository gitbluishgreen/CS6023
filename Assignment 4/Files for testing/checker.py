import numpy as np
import sys
output_file = sys.argv[1]
myoutput_file = sys.argv[2]
num = int(sys.argv[3])

def file_read(fname):
  with open(fname) as f:
    array = []
    for line in f: # read rest of lines
        y = [x for x in line.split()]
        array.append(y)
    return array

output_arr = file_read(output_file)
myoutput_arr = file_read(myoutput_file)

if(output_arr == myoutput_arr[1:]):
	print("Testcase(version1):", num, " PASSED!!")
else:
	print("Testcase:(version1)", num, "FAILED..........................")


v1_time = myoutput_arr[0][0]

f1 = open('data.txt', 'a')

f1.write(str(v1_time) + '\n')

f1.close()


