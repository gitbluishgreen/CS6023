import numpy as np 
import sys 

a1 = np.loadtxt(sys.argv[1])
num = int(sys.argv[2])

a1_avg = np.sum(a1) / num
print("The average time taken are as follows")
print("Version 1:", a1_avg)

