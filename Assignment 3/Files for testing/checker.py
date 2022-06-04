import numpy as np 
import sys
a1 = np.loadtxt(sys.argv[1])
a2 = np.loadtxt(sys.argv[2])
if(np.array_equal(a1,a2)):
  print("Passed!")
else:
  print("Failed!")