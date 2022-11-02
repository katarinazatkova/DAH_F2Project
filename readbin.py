import matplotlib.pyplot as plt
import numpy as np
f = open("kkp.bin","r")
b = np.fromfile(f,dtype=np.float32)
ncol = 7
# number of events
nevent = len(b)/ncol
x = np.split(b,nevent)
#np.set_printoptions(suppress=False, threshold=10000)
for a in x:	print (a)

#plt.plot(x[0])
