#%%

import numpy as np

with open("freq_matrix_without_x.csv") as f:
    ncols = len(f.readline().split(','))

data = np.loadtxt("freq_matrix_without_x.csv", delimiter=',', skiprows=1, usecols=range(1, ncols))

f = open("freq_data_without_x.txt", "w")
for row, a in enumerate(data):
    f.write(str(int(np.count_nonzero(data[row] >= 1))))
    for i, value in enumerate(data[row]):
        if(value >= 1): 
            f.write(" " + str(i) + ":" + str(int(value)))
    f.write("\n")

f.close()
