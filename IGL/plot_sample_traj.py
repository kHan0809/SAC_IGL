import pickle
import os
import numpy as np
import math
import quaternion
import matplotlib.pyplot as plt

task_name = "Pick"
data_concat = []

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for pickle_data in os.listdir(os.getcwd()+'/../Demo/Demo_data'):
    if task_name in pickle_data:
        with open(os.getcwd()+'/../Demo/Demo_data/' + pickle_data, 'rb') as f:
            data = pickle.load(f)
            data_concat.extend(data)

id = 5
x,y,z = zip(*data_concat[id]["observation"][:,:3])
ax.scatter(x,y,z,color='m')
ax.set_xlabel('XX')
ax.set_ylabel('YY')
ax.set_zlabel('ZZ')
x,y,z = zip(*data_concat[id]["observation"][:,3:6])
ax.scatter(x,y,z,color='c')
x,y,z = zip(*data_concat[id]["observation"][:,-3:])
ax.scatter(x,y,z,color='r')

plt.show()