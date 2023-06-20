import matplotlib.pyplot as plt
import numpy as np
import pickle

def LoadData(name_of_pickle: str):
 pickle_file = open(name_of_pickle, 'rb')  # Mode: read + binary
 data_base = pickle.load(pickle_file)
 pickle_file.close()
 return data_base

z = 2.1375687317341973e-16
d = LoadData("/Users/marcusallen/Documents/GitHub/momentGW/examples/D_result")
grid = LoadData("/Users/marcusallen/Documents/GitHub/momentGW/examples/Quad_pointsCC")
p_points = np.arange(0,5,100)


def function(z,p,d):
 return d*(np.sin(z*p)/z)*np.exp(d*p)

print(function(z,p_points,d))
print(grid)
print(d)