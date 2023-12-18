import numpy as np

data= np.load("action.npy", allow_pickle=True)
for i in range(len(data)):
    print(data[i].shape)
