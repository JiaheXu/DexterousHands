import numpy as np

data= np.load("MocapShadowHandDoorOpenInward.npy", allow_pickle=True)
print( data[0] )
#for i in range(len(data)):
#    print(data[i].shape)
