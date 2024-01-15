import torch
import numpy as np
import glob, os, sys, argparse
import rosbag
import rospy
import matplotlib.pyplot as plt
# easy
# env_name = "MocapShadowHandDoorCloseInward" # right view
# env_name = "MocapShadowHandDoorCloseOutward" # right view
env_name = "MocapShadowHandDoorOpenInward" # right view
# env_name = "MocapShadowHandDoorOpenOutward" # right view

# env_name = "MocapShadowHandSwingCup" # right view
# env_name = "MocapShadowHandLiftUnderarm" # right view
# env_name = "MocapShadowHandSwitch" # front view

# medium
# env_name = "MocapShadowHandPushBlock" # left & right view left and right hand?
# env_name = "MocapShadowHandBlockStack" # left & right view left and right hand?
# env_name = "MocapShadowHandGraspAndPlace" # left & right view left and right hand?

# hard
# env_name = "MocapShadowHandScissors"# not easy need two hands front view
# env_name = "MocapShadowHandPen"# need two hands need two hands front view
# env_name = "MocapShadowHandKettle"# need two hands # left & right view left and right hand?

# env_name = "MocapShadowHandPen"

def main():

    dataset_directory = env_name + "/" + env_name
    obs_directory = dataset_directory + "_obs.npy"
    action_directory = dataset_directory + "_action.npy"

    txt_directory = env_name+ "/" + "segment.txt"

    obs = np.load(obs_directory, allow_pickle=True)
    action = np.load(action_directory, allow_pickle=True)

    print("obs shape: ", obs.shape)
    # print("obs[0].shape = ", obs[0].shape)

    line_array = []
    with open(txt_directory, 'r') as file:
        for line in file:
            elements = line.strip().split()
            elements = list(map(int, elements))    
            line_array.append(elements)
    # print("line_array: ", line_array)

    # for each observation in obs 
    for i in range(len(obs)):
        data = obs[i]
        # print("data.shape: ", data.shape)
        index_dict = {}
        # for each observation data, consider right hand related data

        # segments according to some sepcial indexes, for example holding: 0-5 (right hand base xyz, rpy) 6-27(hand joint dof) + right handle position

        for j in range(len(data)):
            obs_data = data[:, j]
            # plt.plot(obs_data, label=f'Observation {j}')
            # plt.show()
            # first derivative using finite differences
            first_derivative = np.gradient(obs_data)
            sorted_indices = np.argsort(np.abs(first_derivative))
            indices_close_to_zero = sorted_indices[: 10]
            for index in indices_close_to_zero:
                if index in index_dict:
                    index_dict[index] += 1
                else:
                    index_dict[index] = 1
        sorted_dict = dict(sorted(index_dict.items(), key=lambda item: item[1], reverse=True))
        print(f"obs[{i+1}], sorted_dict: ", list(sorted_dict.keys())[ :10])



        # change outputs to a file, any file npy txt pkl all are good.
        for tocompare_index in line_array[i]:
            # print("tocompare_index: ", tocompare_index)
            print(f"obs[{i+1}], {tocompare_index}: {is_in_dict(index_dict,tocompare_index)} ")

def is_in_dict(dict, key):
    if key in dict:
        return True
    else:
        return False

    


if __name__ == "__main__":
    main()
