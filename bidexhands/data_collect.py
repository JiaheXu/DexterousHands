import bidexhands as bi
import torch
import numpy as np
import glob, os, sys, argparse
import rosbag
import rospy
# easy
# env_name = "MocapShadowHandDoorCloseInward" # right view
env_name = "MocapShadowHandDoorCloseOutward" # right view
# env_name = "MocapShadowHandDoorOpenInward" # right view
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
algo = "manual"

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
bridge = CvBridge()
import time
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState

obs_all = []
action_all = []
import copy

def get_data(dataset_directory, file_pth):
    obs = np.load(file_pth, allow_pickle=True).flat[0].get("observation")
    action = np.load(file_pth, allow_pickle=True).flat[0].get("action")
    obs_all.append( copy.deepcopy(obs) )
    action_all.append( copy.deepcopy(action) )
    # print("obs: ", obs.shape)
    # print("action: ", action.shape)

def main():

    dataset_directory = "../data/" + env_name

    file_path = os.path.join( dataset_directory, '*.npy')
    filelist = sorted( glob.glob( file_path) )
    
    for file in filelist:
        get_data(dataset_directory , file)

    print("obs_all: ", len(obs_all))
    print("action_all: ", len(action_all))
    np.save(os.path.join(dataset_directory, env_name + "_obs.npy"), obs_all)
    np.save(os.path.join(dataset_directory, env_name + "_action.npy"), action_all)
    demo_data = {}
    demo_data["actions"] = action_all
    demo_data["observations"] = obs_all

    np.save(os.path.join(dataset_directory, env_name + ".npy"), demo_data)


if __name__ == "__main__":
    main()
