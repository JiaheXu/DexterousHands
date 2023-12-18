
import torch
import numpy as np
import glob, os, sys, argparse
import rosbag
import rospy
# easy
# env_name = "MocapShadowHandDoorOpenInward" # right view
# env_name = "MocapShadowHandDoorOpenOutward" # right view
# env_name = "MocapShadowHandDoorCloseInward" # right view
env_name = "MocapShadowHandDoorCloseOutward" # right view
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

def main():
    

    #
    # parser.add_argument("-b", "--bags_dir", help="Input ROS bag name.")
    # dataset_directory = args.bags_dir
    
    # dataset_directory = "../data/test/"

    dataset_directory = "../data/" + env_name

    file_path = os.path.join( dataset_directory, '*.bag')
    filelist = sorted( glob.glob( file_path) )
    for i in range( len(filelist) ): 
        old_name = filelist[i]
        new_name = os.path.join(dataset_directory , "{:02d}".format(i+1) + ".bag")
        print("new_name: ", new_name)
        print("old_name: ", old_name)
        os.rename(old_name, new_name)


if __name__ == "__main__":
    
    main()
