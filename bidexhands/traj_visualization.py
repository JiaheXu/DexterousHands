
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
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion
import std_msgs

from mpl_toolkits import mplot3d
# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

def main():
    #
    # parser.add_argument("-b", "--bags_dir", help="Input ROS bag name.")
    # dataset_directory = args.bags_dir
    
    # dataset_directory = "../data/test"

    dataset_directory = "../data/" + env_name

    file_path = os.path.join(dataset_directory, env_name + ".npy")
    datapoints = np.load(file_path, allow_pickle=True).flat[0].get("observations")
    
    rospy.init_node('talker', anonymous=True)

    rate = rospy.Rate(30) # 10hz

    obj1_pub = rospy.Publisher('obj1', Odometry, queue_size=10)
    obj2_pub = rospy.Publisher('obj2', Odometry, queue_size=10)

    hand1_pub = rospy.Publisher('right_hand', Odometry, queue_size=10)
    hand2_pub = rospy.Publisher('left_hand', Odometry, queue_size=10)

    
    for i in range( 3 ):
        
        length = datapoints[i].shape[0]
        print("length: ", length)
        
        for j in range(length):

            header = std_msgs.msg.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = 'map'
            
            obj1_msg =  Odometry()
            obj2_msg =  Odometry()
            hand1_msg =  Odometry()
            hand2_msg =  Odometry()

            obj1_msg.header = header
            obj2_msg.header = header
            hand1_msg.header = header
            hand2_msg.header = header

            obj1_msg.pose.pose.position = Point( datapoints[i][j][439], datapoints[i][j][440], datapoints[i][j][441])
            obj2_msg.pose.pose.position = Point( datapoints[i][j][442], datapoints[i][j][443], datapoints[i][j][444])

            hand1_msg.pose.pose.position = Point( datapoints[i][j][179], datapoints[i][j][180], datapoints[i][j][181])
            hand2_msg.pose.pose.position = Point( datapoints[i][j][392], datapoints[i][j][393], datapoints[i][j][394])


            obj1_pub.publish(obj1_msg)
            obj2_pub.publish(obj2_msg)
            hand1_pub.publish(hand1_msg)
            hand2_pub.publish(hand2_msg)

            rate.sleep()
            print(j, " / ", length)


if __name__ == "__main__":
    main()
