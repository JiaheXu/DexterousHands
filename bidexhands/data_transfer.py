import bidexhands as bi
import torch
import numpy as np
import glob, os, sys, argparse
import rosbag
import rospy
# easy
# env_name = "MocapShadowHandDoorCloseInward" # right view
# env_name = "MocapShadowHandDoorCloseOutward" # right view
# env_name = "MocapShadowHandDoorOpenInward" # right view
env_name = "MocapShadowHandDoorOpenOutward" # right view

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

z_rot = np.array([
    [0.0, -1.0, 0.0],
    [1.0,  0.0, 0.0],
    [0.0,  0.0, 1.0]
])

x_rot = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, 1.0, 0.0]
]) 

y_rot = np.array([
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0]
]) 

class isaac():
    def __init__(self):

        self.env = bi.make(env_name, algo)
        self.obs = self.env.reset()
        self.terminated = False

        self.count = 0
        self.lower_bound_np = np.array([
            -5.0, -5.0, -5.0, -3.14159, -3.14159, -3.14159,

            -0.3490,  0.0000,  0.0000,  0.0000,
            -0.3490,  0.0000,  0.0000,  0.0000,
            -0.3490,  0.0000,  0.0000,  0.0000,
            0.0000, -0.3490,  0.0000,  0.0000, 0.0000,
            -1.0470,  0.0000, -0.2090, -0.5240, -1.5710])

        self.upper_bound_np = np.array([
            5.0, 5.0, 5.0, 3.14159, 3.14159, 3.14159,
            
            0.3490, 1.5710, 1.5710, 1.5710,
            0.3490, 1.5710, 1.5710, 1.5710, 
            0.3490, 1.5710, 1.5710, 1.5710,
            0.7850, 0.3490, 1.5710, 1.5710, 1.5710,
            1.0470, 1.2220, 0.2090, 0.5240, 0.0000])

        self.middle_bound_np = ( self.upper_bound_np + self.lower_bound_np ) / 2.0
        
        self.scale_np = ( self.upper_bound_np - self.lower_bound_np ) / 2.0
        self.count = 0
        self.init_pos = np.array([0.0, 0.0, 0.0])

    def step(self, action_msg):
    
        self.count = self.count + 1
        if(self.count % 10 == 0):
            print("current count: ", self.count)
        
        action = np.array( action_msg.position )

        act = torch.tensor(action).repeat((self.env.num_envs, 1))
        act = act.to(torch.float32)

        obs, reward, _, info = self.env.step(act)
        done = False
        if(info["successes"][0] == 1):
            print("successes !!!")
            print("successes !!!")
            print("successes !!!")
            self.count = 0
            done = True
            self.env.reset()

        if(info["reset"][0] == 1):
            print("reset !!!")
            print("reset !!!")
            print("reset !!!")
            self.env.reset()
        return obs, done

rospy.init_node("isaac_playback_node")
isaac_node = isaac()
obs_all = []
action_all = []

def make_npy_files(dataset_directory, file):

    bagIn = rosbag.Bag(file, "r")
    count = 0

    isaac_node.env.reset()
    obs_buffer = []
    action_buffer = []
    
    print("runing: ", file, " !!!!!!!!!!")
    print("runing: ", file, " !!!!!!!!!!")
    print("runing: ", file, " !!!!!!!!!!")

    for topic, msg, t in bagIn.read_messages(topics=["/action"]):
        count = count +1

        obs, done = isaac_node.step(msg)

        # if( count < 30):
        #     continue        
        action_buffer.append(msg.position)
        obs = obs.cpu().detach().numpy()
        obs = np.squeeze(obs)
        obs_buffer.append( obs )
        
        if(done):
            count = 0
            break
    
    print("file: ", file)
    print("count: ", count)
    print("")

    if(len(action_buffer) != len(obs_buffer)):
        print("msg number error on ", file)

    obs_buffer = np.array(obs_buffer).copy()
    action_buffer= np.array(action_buffer).copy()
    print("obs_buffer: ", obs_buffer.shape)
    print("action_buffer: ", action_buffer.shape)
    demo_data={}
    demo_data["observation"] = obs_buffer
    demo_data["action"] = action_buffer

    np.save( file[0:-4] + ".npy", demo_data)

    bagIn.close()

def main():


    dataset_directory = "../data/" + env_name
    file_path = os.path.join( dataset_directory, '08.bag')
    make_npy_files(dataset_directory , file_path)



if __name__ == "__main__":
    main()
