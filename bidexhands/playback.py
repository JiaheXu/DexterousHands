import bidexhands as bi
import torch
import numpy as np
import random
from isaacgym import gymapi


# env_name = "ShadowHandDoorCloseInward" # right view
# env_name = "ShadowHandDoorCloseOutward" # right view
# env_name = "ShadowHandDoorOpenInward" # right view
# env_name = "ShadowHandDoorOpenOutward" # right view

# easy
env_name = "MocapShadowHandDoorCloseInward" # right view
# env_name = "MocapShadowHandDoorCloseOutward" # right view

# env_name = "MocapShadowHandDoorOpenInward" # right view
# env_name = "MocapShadowHandDoorOpenOutward" # right view

# env_name = "MocapShadowHandLiftUnderarm" # right view
# env_name = "MocapShadowHandSwitch" # front view
# env_name = "MocapShadowHandSwingCup" # right view

# medium
# env_name = "MocapShadowHandBlockStack" # left & right view left and right hand?
# env_name = "MocapShadowHandPushBlock" # left & right view left and right hand?

# hard
# env_name = "MocapShadowHandGraspAndPlace" # left & right view left and right hand?
# env_name = "MocapShadowHandScissors"# not easy need two hands front view
# env_name = "MocapShadowHandPen"# need two hands need two hands front view
# env_name = "MocapShadowHandKettle"# need two hands # left & right view left and right hand?

algo = "manual"

# algo = "ppo"
import rospy
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
        self.action_sub = rospy.Subscriber("/action", JointState, self.callback)

        self.color_pub = rospy.Publisher("/hand_color", JointState, queue_size=1)

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

        #self.lower_bound = torch.from_numpy(self.lower_bound_np, dtype = torch.float32)
        #self.upper_bound = torch.from_numpy(self.upper_bound_np, dtype = torch.float32)
        self.middle_bound_np = ( self.upper_bound_np + self.lower_bound_np ) / 2.0
        
        self.scale_np = ( self.upper_bound_np - self.lower_bound_np ) / 2.0
        self.count = 0
        self.init_pos = np.array([0.0, 0.0, 0.0])
        self.last_action = None

    def callback(self, action_msg):
        
        if(self.count == 0 ):
            self.env.reset()

        self.count = self.count + 1
        
        print("current count: ", self.count)

        action = np.array( action_msg.position )

        act = torch.tensor(action).repeat((self.env.num_envs, 1))
        act = act.to(torch.float32)

        obs, reward, done, info = self.env.step(act)
        if(info["successes"][0] == 1):
            print("successes !!!")
            print("successes !!!")
            print("successes !!!")
            self.count = 0

        if(info["reset"][0] == 1):
            print("reset !!!")
            print("reset !!!")
            print("reset !!!")
        return
    def run(self):
        rospy.spin()  

def main():
    rospy.init_node("isaac_playback_node")
    isaac_node = isaac()
    isaac_node.run()

if __name__ == "__main__":
    main()
