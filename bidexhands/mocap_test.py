import bidexhands as bi
import torch
import numpy as np


env_name = "MocapShadowHandDoorOpenInward"
# env_name = "MocapShadowHandDoorOpenOutward"
# env_name = "MocapShadowHandDoorCloseInward"
# env_name = "MocapShadowHandDoorCloseOutward"

# env_name = "MocapShadowHandSwingCup"
# env_name = "MocapShadowHandScissors"
# env_name = "MocapShadowHandSwitch"
# env_name = "MocapShadowHandLiftUnderarm"

# env_name = "MocapShadowHandGraspAndPlace"
# env_name = "MocapShadowHandBlockStack"
# env_name = "MocapShadowHandPushBlock"

# env_name = "MocapShadowHandBottleCap" # not ready
# env_name = "MocapShadowHandKettle" #not easy
algo = "manual"

# algo = "ppo"
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
bridge = CvBridge()
import time
from std_msgs.msg import Float32MultiArray

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
        self.qpos_sub = rospy.Subscriber("/qpos/Right", Float32MultiArray, self.callback)
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

        self.hand_action_pub = rospy.Publisher("/action", Float32MultiArray, queue_size=1000)

    def callback(self, qpos_msg):
    
        # idx = [ 6,7,8,  10,11,12,  14,15,16, 18,19,20,21,  23,24,25,26,27]
        self.count = self.count + 1
        
        action = np.array( qpos_msg.data )

        if( self.count == 1): # initialize (x,y,z)
            self.init_pos =  action[0:3].copy()
        action[0:3] = action[0:3] - self.init_pos
        
        root_pos = action[:6].copy()
        print("\n")
        print("new iter root_pose: ", root_pos)
        action[0] = -1 * action[0]
        action[1] = -1 * action[1]

        #action[3], action[5] = action[5], action[3]

        # action[5] = -1 * action[5]
        action[4] = -1 * action[4]  
        action[3] = -1 * action[3]  
        # # door env setting
        # action[0] = -1 * action[0]
        # action[1] = -1 * action[1]
        
        # action[4] = -1 * action[4]        
        # action[0:3] = z_rot @ action[0:3]
        # action[3], action[4] = action[4], action[3]
        # action[3] = -1 * action[3]
        # action[5] = action[5] + np.pi/2
        # print("action[5]: ", action[5]/np.pi )

        # if action[0] < 0.0:
        #     action[0] = action[0]*3
        # action[2] = action[2]*2
        # # door env setting


        ################################################################################        
        # below are template
        ################################################################################ 
        
        action = (action - self.middle_bound_np ) / self.scale_np
        #
        # input should be scaled to -1.0 ~ 1.0
        #

        #action_right = action - action     
        action_right = action
        action_left = action_right.copy()

        action_left = action_left - action_left
        
        #action_left[0:3] = action_left[0:3] - action_left[0:3]
        
        # left hand mirror up right hand 
        # action_left[1] = -1 * action_left[1]
        # action_left[3] = -1 * action_left[3]
        # action_left[5] = -1 * action_left[5]

        

        action = np.concatenate( [action_right, action_left] , axis = 0)        
        #print("action: ", action)
        #action = self.env.action_space.sample()
        action_msg = Float32MultiArray()
        #print("published")
        action_msg.data = action
        self.hand_action_pub.publish(action_msg)



        act = torch.tensor(action).repeat((self.env.num_envs, 1))
        act = act.to(torch.float32)
        #act = act.to("cuda:0")
        #print("act: ", act)
        #print("act.dtype: ", act.dtype)
        obs, reward, done, info = self.env.step(act)
        #print("after act: ", act)
        return

    def run(self):
        rospy.spin()  

def main():
    rospy.init_node("isaac_mocap_node")
    isaac_node = isaac()
    isaac_node.run()

if __name__ == "__main__":
    main()