import bidexhands as bi
import torch
import numpy as np
import copy

# env_name = "MocapShadowHandDoorCloseOutward" # right view
# env_name = "MocapShadowHandDoorOpenInward" # right view
# env_name = "MocapShadowHandGraspAndPlace"

env_name = "MocapShadowHandMicrowave"

# env_name = "MocapShadowHandPushBlock" # left & right view left and right hand?
# env_name = "MocapShadowHandPen"# need two hands need two hands front view
# env_name = "MocapShadowHandKettle"# need two hands # left & right view left and right hand?

# easy
# env_name = "MocapShadowHandDoorCloseInward" # right view
# env_name = "MocapShadowHandDoorCloseOutward" # right view
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

from datetime import datetime
import rosbag

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
        self.qpos_sub = rospy.Subscriber("/qpos/Right", JointState, self.callback)
        self.count = 0
        self.lower_bound_np = np.array([
            -5.0, -5.0, -5.0, -2*3.14159, -2*3.14159, -2*3.14159,

            -0.3490,  0.0000,  0.0000,  0.0000,
            -0.3490,  0.0000,  0.0000,  0.0000,
            -0.3490,  0.0000,  0.0000,  0.0000,
            0.0000, -0.3490,  0.0000,  0.0000, 0.0000,
            -1.0470,  0.0000, -0.2090, -0.5240, -1.5710])

        self.upper_bound_np = np.array([
            5.0, 5.0, 5.0, 2*3.14159, 2*3.14159, 2*3.14159,
            
            0.3490, 1.5710, 1.5710, 1.5710,
            0.3490, 1.5710, 1.5710, 1.5710, 
            0.3490, 1.5710, 1.5710, 1.5710,
            0.7850, 0.3490, 1.5710, 1.5710, 1.5710,
            1.0470, 1.2220, 0.2090, 0.5240, 0.0000])

        #self.lower_bound = torch.from_numpy(self.lower_bound_np, dtype = torch.float32)
        #self.upper_bound = torch.from_numpy(self.upper_bound_np, dtype = torch.float32)
        self.middle_bound_np = ( self.upper_bound_np + self.lower_bound_np ) / 2.0
        
        self.scale_np = ( self.upper_bound_np - self.lower_bound_np ) / 2.0
        self.count = -2
        self.init_pos = np.array([0.0, 0.0, 0.0])

        self.hand_action_pub = rospy.Publisher("/action", JointState, queue_size=1000)
        self.action_buffer = []
        self.last_orient = None
    
    def make_rosbag(self):
        if( len(self.action_buffer) < 100 ):
            self.action_buffer.clear()
            return

        now = datetime.now()
        self.bag_name = now.strftime("%m_%d_%Y_%H:%M:%S") + ".bag"        
        self.bagOut = rosbag.Bag(self.bag_name, "w")
        for msg in self.action_buffer:
            self.bagOut.write("/action", msg, msg.header.stamp)
        self.bagOut.close()
        self.action_buffer.clear()

    def callback(self, qpos_msg):
    
        # idx = [ 6,7,8,  10,11,12,  14,15,16, 18,19,20,21,  23,24,25,26,27]
        self.count = self.count + 1
        print("sim env current count: ", self.count)
        
        action = np.array( qpos_msg.position )

        if( self.count == -1): # initialize (x,y,z)
            self.init_pos =  action[0:3].copy()
        action[0:3] = action[0:3] - self.init_pos
        
        if(self.count < 60):
            return

        action[3], action[5] = action[5], action[3]
        action[0] = action[0]*2
        action[1] = action[1]*2        
        action[2] = action[2]*2  

        action_np = copy.deepcopy(action)
        # if(self.last_orient is not None):
        #     d_raw = action[3] - self.last_orient[0]
        #     d_pitch = action[4] - self.last_orient[1]
        #     d_yaw = action[5] - self.last_orient[2]

        #     if(d_raw > 4.0):
        #         d_raw = d_raw - 2*np.pi 
        #         print("d_raw1: ", d_raw)
        #         print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")  
        #     if(d_pitch > 4.0):
        #         d_pitch = d_pitch - 2*np.pi 
        #         print("d_pitch1: ", d_pitch)
        #         print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")  
        #     if(d_yaw > 4.0):
        #         d_yaw = d_yaw - 2*np.pi 
        #         print("d_yaw1: ", d_yaw)
        #         print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")  



        #     if(d_raw < -4.0):
        #         d_raw = d_raw + 2*np.pi 
        #         print("d_raw1: ", d_raw)
        #         print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")  
        #     if(d_pitch < -4.0):
        #         d_pitch = d_pitch + 2*np.pi 
        #         print("d_pitch1: ", d_pitch)
        #         print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!") 
        #     if(d_yaw < -4.0):
        #         d_yaw = d_yaw + 2*np.pi 
        #         print("d_yaw1: ", d_yaw)
        #         print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")   

        #     print("d_raw: ", d_raw)
        #     print("d_pitch: ", d_pitch)
        #     print("d_yaw: ", d_yaw)

        #     action[3] = self.last_action[3] + d_raw 
        #     action[4] = self.last_action[4] + d_pitch 
        #     action[5] = self.last_action[5] + d_yaw   

            # while(action[3] > 2 * 3.1415927410125732):
            #     action[3] -= 2 * 3.1415927410125732

            # while(action[4] > 2 * 3.1415927410125732):
            #     action[4] -= 2 * 3.1415927410125732
            
            # while(action[5] > 2 * 3.1415927410125732):
            #     action[5] -= 2 * 3.1415927410125732

            # while(action[3] < -2 * 3.1415927410125732):
            #     action[3] += 2 * 3.1415927410125732

            # while(action[4] < -2 * 3.1415927410125732):
            #     action[4] += 2 * 3.1415927410125732
            
            # while(action[5] < -2 * 3.1415927410125732):
            #     action[5] += 2 * 3.1415927410125732
            # print("last_action, d_yaw:", self.last_action[5] , " ", d_yaw)
        print("final action[3:6]: ", action[3:6])
        self.last_orient = copy.deepcopy(action_np[3:6])
        self.last_action = copy.deepcopy(action)
        ################################################################################        
        # below are template
        ################################################################################ 
        
        action = (action - self.middle_bound_np ) / self.scale_np
        
        #
        # input should be scaled to -1.0 ~ 1.0
        #

        action_right = action        
        action_left = action_right.copy()
        #action_left[0:3] = action_left[0:3] - action_left[0:3]
        
        # left hand mirror up right hand 
        action_left = action_left - action_left
        
        
        # mirring
        # action_left[1] = -1 * action_left[1]
        # action_left[3] = -1 * action_left[3]
        # action_left[5] = -1 * action_left[5]


        action = np.concatenate( [action_right, action_left] , axis = 0)        
        action_msg = JointState()
        action_msg.position = action
        action_msg.header = qpos_msg.header

        self.hand_action_pub.publish(action_msg)

        act = torch.tensor(action).repeat((self.env.num_envs, 1))
        act = act.to(torch.float32)
        obs, reward, done, info = self.env.step(act)

        self.action_buffer.append(action_msg)

        if(info["successes"][0] == 1):
            print("successes !!!")
            print("successes !!!")
            print("successes !!!")
            self.make_rosbag()
        if(info["reset"][0] == 1):
            self.action_buffer.clear()
            self.count = 0
        return

    def run(self):
        rospy.spin()  

def main():
    rospy.init_node("isaac_mocap_node")
    isaac_node = isaac()
    isaac_node.run()

if __name__ == "__main__":
    main()
