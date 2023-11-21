import bidexhands as bi
import torch
import numpy as np

env_name = 'Mocap'
#env_name = 'ShadowHandDoorOpenInward'
algo = "manual"

# algo = "ppo"
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
bridge = CvBridge()
import time
from std_msgs.msg import Float32MultiArray



class isaac():
    def __init__(self):


        self.env = bi.make(env_name, algo)
        self.obs = self.env.reset()
        self.terminated = False
        self.qpos_sub = rospy.Subscriber("/qpos/Right", Float32MultiArray, self.callback)
        self.count = 0


    def callback(self, qpos_msg):
        # removed 18 13 9 5
        idx = [ 6,7,8,  10,11,12,  14,15,16, 18,19,20,21,  23,24,25,26,27]
        qpos = np.array( qpos_msg.data )
        qpos = qpos[idx]
        zeros = np.zeros((8,))
        action_right = np.concatenate( [zeros, qpos] , axis = 0)
        # action_right = np.array([   
        #     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        #     0.00774, 1.57079, 1.23738,  -0.28021, 1.57099, 1.40754,  -0.31893, 1.57098, 1.54871,
        #     0.32565, -0.34900, 1.25364, 1.39683,
        #     0.01461, 0.01744, -0.20467, 0.01835, -0.04081 ])
        #action_right = action_right * 0.0
        
        #action_right[24] = ( (self.count % 1000) // 30) * 0.1 - 1.5
        #print("action_right: ", action_right)
        action_left = action_right.copy()
        
        # action = self.env.action_space.sample()
        self.count += 1
        print("self.count: ", self.count)
        #print("action.shape: ", action.shape)
        # print("got a pos msg")
        # action_right =  list(qpos_msg.data) #28 dim 6 + 24 - 2
        # action_right = np.array(action_right)
        # action_left = np.zeros((28,))
        # action = np.concatenate( [action_right, action_left] , axis = 0)
        action = np.concatenate( [action_right, action_left] , axis = 0)
        act = torch.tensor(action).repeat((self.env.num_envs, 1))
        act = act.to(torch.float32)
        obs, reward, done, info = self.env.step(act)
        return

    def run(self):
        rospy.spin()  

def main():
    rospy.init_node("isaac_mocap_node")
    isaac_node = isaac()
    isaac_node.run()

if __name__ == "__main__":
    main()    
# while not terminated:
#     action = env.action_space.sample()* 0.0
#     act = torch.tensor(action).repeat((env.num_envs, 1))
#     obs, reward, done, info = env.step(act)

# while not terminated:
#     action = env.action_space.sample()* 0.0
#     act = torch.tensor(action).repeat((env.num_envs, 1))
#     obs, reward, done, info = env.step(act)