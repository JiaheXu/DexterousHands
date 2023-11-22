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
    
        # idx = [ 6,7,8,  10,11,12,  14,15,16, 18,19,20,21,  23,24,25,26,27]
        
        # qpos = np.array( qpos_msg.data )
        qpos = [0.3410,  1.0000,  1.0000,  1.0000,  0.0446,  1.0000,
          1.0000,  1.0000, -0.1041,  1.0000,  1.0000,  1.0000,  0.0020, -0.2012,
          1.0000,  1.0000,  1.0000,  0.6530,  0.1115, -0.1736,  0.0265, -0.0133
        ]
        qpos = np.array( qpos )
        # qpos[0:6] = 0.0
        # print("qpos.shape: ", qpos.shape)
        #qpos = qpos[idx]
        zeros = np.zeros((6,))
        qpos = np.concatenate( [zeros, qpos] , axis = 0)
        
        action_right = qpos
        action_left = action_right.copy()
        action = np.concatenate( [action_right, action_left] , axis = 0)        
        
        #action = self.env.action_space.sample()

        self.count += 1
        print("self.count: ", self.count)

        act = torch.tensor(action).repeat((self.env.num_envs, 1))
        act = act.to(torch.float32)
        print("act: ", act)
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