import bidexhands as bi
import torch
import numpy as np
env_name = 'Mocap'
algo = "manual"
# env_name = 'ShadowHandDoorOpenOutward'
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
        self.qpos_sub = rospy.Subscriber("/qpos", Float32MultiArray, self.callback)
        self.count = 0


    def callback(self, qpos_msg):
        # removed 18 13 9 5
        #idx = [0, 1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 23]
        action = self.env.action_space.sample()

        # np_pos = np.array(qpos_msg.data)
        
        # np_pos[0:2] = 0.0

        # np_pos[2] = -1.0 * np_pos[2]
        # np_pos[6] = -1.0 * np_pos[6]
        # np_pos[22] = -1.0 * np_pos[22]
        # np_pos[23] = -1.0 * np_pos[23]

        # #print(action)
        # action[ 6:26 ] =  torch.from_numpy( np_pos[idx])
        
        # action[ 0 ] = 1.0 
        # action[ 3 ] = 1.0 
        # #self.count += 1
        # #action[0] = ( self.count//30 ) % 5 * 0.2
        # # action = self.env.action_space.sample() * 0.0
        # # action = torch.from_numpy( np.array(qpos_msg.data))
        act = torch.tensor(action).repeat((self.env.num_envs, 1))
        obs, reward, done, info = self.env.step(act)
        return

    def run(self):
        rospy.spin()  

def main():
    rospy.init_node("isaac_node")
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