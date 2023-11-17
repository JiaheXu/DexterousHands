import bidexhands as bi
import torch
import numpy as np
env_name = 'Manual'
algo = "manual"
#env_name = 'ShadowHandDoorOpenOutward'
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

    def callback(self, qpos_msg):
        action = self.env.action_space.sample() * 0.0
        act = torch.tensor(action).repeat((self.env.num_envs, 1))
        obs, reward, done, info = self.env.step(act)
        return

    def run(self):
        rospy.spin()  

def main():
    rospy.init_node("ShadowHandDoorOpenOutward_manual")
    isaac_node = isaac()
    isaac_node.run()

if __name__ == "__main__":
    main()    
