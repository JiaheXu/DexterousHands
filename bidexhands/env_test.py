import bidexhands as bi
import torch
import numpy as np

#env_name = 'Mocap'
#env_name = 'ShadowHandDoorCloseInward'
env_name = 'ShadowHandDoorOpenInward'
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


    def callback(self, qpos_msg):
    
        # idx = [ 6,7,8,  10,11,12,  14,15,16, 18,19,20,21,  23,24,25,26,27]
        self.count = self.count + 1
        
        action = self.env.action_space.sample()



        act = torch.tensor(action).repeat((self.env.num_envs, 1))
        act = act.to(torch.float32)
        
        obs, reward, done, info = self.env.step(act)
        #print("after act: ", act)
        return

    def run(self):
        rospy.spin()  

def main():
    rospy.init_node("isaac_test_node")
    isaac_node = isaac()
    isaac_node.run()

if __name__ == "__main__":
    main()