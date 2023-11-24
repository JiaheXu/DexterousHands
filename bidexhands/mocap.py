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


    def callback(self, qpos_msg):
    
        # idx = [ 6,7,8,  10,11,12,  14,15,16, 18,19,20,21,  23,24,25,26,27]
        self.count = self.count + 1

        # if(self.count % 5 != 0 ): 
        #     return
        
        qpos = np.array( qpos_msg.data )
        
        # qpos = [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0495,  1.5710,
        #   1.5710,  1.1574, -0.1480,  1.5710,  1.5710,  1.0553, -0.2646,  1.5710,
        #   1.5710,  1.1953,  0.0624, -0.2138,  1.5710,  1.5587,  1.3762,  0.3513,
        #   0.3093,  0.1722,  0.3505, -0.2404]
        # qpos = [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000, 
        #     -0.349,   0.9305,  1.571,   1.571,
        #     -0.3489,  1.2059,  1.571,   1.5513,
        #     -0.349,   0.9256,  1.571,   1.571,
        #     7.8775e-05, -3.4897e-01,  3.4559e-01,  1.5710e+00,  1.5710e+00,
        #     0.0129,  0.3373,  0.1,     0.4326, -0.4756]
        # qpos = [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
        #   0.0,  0.0,  0.0,
        #   0.0,  0.0,  0.0,  
        #   0.0,  0.0,  0.0,  
        #   0.0,  0.0,  0.0,  
        #   0.0,  0.0,  0.0,  0.0,  0.0,
        #   0.3513, 0.3093,  0.1722,  0.3505, -0.2404]
        qpos = [ 0.0000,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000,        
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0]
        qpos = np.array( qpos )
        
        
        root_pos = qpos[:6].copy()
        print("\n")
        print("new iter root_pose: ", root_pos)
        #qpos = qpos[6:]
        #qpos[0:6] = 0.0
        # print("qpos.shape: ", qpos.shape)
        #qpos = qpos[idx]
        
        #zeros = np.zeros((6,))
        
        #qpos = np.concatenate( [zeros, qpos] , axis = 0)
        qpos = (qpos - self.middle_bound_np ) / self.scale_np
        #print("qpos", qpos)
        #action_right = np.concatenate( [root_pos, qpos] , axis = 0)
        
        action_right = qpos
        action_left = action_right.copy()
        action_left[1] = 0.0
        action = np.concatenate( [action_right, action_left] , axis = 0)        
        print("action: ", action)
        #action = self.env.action_space.sample()

        self.count += 1
        print("self.count: ", self.count)

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