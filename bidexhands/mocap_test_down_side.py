import bidexhands as bi
import torch
import numpy as np

env_name = "MocapShadowHandDoorOpenInward"


# env_name = "MocapShadowHandDoorOpenOutward"
# env_name = "MocapShadowHandDoorCloseInward"
# env_name = "MocapShadowHandDoorCloseOutward"
# env_name = "MocapShadowHandSwingCup"
# env_name = "MocapShadowHandLiftUnderarm"
# env_name = "MocapShadowHandPushBlock"
# env_name = "MocapShadowHandBlockStack"



# env_name = "MocapShadowHandGraspAndPlace"
# env_name = "MocapShadowHandScissors"# not easy

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
from scipy.spatial.transform import Rotation  

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

def rot2eul( R , alpha0 = 0.0, beta0 = 0.0, gamma0 = 0.0):
    #print("R:", R)
    beta1 = -np.arcsin(R[2,0]) # Y
    alpha1 = np.arctan2(R[2,1]/np.cos(beta1),R[2,2]/np.cos(beta1)) # X
    gamma1 = np.arctan2(R[1,0]/np.cos(beta1),R[0,0]/np.cos(beta1)) # Z


    beta2 = np.pi + np.arcsin(R[2,0]) # Y
    alpha2 = np.arctan2(R[2,1]/np.cos(beta2),R[2,2]/np.cos(beta2)) # X
    gamma2 = np.arctan2(R[1,0]/np.cos(beta2),R[0,0]/np.cos(beta2)) # Z


    return np.array((alpha1, beta1, gamma1))
    
    if( np.fabs(alpha1 - alpha0) < np.fabs(alpha2 - alpha0) ):
        return np.array((alpha1, beta1, gamma1))
    else:
        return np.array((alpha2, beta2, gamma2))

def eul2rot(theta3, theta2, theta1):
    # theta3: X
    # theta2: Y
    # theta1: Z
    c1 = np.cos( theta1 )
    s1 = np.sin( theta1 )
    c2 = np.cos( theta2 )
    s2 = np.sin( theta2 )
    c3 = np.cos( theta3 )
    s3 = np.sin( theta3 )

    # elif order=='zyx':
    matrix=np.array([[c1*c2, c1*s2*s3-c3*s1, s1*s3+c1*c3*s2],
                         [c2*s1, c1*c3+s1*s2*s3, c3*s1*s2-c1*s3],
                         [-s2, c2*s3, c2*c3]])
    return matrix

class isaac():
    def __init__(self):

        self.env = bi.make(env_name, algo)
        self.obs = self.env.reset()
        self.terminated = False
        self.qpos_sub = rospy.Subscriber("/qpos/Right", JointState, self.callback)
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

        self.hand_action_pub = rospy.Publisher("/action", JointState, queue_size=1000)
        self.action_buffer = []
    
    def make_rosbag(self):
        now = datetime.now()
        self.bag_name = now.strftime("%m_%d_%Y_%H:%M:%S") + ".bag"        
        self.bagOut = rosbag.Bag(self.bag_name, "w")
        for msg in self.action_buffer:
            self.bagOut.write("/action", msg, msg.header.stamp)
        self.bagOut.close()

    def callback(self, qpos_msg):
    
        # idx = [ 6,7,8,  10,11,12,  14,15,16, 18,19,20,21,  23,24,25,26,27]
        self.count = self.count + 1
        print("sim env current count: ", self.count)
        
        action = np.array( qpos_msg.position )

        if( self.count == 1): # initialize (x,y,z)
            self.init_pos =  action[0:3].copy()
        action[0:3] = action[0:3] - self.init_pos
        
        root_pos = action[:6].copy()
        
        action[0] = -1 * action[0]
        action[1] = -1 * action[1]

        action[0:3] = y_rot.T @ action[0:3]

        action[3] = -1 * action[3]
        action[4] = -1 * action[4]

        rot = eul2rot(action[3], action[4], action[5])
        rot =  rot @ y_rot.T

        eul = rot2eul( rot )
        #print("eul",eul)
        action[3] = eul[0]
        action[4] = eul[1]
        action[5] = eul[2]

        # action[4] = eul[1] - np.pi/2

        action[1] = action[1]*2        
        action[2] = action[2]*2
        action[3] = action[3]*2
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

        
        #action[25] = -1*action[25]

        action = np.concatenate( [action_right, action_left] , axis = 0)        

        #action = self.env.action_space.sample()
        action_msg = JointState()
        action_msg.position = action
        action_msg.header = qpos_msg.header

        self.hand_action_pub.publish(action_msg)

        act = torch.tensor(action).repeat((self.env.num_envs, 1))
        act = act.to(torch.float32)
        obs, reward, done, info = self.env.step(act)

        self.action_buffer.append(action_msg)

        if(info["successes"][0] == 1):
            self.make_rosbag()

        return

    def run(self):
        rospy.spin()  

def main():
    
    # rot = eul2rot(1.57, 1 ,-1)
    # print("rot:", rot)

    # eul = rot2eul(rot)
    
    # print("eul: ", eul)
    
    # print("")
    
    # print("rot2:", eul2rot(eul[0], eul[1], eul[2]))
    
    # print("")
    # print("diff: ", rot - eul2rot(eul[0], eul[1], eul[2]))
    rospy.init_node("isaac_mocap_node")
    isaac_node = isaac()
    isaac_node.run()

if __name__ == "__main__":
    main()