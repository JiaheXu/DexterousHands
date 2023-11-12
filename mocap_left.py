"""
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.

Spherical Joint
------------
- Demonstrates usage of spherical joints.
"""

import math
import numpy as np
from isaacgym import gymapi, gymutil

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
bridge = CvBridge()
import time
from std_msgs.msg import Float32MultiArray

def clamp(x, min_value, max_value):
    return max(min(x, max_value), min_value)

# simple asset descriptor for selecting from a list


class AssetDesc:
    def __init__(self, file_name, flip_visual_attachments=False):
        self.file_name = file_name
        self.flip_visual_attachments = flip_visual_attachments


asset_descriptors = [
    # AssetDesc("urdf/spherical_joint.urdf", False),
    # AssetDesc("mjcf/spherical_joint.xml", False),
    # AssetDesc("mjcf/open_ai_assets/hand/shadow_hand.xml", False),  
    # AssetDesc("urdf/shadow_hand_description/shadowhand_with_fingertips.urdf", False),  # okay to use
    # AssetDesc("mjcf/open_ai_assets/hand/shadow_hand_only.xml", False)
    AssetDesc("mjcf/open_ai_assets/hand/shadow_left_full.xml", False), 
]




def random_quaternion():
    """Random quaternion of the form (x, y, z, w).

    Returns:
        np.ndarray: 4-element array.
    """
    r1, r2, r3 = np.random.random(3)

    q1 = math.sqrt(1.0 - r1) * (math.sin(2 * math.pi * r2))
    q2 = math.sqrt(1.0 - r1) * (math.cos(2 * math.pi * r2))
    q3 = math.sqrt(r1) * (math.sin(2 * math.pi * r3))
    q4 = math.sqrt(r1) * (math.cos(2 * math.pi * r3))

    quat_xyzw = np.array([q2, q3, q4, q1])

    if quat_xyzw[-1] < 0:
        quat_xyzw = -quat_xyzw

    return quat_xyzw


def quat2expcoord(q):
    """Converts quaternion to exponential coordinates.

    Args:
        q (np.ndarray): Quaternion as a 4-element array of the form [x, y, z, w].

    Returns:
        np.ndarray: Exponential coordinate as 3-element array.
    """
    if (q[-1] < 0):
        q = -q

    theta = 2. * math.atan2(np.linalg.norm(q[:-1]), q[-1])
    w = (1. / np.sin(theta/2.0)) * q[:-1]

    return w * theta

class isaac():
    def __init__(self):
        # initialize gym
        self.gym = gymapi.acquire_gym()
        # configure sim
        self.sim_params = gymapi.SimParams()
        self.sim_params.dt = 1.0 / 30.0
        self.sim_params.gravity = gymapi.Vec3(0, 0, 0)
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        # parse arguments
        self.args = gymutil.parse_arguments(
            description="Shadwhand: Show example of controlling a shadow hand robot.",  
        )

        if self.args.physics_engine == gymapi.SIM_FLEX:
            pass
        elif self.args.physics_engine == gymapi.SIM_PHYSX:
            self.sim_params.physx.solver_type = 1
            self.sim_params.physx.num_position_iterations = 6
            self.sim_params.physx.num_velocity_iterations = 0
            self.sim_params.physx.num_threads = self.args.num_threads
            self.sim_params.physx.use_gpu = self.args.use_gpu

        self.sim_params.use_gpu_pipeline = False
        if self.args.use_gpu_pipeline:
            print("WARNING: Forcing CPU pipeline.")

        self.sim = self.gym.create_sim(self.args.compute_device_id, self.args.graphics_device_id, self.args.physics_engine, self.sim_params)
        
        if self.sim is None:
            print("*** Failed to create sim")
            quit()

        # create viewer
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            print("*** Failed to create viewer")
            quit()

        # load asset
        self.asset_root = "./assets"
        self.asset_file = asset_descriptors[0].file_name

        self.asset_options = gymapi.AssetOptions()
        self.asset_options.fix_base_link = True
        self.asset_options.flip_visual_attachments = asset_descriptors[0].flip_visual_attachments
        self.asset_options.use_mesh_materials = True

        print("Loading asset '%s' from '%s'" % (self.asset_file, self.asset_root))
        self.asset = self.gym.load_asset(self.sim, self.asset_root, self.asset_file, self.asset_options)

        # get array of DOF names
        self.dof_names = self.gym.get_asset_dof_names(self.asset)
        print("dof: ", len(self.dof_names))
        print(self.dof_names)
        # get array of DOF properties
        self.dof_props = self.gym.get_asset_dof_properties(self.asset)

        # create an array of DOF states that will be used to update the actors
        self.num_dofs = self.gym.get_asset_dof_count(self.asset)
        print("num_dofs: ", self.num_dofs)
        self.dof_states = np.zeros(self.num_dofs, dtype=gymapi.DofState.dtype)

        # get list of DOF types
        self.dof_types = [self.gym.get_asset_dof_type(self.asset, i) for i in range(self.num_dofs)]
        print("dof_types: \n", self.dof_types)
        # get the position slice of the DOF state array
        self.dof_positions = self.dof_states['pos']
        print("default pos: ", self.dof_positions)
        # get the limit-related slices of the DOF properties array

        # print("dof_props: \n", dof_props)
        self.stiffnesses = self.dof_props['stiffness']
        self.dampings = self.dof_props['damping']
        self.armatures = self.dof_props['armature']
        self.has_limits = self.dof_props['hasLimits']
        self.lower_limits = self.dof_props['lower']
        self.upper_limits = self.dof_props['upper']

        # # initialize default positions, limits, and speeds (make sure they are in reasonable ranges)
        defaults = np.zeros(self.num_dofs)
        for i in range(self.num_dofs):
            if self.has_limits[i]:
                if self.dof_types[i] == gymapi.DOF_ROTATION:
                    self.lower_limits[i] = clamp(self.lower_limits[i], -math.pi, math.pi)
                    self.upper_limits[i] = clamp(self.upper_limits[i], -math.pi, math.pi)
                # make sure our default position is in range
                if self.lower_limits[i] > 0.0:
                    defaults[i] = self.lower_limits[i]
                elif self.upper_limits[i] < 0.0:
                    defaults[i] = self.upper_limits[i]
            else:
                # set reasonable animation limits for unlimited joints
                if self.dof_types[i] == gymapi.DOF_ROTATION:
                    # unlimited revolute joint
                    self.lower_limits[i] = -math.pi
                    self.upper_limits[i] = math.pi
                elif self.dof_types[i] == gymapi.DOF_TRANSLATION:
                    # unlimited prismatic joint
                    self.lower_limits[i] = -1.0
                    self.upper_limits[i] = 1.0
                else:
                    print("Unknown DOF type!")
                    exit()
            # set DOF position to default
            self.dof_positions[i] = defaults[i]

        # Print DOF properties
        for i in range(self.num_dofs):
            print("DOF %d" % i)
            print("  Name:     '%s'" % self.dof_names[i])
            print("  Type:     %s" % self.gym.get_dof_type_string(self.dof_types[i]))
            print("  Stiffness:  %r" % self.stiffnesses[i])
            print("  Damping:  %r" % self.dampings[i])
            print("  Armature:  %r" % self.armatures[i])
            print("  Limited?  %r" % self.has_limits[i])
            if self.has_limits[i]:
                print("    Lower   %f" % self.lower_limits[i])
                print("    Upper   %f" % self.upper_limits[i])

        # # set up the env grid
        self.num_envs = 1
        self.num_per_row = 6
        self.spacing = 2.5
        self.env_lower = gymapi.Vec3(-self.spacing, 0.0, -self.spacing)
        self.env_upper = gymapi.Vec3(self.spacing, self.spacing, self.spacing)

        # position the camera
        self.cam_pos = gymapi.Vec3(0.440, 0.256 , 0.629)
        self.cam_target = gymapi.Vec3(0.0, 0.0, 0.0)
        
        self.gym.viewer_camera_look_at(self.viewer, None, self.cam_pos, self.cam_target)

        # cache useful handles
        self.envs = []
        self.actor_handles = []

        print("Creating %d environments" % self.num_envs)
        for i in range(self.num_envs):
            # create env
            env = self.gym.create_env(self.sim, self.env_lower, self.env_upper, self.num_per_row)
            self.envs.append(env)

            # add actor
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
            # pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

            actor_handle = self.gym.create_actor(env, self.asset, pose, "actor", i, 1)
            self.actor_handles.append(actor_handle)

            props = self.gym.get_actor_dof_properties(env, actor_handle)
            props["driveMode"] = (
                          gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS,
                          gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS,
                          gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS,
                          gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS,
                          gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS,
                          )
            props["stiffness"] =  ( 
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,

                          )                          
            
            Tval = 1.0
            Rval = 0.5

            props["damping"] = (
                        Tval, Tval, Tval, Rval, Rval, Rval,
                        0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                        0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                        0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                        0.1, 0.1, 0.1, 0.1, 0.1, 0.1
            )
            
            self.gym.set_actor_dof_properties(env, actor_handle, props)

            # set default DOF positions
            self.gym.set_actor_dof_states(env, actor_handle, self.dof_states, gymapi.STATE_ALL)
        
        # Helper visualization for goal orientation
        # pickle_data = np.load("/home/mmpug/shadow_hand2.pkl", allow_pickle=True)
        # self.meta_data, self.data = pickle_data["meta_data"], pickle_data["data"]
        # print("data: ", self.data)
        self.axes_geom = gymutil.AxesGeometry(0.5)
        self.goal_quat = np.array([0.0, 0.0, 0.0, 1.0])

        self.count = 0
        self.qpos_sub = rospy.Subscriber("/qpos/Left", Float32MultiArray, self.callback)
        #codeself.qpos_sub = rospy.Subscriber("/qpos/Right", Float32MultiArray, self.callback)
    def callback(self, qpos_msg):
        # action =  torch.from_numpy( np.array(qpos_msg.data))
        # act = torch.tensor(action).repeat((self.env.num_envs, 1))
        print("got a pos msg")
        action =  list(qpos_msg.data) #28 dim 6 + 24 - 2

        action = np.array(action)
        pose = action[0:6].copy()
        #print("pose: ", pose)
        #action[6] = -1.0 * action[6]
        #action[10] = -1.0 * action[10]



        print("thumb act: ", action[23:28])
        #action[23] = -1.0 * action[23]
        #action[24] = -1.0 * action[24]
        #action[25] = -1.0 * action[25]
        #action[26] = -1.0 * action[26]

        #action[27] = -1.0 * action[27]

        action = action[4:] # 24 dim
        action[0:2] = 0.0
        #pose = np.array([0.0, 0.0, 0.0,    0.0, 0.0, 0.0])
        
        pose = pose.reshape(-1,1)
        action = action.reshape(-1,1)
        pose_test = pose.copy()

        pose_test = pose_test - pose_test
        #pose_test[0] = - 0.2
        #pose_test[1] = - 0.003
        #pose_test[2] = 0.17

        #pose_test[3:6] = 0.0

        # swith pitch yaw
        #pose_test[4] = (self.count % 100 ) * 0.01
        #pose_test[4] = -1 * pose[5]
        #pose_test[5] = -1 * pose[4]
        
        action = np.concatenate( [pose_test, action] , axis = 0)
        
        action = action.tolist()
        #print("count: ", self.count)
        self.count += 1   
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        #print("pose: ", action[0:6])
        for i in range(self.num_envs):

            self.gym.set_actor_dof_position_targets(self.envs[i], self.actor_handles[i], action)
            # update the viewer

            self.gym.clear_lines(self.viewer)
            goal_viz_T = gymapi.Transform(r=gymapi.Quat(*self.goal_quat))
            gymutil.draw_lines(self.axes_geom, self.gym, self.viewer, self.envs[i], goal_viz_T)


            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            # Wait for dt to elapse in real time.
            # This synchronizes the physics simulation with the rendering rate.
            self.gym.sync_frame_time(self.sim)
        
        return

    def run(self):
        rospy.spin()  


def main():
    rospy.init_node("isaac_mocap_left")
    isaac_node = isaac()
    isaac_node.run()

if __name__ == "__main__":
    main()   
