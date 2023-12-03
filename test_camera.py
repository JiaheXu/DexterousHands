# Import necessary libraries
import cv2
import numpy as np
from isaacgym import gymapi, gymutil

# Initialize Isaac Gym
gym = gymapi.acquire_gym()

# Set up the simulator
sim_params = gymapi.SimParams()
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

# Check if the simulator is created successfully
if sim is None:
    raise Exception("Failed to create the sim")


 # add ground plane
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, plane_params)

# set up env grid
spacing = 2.0
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# create env
env = gym.create_env(sim, env_lower, env_upper, 1)



# add actor
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.0, 0.25, 0.0)
pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
ball_asset = gym.create_sphere(sim, 0.25, None)
actor_handle = gym.create_actor(env, ball_asset, pose, "ball", 0, 0)

# add 2 cameras
cam_props = gymapi.CameraProperties()
cam_props.width = 512
cam_props.height = 512
cam_props.enable_tensors = False
cam1 = gym.create_camera_sensor(env, cam_props)
cam2 = gym.create_camera_sensor(env, cam_props)


# set camera 1 location
gym.set_camera_location(cam1, env, gymapi.Vec3(1, 1, 1), gymapi.Vec3(0, 0, 0))
# set camera 2 location using the cam1's transform
gym.set_camera_location(cam2, env, gymapi.Vec3(1, 1, 3), gymapi.Vec3(0, 0, 0))

viewer = gym.create_viewer(sim, gymapi.CameraProperties())


# Main simulation loop
while True:
    # Step the simulation
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    gym.step_graphics(sim)

    # render sensors and refresh camera tensors
    gym.render_all_camera_sensors(sim)

    # Get camera images from the environment
    image1 = gym.get_camera_image(sim, env, cam1, gymapi.IMAGE_COLOR)
    image2 = gym.get_camera_image(sim, env, cam2, gymapi.IMAGE_COLOR)

    # Convert images to a format suitable for OpenCV
    image1 = image1.reshape(image1.shape[0], -1, 4)[..., :3]
    # image1 = cv2.cvtColor(image1, cv2.COLOR_RGBA2BGR)
    image2 = image2.reshape(image2.shape[0], -1, 4)[..., :3]

    # image2 = cv2.cvtColor(image2, cv2.COLOR_RGBA2BGR)

    # Display images
    cv2.imshow("Camera 1", image1)
    cv2.imshow("Camera 2", image2)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    gym.step_graphics(sim)  
    gym.draw_viewer(viewer, sim, True)

# Cleanup
gym.destroy_sim(sim)
cv2.destroyAllWindows()