'''
Ambient rover configuration for mujoco

This file must be in the following folder: $GYM/gym/envs/mujoco

Where $GYM is your path to the gym folder


rover-4-wheels.xml qpos and qvel

qpos[0] = rover's x position
qpos[1] = rover's y position
qpos[2] = rover's z position
qpos[3] = rover's w quaternion vector
qpos[4] = rover's a quaternion vector
qpos[5] = rover's b quaternion vector
qpos[6] = rover's c quaternion vector
qpos[7] = rear left wheel rotation angle
qpos[8] = rear right wheel rotation angle
qpos[9] = steerbar rotation angle
qpos[10]= front left wheel rotation angle
qpos[11]= front right wheel rotation angle
qpos[12]= drive motor rotation angle (the prism bwtween rear wheels)

qvel[0] = rover's x velocity
qvel[1] = rover's y velocity
qvel[2] = rover's z velocity
qvel[3] = rover's x angular velocity
qvel[4] = rover's y angular velocity
qvel[5] = rover's z angular velocity
qvel[6] = rear left wheel angular velocity
qvel[7] = rear right wheel angular velocity
qvel[8] = steerbar angular velocity
qvel[9]= front left wheel angular velocity
qvel[10]= front right wheel angular velocity
qvel[11]= drive motor angular velocity (the prism bwtween rear wheels)

'''

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class RoverRobotrek4WEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    
    current_goal = 0
    just_reached_goal = False
    gps_error = 0.1
    x_before = [0,0]
    
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'main-trekking-challenge-4wheels.xml', 4)  # __init__(self, model_path, frame_skip):
        utils.EzPickle.__init__(self)

    def step(self, action):
        '''
        Simulation Step : Here is defined the observation_state, the reward and if the episode is done 
        '''
        
        x_before = self.x_before
        self.do_simulation(action, self.frame_skip) #action[0] = steering_torque, action[1] = traction_torque
        gps_exact, ob = self._get_obs()
        x_after = ob[0:2]
        cur_goal = ob[-2:]
        desired_trajectory = cur_goal-x_before
        performed_trajectory = x_after - x_before
        forward_reward = 0.1*np.dot(performed_trajectory,desired_trajectory)/(np.sqrt(np.square(desired_trajectory).sum())*self.dt)
        ctrl_cost = .001 * np.square(action).sum()
        survive_reward = .0001
        self.x_before = x_after
        
        on_base, n_spot = self.line_reader(gps_exact)
        lamp_state = self.sim.model.light_active[1]
        goal, on_base = self.is_in_goal(gps_exact)
        
        if goal:
            self.sim.model.light_active[1]=1
            self.just_reached_goal = True
            self.update_goal()
        else:
            if not on_base:
                self.sim.model.light_active[1]=0
                self.just_reached_goal = False
            
        r = survive_reward + forward_reward - ctrl_cost
        
        if (self.current_goal ==-1):
            done = True
            r = r + 20
        elif (gps_exact[0] < 0) or (gps_exact[1] < 0) or (gps_exact[0] > 44) or (gps_exact[1] > 25): #penalty for leaving the camp
            done = True
            r-=10
        elif self.sim.data.time >= 99.9:
            done = True
        else: done = False
        
        return ob, r, done, {}
    
    def is_in_goal(self, gps_exact):
        on_base, n_spot = self.line_reader(gps_exact)
        if on_base and n_spot==self.current_goal:
            return True,on_base
        else:
            return False,on_base
    
    def _get_obs(self):
        '''
        Returns the observation_state
        '''
        # gps exato
        gps_exact = self.sim.data.body_xpos[1][0:2].copy()
        # vel exata
        speed_exact = [self.sim.data.qvel[0:2]].copy()
        
        #gps sensor com 
        gps_sensor = gps_exact + self.gps_error*np.random.rand(2)        
        # vel sensor
        speed_sensor = speed_exact + self.gps_error*np.random.rand(2)
        
        # orientation_rover
        orientation_rover = self.sim.data.body_xmat[1][0:2].copy()
        rover_ang_speed = self.sim.data.qvel[5]
        steer_bar_angle = self.sim.data.qpos[9]
        steer_bar_angspeed = self.sim.data.qvel[8]
        
        if self.current_goal == 0:
            coordinates_goal = np.asarray([40,20])
            goal = np.asarray([0,0,1])
        elif self.current_goal == 1:
            coordinates_goal = np.asarray([30,2])
            goal = np.asarray([0,1,0])
        elif self.current_goal == 2:
            coordinates_goal = np.asarray([6,18])
            goal = np.asarray([1,0,0])
        elif self.current_goal == -1:
            coordinates_goal = np.asarray([0,0])
            goal = np.asarray([0,0,0])
        
        return gps_exact, np.concatenate([gps_sensor.flat, orientation_rover.flat, [steer_bar_angle], speed_sensor.flat, [rover_ang_speed], [steer_bar_angspeed], goal.flat, coordinates_goal.flat])
        
    def reset_model(self):
        '''
        Reset the model
        '''
        self.set_state(self.init_qpos, self.init_qvel)
        self.current_goal = 0
        just_reached_goal = False
        gps_exact, ob = self._get_obs()
        self.x_before = ob[0:2].copy()+self.gps_error*np.random.rand(2)
        return ob

    def viewer_setup(self):
        '''
        How to render the camera
        '''
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent * 0.5
        v.cam.lookat[2] = 0.12250000000000005 # v.model.stat.center[2]

    def update_goal(self):
        print("Goal", self.current_goal, "reached!")
        if self.current_goal != 2: self.current_goal+=1
        else: self.current_goal = -1
        print("Next Goal:", self.current_goal)

    def line_reader(self, gps_exact):
        '''
        Simulation of a color sensor
        
        returns the following:
            -> on_base = The sensor detects if the rover is over a base or not
            -> n_spot = Which base the rover is on (This is used for updating the goal)
        '''
        if (39.5 <= gps_exact[0] <= 40.5) and ( 19.5 <= gps_exact[1] <= 20.5):
            return 1, 0
        elif (29.5 <= gps_exact[0] <= 30.5) and ( 1.5 <= gps_exact[1] <= 2.5):
            return 1, 1
        elif (5.5 <= gps_exact[0] <= 6.5) and (17.5 <= gps_exact[1] <= 18.5):
            return 1, 2
        else: return 0, -1
