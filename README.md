# RobotrekkingDRL2019

Here is the code for the paper: Deep Reinforcement Learning Control of Autonomous Terrestrial
Wheeled Robots in a Challenge Task, published on Brahur Brasero 2019 Workshop

We used the following python libraries: tensorflow, gym and spinnnig up. Simulation was performed with mujoco 


## Instaling the Environment

The files inside the mujoco folder must be in: `$GYM/gym/envs/mujoco`

The files inside the mujoco/assets folder must be in: `$GYM/gym/envs/mujoco/assets`

Where $GYM is your path to the gym folder


And add the following lines of code in the `:$GYM/gym/envs/__init__.py file`

```python 
register(
    id='Rover3W-v0',
    entry_point='gym.envs.mujoco:RoverRobotrek3WEnv',
    reward_threshold=1000,
    )

register(
    id='Rover4W-v0',
    entry_point='gym.envs.mujoco:RoverRobotrek4WEnv',
    reward_threshold=1000,
    )
```


## DRL

The parameters for the DRL can be found on the `DRL/PPO_spinningup_lines.txt` file, these must be run on the linux terminal

## Inputs and outputs

The environment files have the following inputs and outputs:

`input = [[x_GPS,y_GPS, orientation_rover, line_reader], [x_GOAL, y_GOAL]]`

`output = [mean_Torque_traction, SD_Torque_traction], [mean_Torque_Steering, SD_Torque_Steering]]`


##Demonstration

A Demonstration can be found here: https://youtu.be/y4M_ypfiLig
