# RobotrekkingDRL2019

Here is the code for the paper: Deep Reinforcement Learning Control of Autonomous Terrestrial
Wheeled Robots in a Challenge Task, published on Brahur Brasero 2019 Workshop

We used the following python libraries: tensorflow and gym. Simulation was performed with mujoco 


## Instaling the Environment

If $GYM is your path to the gym folder

The files inside mujoco folder must be in: `$GYM/gym/envs/mujoco`

The files inside mujoco/assets folder must be in: `$GYM/gym/envs/mujoco/assets`


And add the following lines of code in the `:$GYM/gym/envs/__init__.py` file

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


## Demonstration

A Demonstration can be found here: https://youtu.be/y4M_ypfiLig
