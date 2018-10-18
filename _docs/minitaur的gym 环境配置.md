# minitaur的gym 环境配置大概分析（个人看法）

<u>**"""minitaur环境配置在于：plane环境配置，minitaur环境配置，环境的一些参数随机化，动作和观察的交互**</u>

<u>**，pybullet的摄像配置．**</u>

<u>**plane环境配置大体在urdf文件中; minitaur环境封装在minitaur类中；环境的一些参数随机化在minitaur_env_randomizer.py中，　这个很简单分析; 动作和观察的交互，就是动作和观察是如何发生作用的，这还需要更为细致的分析，个人觉得这是难点"""**</u>









分析文件: **minitaur_gym_env.py**. 

##　全局变量:

```python
import os, inspect
import math
import time
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pybullet
from . import bullet_client
from . import minitaur
import os
import pybullet_data
from . import minitaur_env_randomizer
from pkg_resources import parse_version


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))　# 所在文件的目录
parentdir = os.path.dirname(os.path.dirname(currentdir))　# 目录的目录的目录　
os.sys.path.insert(0,parentdir)　# 添加搜索路径

NUM_SUBSTEPS = 5  #?这个是
NUM_MOTORS = 8　# motor的数量
MOTOR_ANGLE_OBSERVATION_INDEX = 0

#?这三个变量分别代表什么含义
MOTOR_VELOCITY_OBSERVATION_INDEX = MOTOR_ANGLE_OBSERVATION_INDEX + NUM_MOTORS #8
MOTOR_TORQUE_OBSERVATION_INDEX = MOTOR_VELOCITY_OBSERVATION_INDEX + NUM_MOTORS #16
BASE_ORIENTATION_OBSERVATION_INDEX = MOTOR_TORQUE_OBSERVATION_INDEX + NUM_MOTORS #24

ACTION_EPS = 0.01 #动作误差
OBSERVATION_EPS = 0.01 #观察误差
RENDER_HEIGHT = 720 #?导出视屏的高度
RENDER_WIDTH = 960  #?导出视屏的宽度

```

## MinitaurBulletEnv类分析

### 1.类的文档说明：

**<u>"""The gym environment for the minitaur.</u>**

  **<u>It simulates the locomotion of a minitaur, a quadruped robot. The state space</u>**
  **<u>include the angles, velocities and torques for all the motors and the action</u>**
  **<u>space is the desired motor angle for each motor. The reward function is based</u>**
  **<u>on how far the minitaur walks in 1000 steps and penalizes the energy</u>**
  **<u>expenditure.</u>**

  <u>"""</u>

(构造了minitaur的gym环境, 状态空间包括所有motor的角度, 速度和力矩. 动作包括所有motor的角度.

奖励函数是基于在1000stepsminitaur走了多远并且惩罚能量的损耗)

### 2.__init__函数

```python
"""所有函数参数都得到了复制, 除了hard_reset, 它直接在函数体里被设置True
增加了?一个step的时间(?以s为单位)的time_step,
	 不知道什么意思的_num_bullet_solver_iterations,
	 observation数值化的列表的_observation,
	 step counter的_env_step_counter,
	 上次base的位置的_last_base_position,
	 ?动作的bound的_action_bound,
	 ?pybullet相关函数: _pybullet_client
	 ?不太清楚的 self._cam_dist = 1.0 
    			self._cam_yaw = 0 
                self._cam_pitch = -30 ，从_reset函数中可以看出这是pybullet中调整摄											的像头一些参数
     ?上一帧的时间_last_frame_time
     
     上界和下界都是
     观察空间上界 observation_high: [ [pi] * 8(joint angle),[MOTOR_SPEED_LIMIT] * 8(joint 								velocity), [OBSERVED_TORQUE_LIMIT] * 8 (joint torque), 									[1] * 4 (Quaternion of base orientation)]
     观察空间下界 observation_low = -observation_high
     动作空间维度: action_dim
     动作空间	action_space 
     观察空间	observation 
     
     主要调用了两个函数: _seed 产生一个固定随机种子的随机函数"
     			   _reset """
	
  def __init__(self,
               urdf_root=pybullet_data.getDataPath(), ##urdf文件夹
               action_repeat=1, ##多少个step之后采取动作
               distance_weight=1.0, ##reward中distance的权重
               energy_weight=0.005, ##reward中energy的权重
               shake_weight=0.0,    ##不需要
               drift_weight=0.0,    ##不需要
               distance_limit=float("inf"),  ##episode距离的上界
               observation_noise_stdev=0.0,  ##观察燥声的标准差
               self_collision_enabled=True,  ##自我碰撞
               motor_velocity_limit=np.inf, ##motor的速度的上界
               pd_control_enabled=False,#not needed to be true if accurate motor model is enabled (has its own better PD)
               leg_model_enabled=True, ## 重新参数化动作空间
               accurate_motor_model_enabled=True, ##是否使用精确的DC motor模型
               motor_kp=1.0,##比例增益
               motor_kd=0.02, ##导数增益
               torque_control_enabled=False, ##采用torque control还是pose　control
               motor_overheat_protection=True,
               hard_reset=True, ##当reset时,是否重载一切东西
               on_rack=False,
               render=False, ##是否渲染仿真
               kd_for_pd_controllers=0.3, 
               env_randomizer=minitaur_env_randomizer.MinitaurEnvRandomizer()): ##随机化物理环境
    """Initialize the minitaur gym environment.

    Args:
      urdf_root: The path to the urdf data folder.
      action_repeat: The number of simulation steps before actions are applied.
      distance_weight: The weight of the distance term in the reward.
      energy_weight: The weight of the energy term in the reward.
      shake_weight: The weight of the vertical shakiness term in the reward.
      drift_weight: The weight of the sideways drift term in the reward.
      distance_limit: The maximum distance to terminate the episode.
      observation_noise_stdev: The standard deviation of observation noise.
      self_collision_enabled: Whether to enable self collision in the sim.
      motor_velocity_limit: The velocity limit of each motor.
      pd_control_enabled: Whether to use PD controller for each motor.
      leg_model_enabled: Whether to use a leg motor to reparameterize the action
        space.
      accurate_motor_model_enabled: Whether to use the accurate DC motor model.
      motor_kp: proportional gain for the accurate motor model.
      motor_kd: derivative gain for the accurate motor model.
      torque_control_enabled: Whether to use the torque control, if set to
        False, pose control will be used.
      motor_overheat_protection: Whether to shutdown the motor that has exerted
        large torque (OVERHEAT_SHUTDOWN_TORQUE) for an extended amount of time
        (OVERHEAT_SHUTDOWN_TIME). See ApplyAction() in minitaur.py for more
        details.
      hard_reset: Whether to wipe the simulation and load everything when reset
        is called. If set to false, reset just place the minitaur back to start
        position and set its pose to initial configuration.
      on_rack: Whether to place the minitaur on rack. This is only used to debug
        the walking gait. In this mode, the minitaur's base is hanged midair so
        that its walking gait is clearer to visualize.
      render: Whether to render the simulation.
      kd_for_pd_controllers: kd value for the pd controllers of the motors
      env_randomizer: An EnvRandomizer to randomize the physical properties
        during reset().
    """
    self._time_step = 0.01 ## addition
    self._action_repeat = action_repeat 
    self._num_bullet_solver_iterations = 300 ## addition
    self._urdf_root = urdf_root
    self._self_collision_enabled = self_collision_enabled
    self._motor_velocity_limit = motor_velocity_limit
    self._observation = []  ## addition
    self._env_step_counter = 0   ## addition
    self._is_render = render
    self._last_base_position = [0, 0, 0]  ## addition
    self._distance_weight = distance_weight
    self._energy_weight = energy_weight
    self._drift_weight = drift_weight
    self._shake_weight = shake_weight
    self._distance_limit = distance_limit
    self._observation_noise_stdev = observation_noise_stdev
    self._action_bound = 1  ## addition
    self._pd_control_enabled = pd_control_enabled
    self._leg_model_enabled = leg_model_enabled
    self._accurate_motor_model_enabled = accurate_motor_model_enabled
    self._motor_kp = motor_kp
    self._motor_kd = motor_kd
    self._torque_control_enabled = torque_control_enabled
    self._motor_overheat_protection = motor_overheat_protection
    self._on_rack = on_rack
    self._cam_dist = 1.0 ## addition
    self._cam_yaw = 0 ## addition
    self._cam_pitch = -30 ## addition
    self._hard_reset = True
    self._kd_for_pd_controllers = kd_for_pd_controllers
    self._last_frame_time = 0.0 ## addition
    print("urdf_root=" + self._urdf_root)
    self._env_randomizer = env_randomizer
    # PD control needs smaller time step for stability.
    if pd_control_enabled or accurate_motor_model_enabled:
      self._time_step /= NUM_SUBSTEPS
      self._num_bullet_solver_iterations /= NUM_SUBSTEPS
      self._action_repeat *= NUM_SUBSTEPS

    if self._is_render:
      self._pybullet_client = bullet_client.BulletClient(
          connection_mode=pybullet.GUI)
    else:
      self._pybullet_client = bullet_client.BulletClient()

    self._seed() #产生一个随机种子的随机函数
    self.reset() 
    observation_high = (
        self.minitaur.GetObservationUpperBound() + OBSERVATION_EPS)
    observation_low = (
        self.minitaur.GetObservationLowerBound() - OBSERVATION_EPS)
    action_dim = 8
    action_high = np.array([self._action_bound] * action_dim)
    self.action_space = spaces.Box(-action_high, action_high)
    self.observation_space = spaces.Box(observation_low, observation_high)
    self.viewer = None
    self._hard_reset = hard_reset  # This assignment need to be after reset()
```



## 3._reset函数

```python
"""配置plane，minitur和随机化环境"""

def _reset(self):
    if self._hard_reset:
      self._pybullet_client.resetSimulation()  ##从空世界开始
      self._pybullet_client.setPhysicsEngineParameter(
          numSolverIterations=int(self._num_bullet_solver_iterations)) 				##numSolverIterations是指Choose  the  number  of  constraint  solver iteration
      self._pybullet_client.setTimeStep(self._time_step) 
      plane = self._pybullet_client.loadURDF("%s/plane.urdf" % self._urdf_root) ## 加载平面的urdf
      self._pybullet_client.changeVisualShape(plane,-1,rgbaColor=[1,1,1,0.9]) ##改变形状的纹理
      self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_PLANAR_REFLECTION,0) ##？You  can  configure  some  settings  of  the  built-in  OpenGL  visualizer,  such  as  enabling  or  disabling　wireframe,  shadows  and  GUI  rendering.
        
      self._pybullet_client.setGravity(0, 0, -10)　##设置重力
      acc_motor = self._accurate_motor_model_enabled
      motor_protect = self._motor_overheat_protection
      self.minitaur = (minitaur.Minitaur(
          pybullet_client=self._pybullet_client,
          urdf_root=self._urdf_root,
          time_step=self._time_step,
          self_collision_enabled=self._self_collision_enabled,
          motor_velocity_limit=self._motor_velocity_limit,
          pd_control_enabled=self._pd_control_enabled,
          accurate_motor_model_enabled=acc_motor,
          motor_kp=self._motor_kp,
          motor_kd=self._motor_kd,
          torque_control_enabled=self._torque_control_enabled,
          motor_overheat_protection=motor_protect,
          on_rack=self._on_rack,
          kd_for_pd_controllers=self._kd_for_pd_controllers))
    else:
      self.minitaur.Reset(reload_urdf=False)

    if self._env_randomizer is not None:
      self._env_randomizer.randomize_env(self)

    self._env_step_counter = 0
    self._last_base_position = [0, 0, 0]
    self._objectives = []
    
    self._pybullet_client.resetDebugVisualizerCamera(
        self._cam_dist, self._cam_yaw, self._cam_pitch, [0, 0, 0]) ##设置摄像的相关东西，　英文解释：For the 3D OpenGL Visualizer, set the camera distance, yaw, pitch and target position.
    if not self._torque_control_enabled:
      for _ in range(100):
        if self._pd_control_enabled or self._accurate_motor_model_enabled:
          self.minitaur.ApplyAction([math.pi / 2] * 8)
        self._pybullet_client.stepSimulation()
    return self._noisy_observation() ##返回带有燥音的观察

```

