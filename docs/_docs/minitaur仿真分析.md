# mitaur仿真分析(个人看法)

<u>**(个人认为minitaur仿真包含两大部分: 系统辨识(就是urdf文件中关于minitaur的描述), 约束条件(就是python代码中涉及到的细节) , 不同的模式(开环系统还是反馈系统(根据init中的on_rock是否为真)))**</u>

<u>**(个人认为: 约束条件包括: 是否存在自我碰撞(?在仿真中自我碰撞到底指什么)--init函数, motor的类型(是DC motor, pd motor还是别的; 在DC motor下还分?torque control和?pose control)--init函数), 对leg的限制(调用了多个_pybullet_client的函数, 因此得具体分析_pybullet_client的定义))**</u>

<u>**(个人认为: 如果想明白minitaur的所有细节, 得懂urdf, pybullet, 机器人一些相关的知识, motor相关知识)**</u>



## 以minitaur中的函数为主线, 分析mitaur仿真

**minitaur.py** () 路径:bullet3-master/examples/pybullet/gym/pybullet_envs/bullet. 

###　设置一些初始变量,约束条件和一些objects的名称

```python
INIT_POSITION = [0, 0, .2] # 初始位置
INIT_ORIENTATION = [0, 0, 0, 1] # 初始朝向
KNEE_CONSTRAINT_POINT_RIGHT = [0, 0.005, 0.2] #?应该是右边膝盖的约束
KNEE_CONSTRAINT_POINT_LEFT = [0, 0.01, 0.2] #?应该是左边膝盖的约束
OVERHEAT_SHUTDOWN_TORQUE = 2.45 # ?过热关掉的力矩
OVERHEAT_SHUTDOWN_TIME = 1.0 // # ?过热关掉的时间. 下文中说 shutdown the motor 												that has exerted large torque 															(OVERHEAT_SHUTDOWN_TORQUE) for an extended amount of 									time (OVERHEAT_SHUTDOWN_TIME).

LEG_POSITION = ["front_left", "back_left", "front_right", "back_right"] #四只腿,分别为前左脚,																		后左脚,前右脚,后右脚
MOTOR_NAMES = [
    "motor_front_leftL_joint", "motor_front_leftR_joint",
    "motor_back_leftL_joint", "motor_back_leftR_joint",
    "motor_front_rightL_joint", "motor_front_rightR_joint",
    "motor_back_rightL_joint", "motor_back_rightR_joint"
] # 四只腿各自所对应的左边的发动机和右边的发动机

"""接下来为物体的id, ?leg(leg为什么会有16个), ?foot_link_id(foot为什么会有8个)"""
LEG_LINK_ID = [2, 3, 5, 6, 8, 9, 11, 12, 15, 16, 18, 19, 21, 22, 24, 25]
MOTOR_LINK_ID = [1, 4, 7, 10, 14, 17, 20, 23]
FOOT_LINK_ID = [3, 6, 9, 12, 16, 19, 22, 25]
BASE_LINK_ID = -1 #base编号
```



### 接下来为Minitaur类

+ **__init__**.  仿真minitaur

```python

"""重点是在说明那一块, 参数都被复制在类属性中,各自复制后的名字都在前面加上一个_"""
"""只有在DC motor下, 才有torque control和pose control的区别"""
"""除了复制的变量之外,还有两个变量, num_motors, num_legs, 在DC motor下才定义的_motor_model""" 
"""?motormodel类涉及到了专业知识"""
"""变量命名完之后,调用了reset函数"""
class Minitaur(object):
  """The minitaur class that simulates a quadruped robot from Ghost Robotics.

  """

  def __init__(self,
               pybullet_client, #?客户端
               urdf_root= os.path.join(os.path.dirname(__file__),"../data"), #urdf文件路径
               time_step=0.01, # 仿真的time step
               self_collision_enabled=False, #?自我碰撞 
               motor_velocity_limit=np.inf, #motor速度的上界
               pd_control_enabled=False, #?是否使用PD control
               accurate_motor_model_enabled=False, #?是否使用精确的DC motol
               motor_kp=1.0,	#?精确的motol model的kp
               motor_kd=0.02,  #?精确的motor model的kd
               torque_control_enabled=False,    #?使用torque_control或者pose control
               motor_overheat_protection=False, #是否开启motor 过热保护
               on_rack=False,				#debug gait
               kd_for_pd_controllers=0.3):  #pd control的kd value
    """Constructs a minitaur and reset it to the initial states.

    Args:
      pybullet_client: The instance of BulletClient to manage different
        simulations. 
      urdf_root: The path to the urdf folder.
      time_step: The time step of the simulation.
      self_collision_enabled: Whether to enable os.path.join(os.path.dirname(__file__),"../data").
      motor_velocity_limit: The upper limit of the motor velocity.
      pd_control_enabled: Whether to use PD control for the motors.
      accurate_motor_model_enabled: Whether to use the accurate DC motor model.
      motor_kp: proportional gain for the accurate motor model
      motor_kd: derivative gain for the acurate motor model
      torque_control_enabled: Whether to use the torque control, if set to
        False, pose control will be used.
      motor_overheat_protection: Whether to shutdown the motor that has exerted
        large torque (OVERHEAT_SHUTDOWN_TORQUE) for an extended amount of time
        (OVERHEAT_SHUTDOWN_TIME). See ApplyAction() in minitaur.py for more
        details.
      on_rack: Whether to place the minitaur on rack. This is only used to debug
        the walking gait. In this mode, the minitaur's base is hanged midair so
        that its walking gait is clearer to visualize.
      kd_for_pd_controllers: kd value for the pd controllers of the motors.
    """
    self.num_motors = 8
    self.num_legs = int(self.num_motors / 2)
    self._pybullet_client = pybullet_client
    self._urdf_root = urdf_root
    self._self_collision_enabled = self_collision_enabled
    self._motor_velocity_limit = motor_velocity_limit
    self._pd_control_enabled = pd_control_enabled
    self._motor_direction = [-1, -1, -1, -1, 1, 1, 1, 1]
    self._observed_motor_torques = np.zeros(self.num_motors)
    self._applied_motor_torques = np.zeros(self.num_motors)
    self._max_force = 3.5
    self._accurate_motor_model_enabled = accurate_motor_model_enabled
    self._torque_control_enabled = torque_control_enabled
    self._motor_overheat_protection = motor_overheat_protection
    self._on_rack = on_rack
    
    
    """只有在DC motor下, 才有torque control和pose control的区别"""
    if self._accurate_motor_model_enabled:
      self._kp = motor_kp
      self._kd = motor_kd
      self._motor_model = motor.MotorModel(
          torque_control_enabled=self._torque_control_enabled,
          kp=self._kp,
          kd=self._kd)
    elif self._pd_control_enabled:
      self._kp = 8
      self._kd = kd_for_pd_controllers
    else:
      self._kp = 1
      self._kd = 1
    self.time_step = time_step
    self.Reset()
```

+ reset函数

```python
"""由于在init函数中调用reset函数时没有穿入任何参数,因此reload_urdf必然为True
	根据是否有自我碰撞，对load_urf使用不同的参数,如果自我碰撞就加上self._pybullet_client.URDF_USE_SELF_COLLISION
	并且调用了 self._BuildJointNameToIdDict()
      self._BuildMotorIdList()
      self._RecordMassInfoFromURDF()
      self.ResetPose(add_constraint=True)四个函数.
     如果是一个开环系统,就加上一个限制条件
	"""

def Reset(self, reload_urdf=True):
    """Reset the minitaur to its initial states.

    Args:
      reload_urdf: Whether to reload the urdf file. If not, Reset() just place
        the minitaur back to its starting position.
    """
    if reload_urdf:
      if self._self_collision_enabled:
        self.quadruped = self._pybullet_client.loadURDF(
            "%s/quadruped/minitaur.urdf" % self._urdf_root,
            INIT_POSITION,
            flags=self._pybullet_client.URDF_USE_SELF_COLLISION)
      else:
        self.quadruped = self._pybullet_client.loadURDF(
            "%s/quadruped/minitaur.urdf" % self._urdf_root, INIT_POSITION)
        
        
      ##接下来为四个调用的函数,
      self._BuildJointNameToIdDict()
      self._BuildMotorIdList()
      self._RecordMassInfoFromURDF()
      self.ResetPose(add_constraint=True)
        
        
      ##如果是开环系统的话,就对minitaur加一个限制条件,从而使得它在原地
      if self._on_rack:
        self._pybullet_client.createConstraint(
            self.quadruped, -1, -1, -1, self._pybullet_client.JOINT_FIXED,
            [0, 0, 0], [0, 0, 0], [0, 0, 1])
    else:
      self._pybullet_client.resetBasePositionAndOrientation(
          self.quadruped, INIT_POSITION, INIT_ORIENTATION)
      self._pybullet_client.resetBaseVelocity(self.quadruped, [0, 0, 0],
                                              [0, 0, 0])
      self.ResetPose(add_constraint=False)

    self._overheat_counter = np.zeros(self.num_motors)
    self._motor_enabled_list = [True] * self.num_motors
```

+ 分析四个调用的函数:**self._BuildJointNameToIdDict**
        **self._BuildMotorIdList**
        **self._RecordMassInfoFromURDF()**
       **self.ResetPose(add_constraint=True)**

  ```python
  """?创建从名字到id的映射_joint_nam_to_id"""
  def _BuildJointNameToIdDict(self):
      num_joints = self._pybullet_client.getNumJoints(self.quadruped)
      self._joint_name_to_id = {}
      for i in range(num_joints):
        joint_info = self._pybullet_client.getJointInfo(self.quadruped, i)
        self._joint_name_to_id[joint_info[1].decode("UTF-8")] = joint_info[0]
    
  """?创建motor_name对应的id列表 _motor_id_list"""
    def _BuildMotorIdList(self):
      self._motor_id_list = [
          self._joint_name_to_id[motor_name] for motor_name in MOTOR_NAMES
      ]
      
      """?创建base mass的_base_mass_urdf和[leg mass, motor mass]的列表_leg_masses_urdf"""
      def _RecordMassInfoFromURDF(self):
      	self._base_mass_urdf = self._pybullet_client.getDynamicsInfo(
          self.quadruped, BASE_LINK_ID)[0]
      	self._leg_masses_urdf = []
      	self._leg_masses_urdf.append(
          self._pybullet_client.getDynamicsInfo(self.quadruped, LEG_LINK_ID[0])[
              0])
      	self._leg_masses_urdf.append(
          self._pybullet_client.getDynamicsInfo(self.quadruped, MOTOR_LINK_ID[0])[
              0])
  ```

  

```python
 """对所有的leg都调用了_ResetPoseForLeg"""
 def ResetPose(self, add_constraint):
    """Reset the pose of the minitaur.

    Args:
      add_constraint: Whether to add a constraint at the joints of two feet.
    """
    for i in range(self.num_legs):
      self._ResetPoseForLeg(i, add_constraint)
    
 """调用了一系列的_pybullet_client函数, 有点不懂这些"""
 def _ResetPoseForLeg(self, leg_id, add_constraint):
    """Reset the initial pose for the leg.

    Args:
      leg_id: It should be 0, 1, 2, or 3, which represents the leg at
        front_left, back_left, front_right and back_right.
      add_constraint: Whether to add a constraint at the joints of two feet.
    """
    knee_friction_force = 0
    half_pi = math.pi / 2.0
    knee_angle = -2.1834

    leg_position = LEG_POSITION[leg_id]
    self._pybullet_client.resetJointState(
        self.quadruped,
        self._joint_name_to_id["motor_" + leg_position + "L_joint"],
        self._motor_direction[2 * leg_id] * half_pi,
        targetVelocity=0)
    self._pybullet_client.resetJointState(
        self.quadruped,
        self._joint_name_to_id["knee_" + leg_position + "L_link"],
        self._motor_direction[2 * leg_id] * knee_angle,
        targetVelocity=0)
    self._pybullet_client.resetJointState(
        self.quadruped,
        self._joint_name_to_id["motor_" + leg_position + "R_joint"],
        self._motor_direction[2 * leg_id + 1] * half_pi,
        targetVelocity=0)
    self._pybullet_client.resetJointState(
        self.quadruped,
        self._joint_name_to_id["knee_" + leg_position + "R_link"],
        self._motor_direction[2 * leg_id + 1] * knee_angle,
        targetVelocity=0)
    if add_constraint:
      self._pybullet_client.createConstraint(
          self.quadruped, self._joint_name_to_id["knee_"
                                                 + leg_position + "R_link"],
          self.quadruped, self._joint_name_to_id["knee_"
                                                 + leg_position + "L_link"],
          self._pybullet_client.JOINT_POINT2POINT, [0, 0, 0],
          KNEE_CONSTRAINT_POINT_RIGHT, KNEE_CONSTRAINT_POINT_LEFT)

    if self._accurate_motor_model_enabled or self._pd_control_enabled:
      # Disable the default motor in pybullet.
      self._pybullet_client.setJointMotorControl2(
          bodyIndex=self.quadruped,
          jointIndex=(self._joint_name_to_id["motor_"
                                             + leg_position + "L_joint"]),
          controlMode=self._pybullet_client.VELOCITY_CONTROL,
          targetVelocity=0,
          force=knee_friction_force)
      self._pybullet_client.setJointMotorControl2(
          bodyIndex=self.quadruped,
          jointIndex=(self._joint_name_to_id["motor_"
                                             + leg_position + "R_joint"]),
          controlMode=self._pybullet_client.VELOCITY_CONTROL,
          targetVelocity=0,
          force=knee_friction_force)

    else:
      self._SetDesiredMotorAngleByName(
          "motor_" + leg_position + "L_joint",
          self._motor_direction[2 * leg_id] * half_pi)
      self._SetDesiredMotorAngleByName("motor_" + leg_position + "R_joint",
                                       self._motor_direction[2 * leg_id
                                                             + 1] * half_pi)

    self._pybullet_client.setJointMotorControl2(
        bodyIndex=self.quadruped,
        jointIndex=(self._joint_name_to_id["knee_" + leg_position + "L_link"]),
        controlMode=self._pybullet_client.VELOCITY_CONTROL,
        targetVelocity=0,
        force=knee_friction_force)
    self._pybullet_client.setJointMotorControl2(
        bodyIndex=self.quadruped,
        jointIndex=(self._joint_name_to_id["knee_" + leg_position + "R_link"]),
        controlMode=self._pybullet_client.VELOCITY_CONTROL,
        targetVelocity=0,
        force=knee_friction_force)

```

