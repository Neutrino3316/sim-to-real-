---
title: "代码分析"
permalink: /pybullet_minitaur/
last_modified_at: 2018-08-07T10:34:04+08:00
toc: true
toc_sticky: true
---

[TOC]

Figuring out how pybullet minitaur works

# Source

[paper](https://arxiv.org/abs/1804.10332)

[code](https://git.io/vp0V3)

# Code analysis

## Run the code

### Origin method

First, `cd` to `~/zyc/bullet3-master/examples/pybullet/gym/pybullet_envs/` this folder and then run the following command.
```bash
python3 -m pybullet_envs.agents.train_ppo --config=pybullet_pendulum --logdir=pendulum
```

### Some outputs

```
hadoop@gpu3:~/zyc/bullet3-master/examples/pybullet/gym/pybullet_envs$ python3 -m pybullet_envs.agents.train_ppo --config=pybullet_pendulum --logdir=pendulum
pybullet build time: Jul 12 2018 10:40:53
current_dir=/home/hadoop/.local/lib/python3.5/site-packages/pybullet_envs/bullet
INFO:tensorflow:Start a new run and write summaries and checkpoints to pendulum/20180816T103105-pybullet_pendulum.
WARN: Environment '<class 'pybullet_envs.gym_pendulum_envs.InvertedPendulumBulletEnv'>' has deprecated methods '_step' and '_reset' rather than 'step' and 'reset'. Compatibility code invoked. Set _gym_disable_underscore_compat = True to disable this behavior.
WARN: Environment '<class 'pybullet_envs.gym_pendulum_envs.InvertedPendulumBulletEnv'>' has deprecated methods '_step' and '_reset' rather than 'step' and 'reset'. Compatibility code invoked. Set _gym_disable_underscore_compat = True to disable this behavior.
INFO:tensorflow:Not normalizing infinite observation range.
INFO:tensorflow:Not normalizing infinite observation range.
WARN: Environment '<class 'pybullet_envs.gym_pendulum_envs.InvertedPendulumBulletEnv'>' has deprecated methods '_step' and '_reset' rather than 'step' and 'reset'. Compatibility code invoked. Set _gym_disable_underscore_compat = True to disable this behavior.
INFO:tensorflow:Not normalizing infinite observation range.
WARN: Environment '<class 'pybullet_envs.gym_pendulum_envs.InvertedPendulumBulletEnv'>' has deprecated methods '_step' and '_reset' rather than 'step' and 'reset'. Compatibility code invoked. Set _gym_disable_underscore_compat = True to disable this behavior.
INFO:tensorflow:Not normalizing infinite observation range.
WARN: Environment '<class 'pybullet_envs.gym_pendulum_envs.InvertedPendulumBulletEnv'>' has deprecated methods '_step' and '_reset' rather than 'step' and 'reset'. Compatibility code invoked. Set _gym_disable_underscore_compat = True to disable this behavior.
INFO:tensorflow:Not normalizing infinite observation range.
WARN: Environment '<class 'pybullet_envs.gym_pendulum_envs.InvertedPendulumBulletEnv'>' has deprecated methods '_step' and '_reset' rather than 'step' and 'reset'. Compatibility code invoked. Set _gym_disable_underscore_compat = True to disable this behavior.
INFO:tensorflow:Not normalizing infinite observation range.
WARN: Environment '<class 'pybullet_envs.gym_pendulum_envs.InvertedPendulumBulletEnv'>' has deprecated methods '_step' and '_reset' rather than 'step' and 'reset'. Compatibility code invoked. Set _gym_disable_underscore_compat = True to disable this behavior.
INFO:tensorflow:Not normalizing infinite observation range.
WARN: Environment '<class 'pybullet_envs.gym_pendulum_envs.InvertedPendulumBulletEnv'>' has deprecated methods '_step' and '_reset' rather than 'step' and 'reset'. Compatibility code invoked. Set _gym_disable_underscore_compat = True to disable this behavior.
INFO:tensorflow:Not normalizing infinite observation range.
WARN: Environment '<class 'pybullet_envs.gym_pendulum_envs.InvertedPendulumBulletEnv'>' has deprecated methods '_step' and '_reset' rather than 'step' and 'reset'. Compatibility code invoked. Set _gym_disable_underscore_compat = True to disable this behavior.
INFO:tensorflow:Not normalizing infinite observation range.
WARN: Environment '<class 'pybullet_envs.gym_pendulum_envs.InvertedPendulumBulletEnv'>' has deprecated methods '_step' and '_reset' rather than 'step' and 'reset'. Compatibility code invoked. Set _gym_disable_underscore_compat = True to disable this behavior.
INFO:tensorflow:Not normalizing infinite observation range.
WARN: Environment '<class 'pybullet_envs.gym_pendulum_envs.InvertedPendulumBulletEnv'>' has deprecated methods '_step' and '_reset' rather than 'step' and 'reset'. Compatibility code invoked. Set _gym_disable_underscore_compat = True to disable this behavior.
INFO:tensorflow:Not normalizing infinite observation range.
WARN: Environment '<class 'pybullet_envs.gym_pendulum_envs.InvertedPendulumBulletEnv'>' has deprecated methods '_step' and '_reset' rather than 'step' and 'reset'. Compatibility code invoked. Set _gym_disable_underscore_compat = True to disable this behavior.
INFO:tensorflow:Not normalizing infinite observation range.
WARN: Environment '<class 'pybullet_envs.gym_pendulum_envs.InvertedPendulumBulletEnv'>' has deprecated methods '_step' and '_reset' rather than 'step' and 'reset'. Compatibility code invoked. Set _gym_disable_underscore_compat = True to disable this behavior.
INFO:tensorflow:Not normalizing infinite observation range.
WARN: Environment '<class 'pybullet_envs.gym_pendulum_envs.InvertedPendulumBulletEnv'>' has deprecated methods '_step' and '_reset' rather than 'step' and 'reset'. Compatibility code invoked. Set _gym_disable_underscore_compat = True to disable this behavior.
INFO:tensorflow:Not normalizing infinite observation range.
WARN: Environment '<class 'pybullet_envs.gym_pendulum_envs.InvertedPendulumBulletEnv'>' has deprecated methods '_step' and '_reset' rather than 'step' and 'reset'. Compatibility code invoked. Set _gym_disable_underscore_compat = True to disable this behavior.
INFO:tensorflow:Not normalizing infinite observation range.
WARN: Environment '<class 'pybullet_envs.gym_pendulum_envs.InvertedPendulumBulletEnv'>' has deprecated methods '_step' and '_reset' rather than 'step' and 'reset'. Compatibility code invoked. Set _gym_disable_underscore_compat = True to disable this behavior.
INFO:tensorflow:Not normalizing infinite observation range.
WARN: Environment '<class 'pybullet_envs.gym_pendulum_envs.InvertedPendulumBulletEnv'>' has deprecated methods '_step' and '_reset' rather than 'step' and 'reset'. Compatibility code invoked. Set _gym_disable_underscore_compat = True to disable this behavior.
INFO:tensorflow:Not normalizing infinite observation range.
WARN: Environment '<class 'pybullet_envs.gym_pendulum_envs.InvertedPendulumBulletEnv'>' has deprecated methods '_step' and '_reset' rather than 'step' and 'reset'. Compatibility code invoked. Set _gym_disable_underscore_compat = True to disable this behavior.
INFO:tensorflow:Not normalizing infinite observation range.
WARN: Environment '<class 'pybullet_envs.gym_pendulum_envs.InvertedPendulumBulletEnv'>' has deprecated methods '_step' and '_reset' rather than 'step' and 'reset'. Compatibility code invoked. Set _gym_disable_underscore_compat = True to disable this behavior.
INFO:tensorflow:Not normalizing infinite observation range.
WARN: Environment '<class 'pybullet_envs.gym_pendulum_envs.InvertedPendulumBulletEnv'>' has deprecated methods '_step' and '_reset' rather than 'step' and 'reset'. Compatibility code invoked. Set _gym_disable_underscore_compat = True to disable this behavior.
INFO:tensorflow:Not normalizing infinite observation range.
WARN: Environment '<class 'pybullet_envs.gym_pendulum_envs.InvertedPendulumBulletEnv'>' has deprecated methods '_step' and '_reset' rather than 'step' and 'reset'. Compatibility code invoked. Set _gym_disable_underscore_compat = True to disable this behavior.
INFO:tensorflow:Not normalizing infinite observation range.
WARN: Environment '<class 'pybullet_envs.gym_pendulum_envs.InvertedPendulumBulletEnv'>' has deprecated methods '_step' and '_reset' rather than 'step' and 'reset'. Compatibility code invoked. Set _gym_disable_underscore_compat = True to disable this behavior.
INFO:tensorflow:Not normalizing infinite observation range.
WARN: Environment '<class 'pybullet_envs.gym_pendulum_envs.InvertedPendulumBulletEnv'>' has deprecated methods '_step' and '_reset' rather than 'step' and 'reset'. Compatibility code invoked. Set _gym_disable_underscore_compat = True to disable this behavior.
INFO:tensorflow:Not normalizing infinite observation range.
WARN: Environment '<class 'pybullet_envs.gym_pendulum_envs.InvertedPendulumBulletEnv'>' has deprecated methods '_step' and '_reset' rather than 'step' and 'reset'. Compatibility code invoked. Set _gym_disable_underscore_compat = True to disable this behavior.
INFO:tensorflow:Not normalizing infinite observation range.
WARN: Environment '<class 'pybullet_envs.gym_pendulum_envs.InvertedPendulumBulletEnv'>' has deprecated methods '_step' and '_reset' rather than 'step' and 'reset'. Compatibility code invoked. Set _gym_disable_underscore_compat = True to disable this behavior.
INFO:tensorflow:Not normalizing infinite observation range.
WARN: Environment '<class 'pybullet_envs.gym_pendulum_envs.InvertedPendulumBulletEnv'>' has deprecated methods '_step' and '_reset' rather than 'step' and 'reset'. Compatibility code invoked. Set _gym_disable_underscore_compat = True to disable this behavior.
INFO:tensorflow:Not normalizing infinite observation range.
WARN: Environment '<class 'pybullet_envs.gym_pendulum_envs.InvertedPendulumBulletEnv'>' has deprecated methods '_step' and '_reset' rather than 'step' and 'reset'. Compatibility code invoked. Set _gym_disable_underscore_compat = True to disable this behavior.
INFO:tensorflow:Not normalizing infinite observation range.
WARN: Environment '<class 'pybullet_envs.gym_pendulum_envs.InvertedPendulumBulletEnv'>' has deprecated methods '_step' and '_reset' rather than 'step' and 'reset'. Compatibility code invoked. Set _gym_disable_underscore_compat = True to disable this behavior.
INFO:tensorflow:Not normalizing infinite observation range.
WARN: Environment '<class 'pybullet_envs.gym_pendulum_envs.InvertedPendulumBulletEnv'>' has deprecated methods '_step' and '_reset' rather than 'step' and 'reset'. Compatibility code invoked. Set _gym_disable_underscore_compat = True to disable this behavior.
INFO:tensorflow:Not normalizing infinite observation range.
WARN: Environment '<class 'pybullet_envs.gym_pendulum_envs.InvertedPendulumBulletEnv'>' has deprecated methods '_step' and '_reset' rather than 'step' and 'reset'. Compatibility code invoked. Set _gym_disable_underscore_compat = True to disable this behavior.
INFO:tensorflow:Not normalizing infinite observation range.
WARNING:tensorflow:Variable /= will be deprecated. Use variable.assign_div if you want assignment to the variable value or 'x = x / y' if you want a new python Tensor object.
INFO:tensorflow:Graph contains 42803 trainable variables.
2018-08-16 10:31:10.114847: I tensorflow/core/platform/cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2018-08-16 10:31:10.217949: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1212] Found device 0 with properties:
name: GeForce GTX TITAN X major: 5 minor: 2 memoryClockRate(GHz): 1.076
pciBusID: 0000:41:00.0
totalMemory: 11.92GiB freeMemory: 11.60GiB
2018-08-16 10:31:10.217985: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1312] Adding visible gpu devices: 0
2018-08-16 10:31:10.385639: I tensorflow/core/common_runtime/gpu/gpu_device.cc:993] Creating TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 11228 MB memory) -> physical GPU (device: 0, name: GeForce GTX TITAN X, pci bus id: 0000:41:00.0, compute capability: 5.2)
INFO:tensorflow:
--------------------------------------------------
Phase train (phase step 0, global step 0).
return and value: [14.5767927][0.0285066701]
normalized advantage: [-5.61396291e-07]
value loss: [1260.94226]
policy loss: [-0.270250082]
current penalty: [1]
kl change: [0.00071804109]
decrease penalty [0]
return and value: [30.424963][0.590325177]
normalized advantage: [-1.53446194e-06]
policy loss: [-0.300998092]
value loss: [4036.73511]
current penalty: [0.666666687]
kl change: [0.00107282447]
decrease penalty [0]
return and value: [27.213398][1.08306026]
normalized advantage: [-3.15666199e-07]
value loss: [3869.00098]
policy loss: [-0.280239671]
current penalty: [0.444444478]
kl change: [0.00253911247]
decrease penalty [0]
return and value: [33.0769][1.72173464]
normalized advantage: [-2.39563e-06]
kl cutoff! [1]
kl cutoff! [1]
kl cutoff! [1]
kl cutoff! [1]
kl cutoff! [1]
kl cutoff! [1]
kl cutoff! [1]
kl cutoff! [1]
kl cutoff! [1]
value loss: [4645.58887]
policy loss: [-0.307737887]
current penalty: [0.296296328]
kl change: [0.00337141287]
decrease penalty [0]
return and value: [56.4372749][2.3777442]
normalized advantage: [4.66028837e-07]
kl cutoff! [1]
kl cutoff! [2]
kl cutoff! [2]
kl cutoff! [2]
kl cutoff! [2]
kl cutoff! [2]
kl cutoff! [2]
kl cutoff! [2]
kl cutoff! [1]
kl cutoff! [1]
kl cutoff! [1]
kl cutoff! [1]
kl cutoff! [1]
kl cutoff! [1]
kl cutoff! [2]
value loss: [10686.9717]
policy loss: [-0.332822502]
current penalty: [0.197530895]
kl change: [0.00428798888]
decrease penalty [0]
INFO:tensorflow:Score 33.11538314819336.
INFO:tensorflow:
--------------------------------------------------
Phase eval (phase step 0, global step 6000).
INFO:tensorflow:Score 55.453609466552734.
INFO:tensorflow:
--------------------------------------------------
Phase train (phase step 6000, global step 12000).
return and value: [47.1139259][3.29912949]
normalized advantage: [5.48362721e-07]
kl cutoff! [1]
kl cutoff! [1]
policy loss: [-0.340067744]
value loss: [7130.77148]
current penalty: [0.131687269]
kl change: [0.00478364155]
decrease penalty [0]
return and value: [109.980789][4.03648567]
normalized advantage: [2.34285991e-07]
kl cutoff! [1]
kl cutoff! [1]
kl cutoff! [3]
kl cutoff! [3]
kl cutoff! [2]
kl cutoff! [2]
kl cutoff! [1]
value loss: [25955.6426]
policy loss: [-0.37302655]
current penalty: [0.0877915174]
kl change: [0.00810990389]
INFO:tensorflow:Score 59.21126937866211.
INFO:tensorflow:
--------------------------------------------------
Phase eval (phase step 6000, global step 18000).
INFO:tensorflow:Score 127.45161437988281.
INFO:tensorflow:
--------------------------------------------------
Phase train (phase step 12000, global step 24000).
return and value: [156.173492][5.18598413]
normalized advantage: [-2.09109e-06]
kl cutoff! [1]
kl cutoff! [1]
kl cutoff! [3]
kl cutoff! [3]
kl cutoff! [3]
kl cutoff! [1]
kl cutoff! [1]
kl cutoff! [1]
kl cutoff! [1]
kl cutoff! [1]
kl cutoff! [1]
kl cutoff! [1]
value loss: [41576.8203]
policy loss: [-0.38914451]
current penalty: [0.0877915174]
kl change: [0.00671380851]
decrease penalty [0]
INFO:tensorflow:Score 114.66666412353516.
INFO:tensorflow:
--------------------------------------------------
Phase eval (phase step 12000, global step 30000).
INFO:tensorflow:Score 153.63333129882812.
INFO:tensorflow:
--------------------------------------------------
Phase train (phase step 18000, global step 36000).
return and value: [350.295502][6.14041424]
normalized advantage: [-8.52584833e-07]
kl cutoff! [1]
kl cutoff! [2]
kl cutoff! [2]
kl cutoff! [2]
kl cutoff! [2]
kl cutoff! [1]
kl cutoff! [1]
kl cutoff! [1]
kl cutoff! [2]
policy loss: [-0.357521236]
value loss: [126717.742]
current penalty: [0.0585276783]
kl change: [0.0113684637]
INFO:tensorflow:Score 152.09677124023438.
INFO:tensorflow:
--------------------------------------------------
Phase eval (phase step 18000, global step 42000).
INFO:tensorflow:Score 200.0.
INFO:tensorflow:
--------------------------------------------------

```

Then run the tensorboard

```bash
python3 /usr/local/lib/python3.5/dist-packages/tensorboard/main.py --logdir=pendulum --host=192.168.1.115
```



### Run with jupyter notebook

#### Rewrite

The [train_test_v0.1.0.ipynb](http://192.168.1.115:8888/notebooks/zyc/bullet3-master/examples/pybullet/gym/pybullet_envs/agents/train_test_v0.1.0.ipynb) is the first rewritten file. It's rewritten from [train_ppo.py](https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/agents/train_ppo.py), which is the code that will be excuted if you use the command above. Mainly, it modifies the `sys.path` , so that it can import some custom packages.

```python
import sys
sys.path.append('/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_envs/agents')
sys.path.append('/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_envs')
sys.path.append('/home/hadoop/zyc/bullet3-master/examples/pybullet/gym')
```

Then change

```python
from . import tools
from . import configs
from . import utility
```

to

```python
import tools
import configs
import utility
```

#### Run the code

1. cd to `~/zyc/bullet3-master/examples/pybullet/gym/pybullet_envs/agents` and then `rm -rf minitaur`
2. Run the code in http://192.168.1.115:8888/notebooks/zyc/bullet3-master/examples/pybullet/gym/pybullet_envs/agents/train_test_v0.1.0.ipynb
3. `python3 /usr/local/lib/python3.5/dist-packages/tensorboard/main.py --logdir=minitaur --host=192.168.1.115`
4. python -m pybullet_envs.agents.visualize_ppo --logdir=pendulum/xxxxx --outdir=pendulum_video

## Find out which part is important

`train_ppo.py`, `configs.py` and `networks.py` are the key files to understand this program.

### train_ppo.py

This is the code that will be excuted. First, we take a look of this main function.

#### Original main function

```python
if __name__ == '__main__':
    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_string(
            'logdir', None,
            'Base directory to store logs.')
    tf.app.flags.DEFINE_string(
            'timestamp', datetime.datetime.now().strftime('%Y%m%dT%H%M%S'),
            'Sub directory to store logs.')
    tf.app.flags.DEFINE_string(
            'config', None,
            'Configuration to execute.')
    tf.app.flags.DEFINE_boolean(
            'env_processes', True,
            'Step environments in separate processes to circumvent the GIL.')
    tf.app.run()
```

#### Jupyter Notebook 

```python
# tf.reset_default_graph()
with tf.device('/gpu:0'):
#     tf.reset_default_graph()
    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_string(
        'logdir', 'pendulum', #None
        'Base directory to store logs.')
    tf.app.flags.DEFINE_string(
        'timestamp', datetime.datetime.now().strftime('%Y%m%dT%H%M%S'),
        'Sub directory to store logs.')
    tf.app.flags.DEFINE_string(
        'config', 'pybullet_pendulum', #None
        'Configuration to execute.')
    tf.app.flags.DEFINE_boolean(
        'env_processes', True,
        'Step environments in separate processes to circumvent the GIL.')
    tf.app.run()#Runs the program with an optional 'main' function and 'argv' list.
```

Jupyter Notebook can't pass the arguments with command line, that is, we can't pass arguments to these flags. Instead, I reset the default value.

#### `tf.app.run()`and`main(_)`

`tf.app.run()`calls the main function, which is the following code.

```python
def main(_):
    """Create or load configuration and launch the trainer."""
    utility.set_up_logging()
    if not FLAGS.config:
        raise KeyError('You must specify a configuration.')
    logdir = FLAGS.logdir and os.path.expanduser(os.path.join(
            FLAGS.logdir, '{}-{}'.format(FLAGS.timestamp, FLAGS.config)))
    try:
        config = utility.load_config(logdir)
    except IOError:
        config = tools.AttrDict(getattr(configs, FLAGS.config)())
        config = utility.save_config(config, logdir)
    for score in train(config, FLAGS.env_processes):
        tf.logging.info('Score {}.'.format(score))
```

It's confusing that `FLAGS.config` and `configs` is not the same thing. `configs` is the package imported by `import configs` , which is defined in `configs.py`.

If we delete the `pybullet_pendulum()` function in `configs.py`, line 11 will raise an error.

```python
config = tools.AttrDict(getattr(configs, FLAGS.config)())
```

> AttributeError: module 'configs' has no attribute 'pybullet_pendulum'

This shows that the `config` flag is the key to call the function in `configs.py`

#### `train.py`

```python
def train(config, env_processes):
    """Training and evaluation entry point yielding scores.

    Resolves some configuration attributes, creates environments, graph, and
    training loop. By default, assigns all operations to the CPU.

    Args:
        config: Object providing configurations via attributes.
        env_processes: Whether to step environments in separate processes.

    Yields:
        Evaluation scores.
    """
    tf.reset_default_graph()
    if config.update_every % config.num_agents:
        tf.logging.warn('Number of agents should divide episodes per update.')
    with tf.device('/gpu:0'):
        batch_env = utility.define_batch_env(
                lambda: _create_environment(config),
                config.num_agents, env_processes)
        graph = utility.define_simulation_graph(
                batch_env, config.algorithm, config)
        loop = _define_loop(
                graph, config.logdir,
                config.update_every * config.max_length,
                config.eval_episodes * config.max_length)
        total_steps = int(
                config.steps / config.update_every *
                (config.update_every + config.eval_episodes))
    # Exclude episode related variables since the Python state of environments is
    # not checkpointed and thus new episodes start after resuming.
    saver = utility.define_saver(exclude=(r'.*_temporary/.*',))
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        utility.initialize_variables(sess, saver, config.logdir)
        for score in loop.run(sess, saver, total_steps):
            yield score
    batch_env.close()
```



### configs.py

In this file, there are few example configurations using the PPO algorithm.

```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import sys
sys.path.append('/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_envs/agents')
sys.path.append('/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_envs')
sys.path.append('/home/hadoop/zyc/bullet3-master/examples/pybullet/gym')
import ppo
import networks

# from . import ppo
# from . import networks
from pybullet_envs.bullet import minitaur_gym_env
from pybullet_envs.bullet import minitaur_duck_gym_env
from pybullet_envs.bullet import minitaur_env_randomizer
import pybullet_envs.bullet.minitaur_gym_env as minitaur_gym_env
import pybullet_envs
import tensorflow as tf

def default():
    """Default configuration for PPO."""
    # General
    algorithm = ppo.PPOAlgorithm
    num_agents = 30
    eval_episodes = 30
    use_gpu = False
    # Network
    network = networks.feed_forward_gaussian
    weight_summaries = dict(
            all=r'.*',
            policy=r'.*/policy/.*',
            value=r'.*/value/.*')
    policy_layers = 200, 100
    value_layers = 200, 100
    init_mean_factor = 0.1
    init_logstd = -1
    # Optimization
    update_every = 30
    update_epochs = 25
    optimizer = tf.train.AdamOptimizer
    update_epochs_policy = 64
    update_epochs_value = 64
    learning_rate = 1e-4  
    # Losses
    discount = 0.995
    kl_target = 1e-2
    kl_cutoff_factor = 2
    kl_cutoff_coef = 1000
    kl_init_penalty = 1
    return locals()
```

A pendulum example

```python
def pybullet_pendulum():
    locals().update(default())
    env = 'InvertedPendulumBulletEnv-v0'
    max_length = 200
    steps = 5e7  # 50M
    return locals()
```

A minitaur example

```python
def pybullet_minitaur():
    """Configuration specific to minitaur_gym_env.MinitaurBulletEnv class."""
    locals().update(default())
    randomizer = (minitaur_env_randomizer.MinitaurEnvRandomizer())
    env = functools.partial(
            minitaur_gym_env.MinitaurBulletEnv,
            accurate_motor_model_enabled=True,
            motor_overheat_protection=True,
            pd_control_enabled=True,
            env_randomizer=randomizer,
            render=False)
    max_length = 1000
    steps = 3e7  # 30M
    # tf.logging.info('locals of pybullet_minitaur ===================================== {}.'.format(locals()))
    return locals()
```

Every example is build on the default configuration.

If we take a look of the `locals()` in `pybullet_minitaur()` , it's something like this.

```python
{
    'optimizer': < class 'tensorflow.python.training.adam.AdamOptimizer' > ,
    'policy_layers': (200, 100),
    'randomizer': < pybullet_envs.bullet.minitaur_env_randomizer.MinitaurEnvRandomizer object at 0x7fce9ae307f0 > ,
    'update_every': 30,
    'env': functools.partial( < class 'pybullet_envs.bullet.minitaur_gym_env.MinitaurBulletEnv' > , motor_overheat_protection = True, accurate_motor_model_enabled = True, pd_control_enabled = True, render = False, env_randomizer = < pybullet_envs.bullet.minitaur_env_randomizer.MinitaurEnvRandomizer object at 0x7fce9ae307f0 > ),
    'algorithm': < class 'ppo.algorithm.PPOAlgorithm' > ,
    'network': < function feed_forward_gaussian at 0x7fce9bbcc268 > ,
    'value_layers': (200, 100),
    'init_logstd': -1,
    'use_gpu': True,
    'learning_rate': 0.0001,
    'update_epochs_policy': 64,
    'num_agents': 30,
    'kl_target': 0.01,
    'max_length': 1000,
    'update_epochs_value': 64,
    'kl_cutoff_coef': 1000,
    'kl_cutoff_factor': 2,
    'kl_init_penalty': 1,
    'discount': 0.995,
    'update_epochs': 25,
    'init_mean_factor': 0.1,
    'steps': 30000000.0,
    'eval_episodes': 30,
    'weight_summaries': {
        'all': '.*',
        'value': '.*/value/.*',
        'policy': '.*/policy/.*'
    }
}
```

# MinitaurBulletEnv

## environment location

The env is defined in [pybullet_envs/bullet/minitaur_gym_env.py](https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/bullet/minitaur.py)

Here is an example of how to import it.

```python
import pybullet_envs.bullet.minitaur_gym_env as minitaur_gym_env
minitaur_gym_env.MinitaurBulletEnv
```

## figuring out how it works

### global variables

There are 8 motors and 4 legs.

```python
INIT_POSITION = [0, 0, .2]
INIT_ORIENTATION = [0, 0, 0, 1]
KNEE_CONSTRAINT_POINT_RIGHT = [0, 0.005, 0.2]
KNEE_CONSTRAINT_POINT_LEFT = [0, 0.01, 0.2]
OVERHEAT_SHUTDOWN_TORQUE = 2.45
OVERHEAT_SHUTDOWN_TIME = 1.0
LEG_POSITION = ["front_left", "back_left", "front_right", "back_right"]
MOTOR_NAMES = [
    "motor_front_leftL_joint", "motor_front_leftR_joint",
    "motor_back_leftL_joint", "motor_back_leftR_joint",
    "motor_front_rightL_joint", "motor_front_rightR_joint",
    "motor_back_rightL_joint", "motor_back_rightR_joint"
]
LEG_LINK_ID = [2, 3, 5, 6, 8, 9, 11, 12, 15, 16, 18, 19, 21, 22, 24, 25]
MOTOR_LINK_ID = [1, 4, 7, 10, 14, 17, 20, 23]
FOOT_LINK_ID = [3, 6, 9, 12, 16, 19, 22, 25]
BASE_LINK_ID = -1
```

### The Minitaur class

```python
class Minitaur(object):
  """The minitaur class that simulates a quadruped robot from Ghost Robotics.
  """

  def __init__(self,
               pybullet_client,
               urdf_root= os.path.join(os.path.dirname(__file__),"../data"),
               time_step=0.01,
               self_collision_enabled=False,
               motor_velocity_limit=np.inf,
               pd_control_enabled=False,
               accurate_motor_model_enabled=False,
               motor_kp=1.0,
               motor_kd=0.02,
               torque_control_enabled=False,
               motor_overheat_protection=False,
               on_rack=False,
               kd_for_pd_controllers=0.3):
    """Constructs a minitaur and reset it to the initial states.
    Args:
      pybullet_client: The instance of BulletClient to manage different
        simulations.
      urdf_root: The path to the urdf folder.
      time_step: The time step of the simulation.
      self_collision_enabled: Whether to enable self collision.
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
    self._urdf_root = urdf_root  # urdf_root: The path to the urdf folder.
    self._self_collision_enabled = self_collision_enabled
    self._motor_velocity_limit = motor_velocity_limit
    self._pd_control_enabled = pd_control_enabled  # can't understand
    self._motor_direction = [-1, -1, -1, -1, 1, 1, 1, 1]
    self._observed_motor_torques = np.zeros(self.num_motors)  # torque is  a force that produces or tends to produce rotation or torsion, an automobile engine delivers torque to the drive shaft
    self._applied_motor_torques = np.zeros(self.num_motors)
    self._max_force = 3.5
    self._accurate_motor_model_enabled = accurate_motor_model_enabled
    self._torque_control_enabled = torque_control_enabled
    self._motor_overheat_protection = motor_overheat_protection
    self._on_rack = on_rack  # can't understand
    if self._accurate_motor_model_enabled:
      self._kp = motor_kp
      self._kd = motor_kd
      self._motor_model = motor.MotorModel(
          torque_control_enabled=self._torque_control_enabled,
          kp=self._kp,  # motor_kp: proportional gain for the accurate motor model
          kd=self._kd)  # motor_kd: derivative gain for the acurate motor model
    elif self._pd_control_enabled:
      self._kp = 8
      self._kd = kd_for_pd_controllers
    else:
      self._kp = 1
      self._kd = 1
    self.time_step = time_step
    self.Reset()

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

  def _BuildJointNameToIdDict(self):
    num_joints = self._pybullet_client.getNumJoints(self.quadruped)
    self._joint_name_to_id = {}
    for i in range(num_joints):
      joint_info = self._pybullet_client.getJointInfo(self.quadruped, i)
      self._joint_name_to_id[joint_info[1].decode("UTF-8")] = joint_info[0]

  def _BuildMotorIdList(self):
    self._motor_id_list = [
        self._joint_name_to_id[motor_name] for motor_name in MOTOR_NAMES
    ]

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
      self._BuildJointNameToIdDict()
      self._BuildMotorIdList()
      self._RecordMassInfoFromURDF()
      self.ResetPose(add_constraint=True)
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

  def _SetMotorTorqueById(self, motor_id, torque):
    self._pybullet_client.setJointMotorControl2(
        bodyIndex=self.quadruped,
        jointIndex=motor_id,
        controlMode=self._pybullet_client.TORQUE_CONTROL,
        force=torque)

  def _SetDesiredMotorAngleById(self, motor_id, desired_angle):
    self._pybullet_client.setJointMotorControl2(
        bodyIndex=self.quadruped,
        jointIndex=motor_id,
        controlMode=self._pybullet_client.POSITION_CONTROL,
        targetPosition=desired_angle,
        positionGain=self._kp,
        velocityGain=self._kd,
        force=self._max_force)

  def _SetDesiredMotorAngleByName(self, motor_name, desired_angle):
    self._SetDesiredMotorAngleById(self._joint_name_to_id[motor_name],
                                   desired_angle)

  def ResetPose(self, add_constraint):
    """Reset the pose of the minitaur.
    Args:
      add_constraint: Whether to add a constraint at the joints of two feet.
    """
    for i in range(self.num_legs):
      self._ResetPoseForLeg(i, add_constraint)

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

  def GetBasePosition(self):
    """Get the position of minitaur's base.
    Returns:
      The position of minitaur's base.
    """
    position, _ = (
        self._pybullet_client.getBasePositionAndOrientation(self.quadruped))
    return position

  def GetBaseOrientation(self):
    """Get the orientation of minitaur's base, represented as quaternion.
    Returns:
      The orientation of minitaur's base.
    """
    _, orientation = (
        self._pybullet_client.getBasePositionAndOrientation(self.quadruped))
    return orientation

  def GetActionDimension(self):
    """Get the length of the action list.
    Returns:
      The length of the action list.
    """
    return self.num_motors

  def GetObservationUpperBound(self):
    """Get the upper bound of the observation.
    Returns:
      The upper bound of an observation. See GetObservation() for the details
        of each element of an observation.
    """
    upper_bound = np.array([0.0] * self.GetObservationDimension())
    upper_bound[0:self.num_motors] = math.pi  # Joint angle.
    upper_bound[self.num_motors:2 * self.num_motors] = (
        motor.MOTOR_SPEED_LIMIT)  # Joint velocity.
    upper_bound[2 * self.num_motors:3 * self.num_motors] = (
        motor.OBSERVED_TORQUE_LIMIT)  # Joint torque.
    upper_bound[3 * self.num_motors:] = 1.0  # Quaternion of base orientation.
    return upper_bound

  def GetObservationLowerBound(self):
    """Get the lower bound of the observation."""
    return -self.GetObservationUpperBound()

  def GetObservationDimension(self):
    """Get the length of the observation list.
    Returns:
      The length of the observation list.
    """
    return len(self.GetObservation())

  def GetObservation(self):
    """Get the observations of minitaur.
    It includes the angles, velocities, torques and the orientation of the base.
    Returns:
      The observation list. observation[0:8] are motor angles. observation[8:16]
      are motor velocities, observation[16:24] are motor torques.
      observation[24:28] is the orientation of the base, in quaternion form.
    """
    observation = []
    observation.extend(self.GetMotorAngles().tolist())
    observation.extend(self.GetMotorVelocities().tolist())
    observation.extend(self.GetMotorTorques().tolist())
    observation.extend(list(self.GetBaseOrientation()))
    return observation

  def ApplyAction(self, motor_commands):
    """Set the desired motor angles to the motors of the minitaur.
    The desired motor angles are clipped based on the maximum allowed velocity.
    If the pd_control_enabled is True, a torque is calculated according to
    the difference between current and desired joint angle, as well as the joint
    velocity. This torque is exerted to the motor. For more information about
    PD control, please refer to: https://en.wikipedia.org/wiki/PID_controller.
    Args:
      motor_commands: The eight desired motor angles.
    """
    if self._motor_velocity_limit < np.inf:
      current_motor_angle = self.GetMotorAngles()
      motor_commands_max = (
          current_motor_angle + self.time_step * self._motor_velocity_limit)
      motor_commands_min = (
          current_motor_angle - self.time_step * self._motor_velocity_limit)
      motor_commands = np.clip(motor_commands, motor_commands_min,
                               motor_commands_max)

    if self._accurate_motor_model_enabled or self._pd_control_enabled:
      q = self.GetMotorAngles()
      qdot = self.GetMotorVelocities()
      if self._accurate_motor_model_enabled:
        actual_torque, observed_torque = self._motor_model.convert_to_torque(
            motor_commands, q, qdot)
        if self._motor_overheat_protection:
          for i in range(self.num_motors):
            if abs(actual_torque[i]) > OVERHEAT_SHUTDOWN_TORQUE:
              self._overheat_counter[i] += 1
            else:
              self._overheat_counter[i] = 0
            if (self._overheat_counter[i] >
                OVERHEAT_SHUTDOWN_TIME / self.time_step):
              self._motor_enabled_list[i] = False

        # The torque is already in the observation space because we use
        # GetMotorAngles and GetMotorVelocities.
        self._observed_motor_torques = observed_torque

        # Transform into the motor space when applying the torque.
        self._applied_motor_torque = np.multiply(actual_torque,
                                                 self._motor_direction)

        for motor_id, motor_torque, motor_enabled in zip(
            self._motor_id_list, self._applied_motor_torque,
            self._motor_enabled_list):
          if motor_enabled:
            self._SetMotorTorqueById(motor_id, motor_torque)
          else:
            self._SetMotorTorqueById(motor_id, 0)
      else:
        torque_commands = -self._kp * (q - motor_commands) - self._kd * qdot

        # The torque is already in the observation space because we use
        # GetMotorAngles and GetMotorVelocities.
        self._observed_motor_torques = torque_commands

        # Transform into the motor space when applying the torque.
        self._applied_motor_torques = np.multiply(self._observed_motor_torques,
                                                  self._motor_direction)

        for motor_id, motor_torque in zip(self._motor_id_list,
                                          self._applied_motor_torques):
          self._SetMotorTorqueById(motor_id, motor_torque)
    else:
      motor_commands_with_direction = np.multiply(motor_commands,
                                                  self._motor_direction)
      for motor_id, motor_command_with_direction in zip(
          self._motor_id_list, motor_commands_with_direction):
        self._SetDesiredMotorAngleById(motor_id, motor_command_with_direction)

  def GetMotorAngles(self):
    """Get the eight motor angles at the current moment.
    Returns:
      Motor angles.
    """
    motor_angles = [
        self._pybullet_client.getJointState(self.quadruped, motor_id)[0]
        for motor_id in self._motor_id_list
    ]
    motor_angles = np.multiply(motor_angles, self._motor_direction)
    return motor_angles

  def GetMotorVelocities(self):
    """Get the velocity of all eight motors.
    Returns:
      Velocities of all eight motors.
    """
    motor_velocities = [
        self._pybullet_client.getJointState(self.quadruped, motor_id)[1]
        for motor_id in self._motor_id_list
    ]
    motor_velocities = np.multiply(motor_velocities, self._motor_direction)
    return motor_velocities

  def GetMotorTorques(self):
    """Get the amount of torques the motors are exerting.
    Returns:
      Motor torques of all eight motors.
    """
    if self._accurate_motor_model_enabled or self._pd_control_enabled:
      return self._observed_motor_torques
    else:
      motor_torques = [
          self._pybullet_client.getJointState(self.quadruped, motor_id)[3]
          for motor_id in self._motor_id_list
      ]
      motor_torques = np.multiply(motor_torques, self._motor_direction)
    return motor_torques

  def ConvertFromLegModel(self, actions):
    """Convert the actions that use leg model to the real motor actions.
    Args:
      actions: The theta, phi of the leg model.
    Returns:
      The eight desired motor angles that can be used in ApplyActions().
    """

    motor_angle = copy.deepcopy(actions)
    scale_for_singularity = 1
    offset_for_singularity = 1.5
    half_num_motors = int(self.num_motors / 2)
    quater_pi = math.pi / 4
    for i in range(self.num_motors):
      action_idx = i // 2
      forward_backward_component = (-scale_for_singularity * quater_pi * (
          actions[action_idx + half_num_motors] + offset_for_singularity))
      extension_component = (-1)**i * quater_pi * actions[action_idx]
      if i >= half_num_motors:
        extension_component = -extension_component
      motor_angle[i] = (
          math.pi + forward_backward_component + extension_component)
    return motor_angle

  def GetBaseMassFromURDF(self):
    """Get the mass of the base from the URDF file."""
    return self._base_mass_urdf

  def GetLegMassesFromURDF(self):
    """Get the mass of the legs from the URDF file."""
    return self._leg_masses_urdf

  def SetBaseMass(self, base_mass):
    self._pybullet_client.changeDynamics(
        self.quadruped, BASE_LINK_ID, mass=base_mass)

  def SetLegMasses(self, leg_masses):
    """Set the mass of the legs.
    A leg includes leg_link and motor. All four leg_links have the same mass,
    which is leg_masses[0]. All four motors have the same mass, which is
    leg_mass[1].
    Args:
      leg_masses: The leg masses. leg_masses[0] is the mass of the leg link.
        leg_masses[1] is the mass of the motor.
    """
    for link_id in LEG_LINK_ID:
      self._pybullet_client.changeDynamics(
          self.quadruped, link_id, mass=leg_masses[0])
    for link_id in MOTOR_LINK_ID:
      self._pybullet_client.changeDynamics(
          self.quadruped, link_id, mass=leg_masses[1])

  def SetFootFriction(self, foot_friction):
    """Set the lateral friction of the feet.
    Args:
      foot_friction: The lateral friction coefficient of the foot. This value is
        shared by all four feet.
    """
    for link_id in FOOT_LINK_ID:
      self._pybullet_client.changeDynamics(
          self.quadruped, link_id, lateralFriction=foot_friction)

  def SetBatteryVoltage(self, voltage):
    if self._accurate_motor_model_enabled:
      self._motor_model.set_voltage(voltage)

  def SetMotorViscousDamping(self, viscous_damping):
    if self._accurate_motor_model_enabled:
      self._motor_model.set_viscous_damping(viscous_damping)
```


# pybullet

[GitHub repository](https://github.com/bulletphysics/bullet3)

[BulletQuickstart.pdf on GitHub](https://github.com/bulletphysics/bullet3/blob/master/docs/BulletQuickstart.pdf)

[Bullet_User_Manual.pdf on GitHub](https://github.com/bulletphysics/bullet3/blob/master/docs/Bullet_User_Manual.pdf)

[Offical Site](http://bulletphysics.org/)

[cuPyBullet Quickstart Guide on Google docs](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA)

# Algorithm

## PPO algorithm

PPO algorithm is the abbreviation of Proximal Policy Optimization algorithms, which is proposed in 2017. This paper can download from [arxiv](https://arxiv.org/abs/1707.06347).

A [blog](https://blog.openai.com/openai-baselines-ppo/) on openai explain it.

# Remaining questions

## 1. `config` flag

### Q: How does the `config` flag work?

How do the flags defined in the following code work? What kind of connection is between the `config` flag and the `pybullet_pendulum` function in `config.py`. Is it simply call the function which have the same name to the `config` flag?

```python
# tf.reset_default_graph()
with tf.device('/gpu:0'):
#     tf.reset_default_graph()
    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_string(
        'logdir', 'pendulum', #None
        'Base directory to store logs.')
    tf.app.flags.DEFINE_string(
        'timestamp', datetime.datetime.now().strftime('%Y%m%dT%H%M%S'),
        'Sub directory to store logs.')
    tf.app.flags.DEFINE_string(
        'config', 'pybullet_pendulum', #None
        'Configuration to execute.')
    tf.app.flags.DEFINE_boolean(
        'env_processes', True,
        'Step environments in separate processes to circumvent the GIL.')
    tf.app.run()#Runs the program with an optional 'main' function and 'argv' list.
```

## 2. `return locals()`

### Q: What does `return locals()` mean?

In the `config.py` file, the `pybullet_pendulum` function is defined as below.

```python
def pybullet_pendulum():
    locals().update(default())
    env = 'InvertedPendulumBulletEnv-v0'
    max_length = 200
    steps = 5e7  # 50M
    return locals()
```

### A: `locals()` is a python build-in functions

According to [the python official documents](https://docs.python.org/3/library/functions.html#locals), `locals()` is the function that update and return a dictionary representing the current local symbol table. Free variables are returned by `locals()` when it is called in function blocks, but not in class blocks. 

> **Note**: The contents of this dictionary should not be modified; changes may not affect the values of local and free variables used by the interpreter.

For example, if we run locals() in a single cell in jupyter notebook, it may give out something like this:

```python
{'In': ['', 'locals()'],
 'Out': {},
 '_': '',
 '__': '',
 '___': '',
 '__builtin__': <module 'builtins' (built-in)>,
 '__builtins__': <module 'builtins' (built-in)>,
 '__doc__': 'Automatically created module for IPython interactive environment',
 '__loader__': None,
 '__name__': '__main__',
 '__package__': None,
 '__spec__': None,
 '_dh': ['/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_envs/agents'],
 '_i': '',
 '_i1': 'locals()',
 '_ih': ['', 'locals()'],
 '_ii': '',
 '_iii': '',
 '_oh': {},
 'exit': <IPython.core.autocall.ZMQExitAutocall at 0x7f70f51a1198>,
 'get_ipython': <bound method InteractiveShell.get_ipython of <ipykernel.zmqshell.ZMQInteractiveShell object at 0x7f70f61d26a0>>,
 'quit': <IPython.core.autocall.ZMQExitAutocall at 0x7f70f51a1198>}
```

## 3. urdf file

### Q: What's urdf file?

In [/examples/pybullet/gym/pybullet_envs/bullet/minitaur.py](https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/bullet/minitaur.py) the definition of Minitaur class mentioned the urdf.
> urdf_root: The path to the urdf folder.

### A: ROS

A [wiki](http://wiki.ros.org/urdf) on ROS may explain it.

> This package contains a C++ parser for the Unified Robot Description Format (URDF), which is an XML format for representing a robot model. 


































