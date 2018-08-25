---
title: "代码分析"
permalink: /pybullet_minitaur/
last_modified_at: 2018-08-07T10:34:04+08:00
toc: true
toc_sticky: true
---

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

```bash
python3 -m pybullet_envs.agents.visualize_ppo --logdir=minitaur/20180816T174103-pybullet_minitaur --outdir=20180816T174103-pybullet_minitaur_video
```

```python
python3 -m pybullet_envs.agents.visualize_ppo --logdir=minitaur2/20180817T002013-pybullet_minitaur --outdir=20180817T002013-pybullet_minitaur_video
```



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

# Minitaur

## MinitaurBullet environment `minitaur_gym_env.py`

The env is defined in [pybullet_envs/bullet/minitaur_gym_env.py](https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/bullet/minitaur_gym_env.py)

Here is an example of how to import it.

```python
import pybullet_envs.bullet.minitaur_gym_env as minitaur_gym_env
minitaur_gym_env.MinitaurBulletEnv
```

### global varibales

```python
NUM_SUBSTEPS = 5
NUM_MOTORS = 8
MOTOR_ANGLE_OBSERVATION_INDEX = 0
MOTOR_VELOCITY_OBSERVATION_INDEX = MOTOR_ANGLE_OBSERVATION_INDEX + NUM_MOTORS  # 8
MOTOR_TORQUE_OBSERVATION_INDEX = MOTOR_VELOCITY_OBSERVATION_INDEX + NUM_MOTORS  # 16
BASE_ORIENTATION_OBSERVATION_INDEX = MOTOR_TORQUE_OBSERVATION_INDEX + NUM_MOTORS  # 24
ACTION_EPS = 0.01
OBSERVATION_EPS = 0.01
RENDER_HEIGHT = 720
RENDER_WIDTH = 960
```

TODO: what does the EPS means?

### class MinitaurBulletEnv(gym.Env)

```python
class MinitaurBulletEnv(gym.Env):
  """The gym environment for the minitaur.
  It simulates the locomotion of a minitaur, a quadruped robot. The state space
  include the angles, velocities and torques for all the motors and the action
  space is the desired motor angle for each motor. The reward function is based
  on how far the minitaur walks in 1000 steps and penalizes the energy
  expenditure.
  """
  metadata = {
      "render.modes": ["human", "rgb_array"],
      "video.frames_per_second": 50
  }

  def __init__(self,
               urdf_root=pybullet_data.getDataPath(),
               action_repeat=1,
               distance_weight=1.0,
               energy_weight=0.005,
               shake_weight=0.0,
               drift_weight=0.0,
               distance_limit=float("inf"),
               observation_noise_stdev=0.0,
               self_collision_enabled=True,
               motor_velocity_limit=np.inf,
               pd_control_enabled=False,#not needed to be true if accurate motor model is enabled (has its own better PD)
               leg_model_enabled=True,
               accurate_motor_model_enabled=True,
               motor_kp=1.0,
               motor_kd=0.02,
               torque_control_enabled=False,
               motor_overheat_protection=True,
               hard_reset=True,
               on_rack=False,
               render=False,
               kd_for_pd_controllers=0.3,
               env_randomizer=minitaur_env_randomizer.MinitaurEnvRandomizer()):
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
    self._time_step = 0.01
    self._action_repeat = action_repeat
    self._num_bullet_solver_iterations = 300
    self._urdf_root = urdf_root
    self._self_collision_enabled = self_collision_enabled
    self._motor_velocity_limit = motor_velocity_limit
    self._observation = []
    self._env_step_counter = 0
    self._is_render = render
    self._last_base_position = [0, 0, 0]
    self._distance_weight = distance_weight
    self._energy_weight = energy_weight
    self._drift_weight = drift_weight
    self._shake_weight = shake_weight
    self._distance_limit = distance_limit
    self._observation_noise_stdev = observation_noise_stdev
    self._action_bound = 1
    self._pd_control_enabled = pd_control_enabled
    self._leg_model_enabled = leg_model_enabled
    self._accurate_motor_model_enabled = accurate_motor_model_enabled
    self._motor_kp = motor_kp
    self._motor_kd = motor_kd
    self._torque_control_enabled = torque_control_enabled
    self._motor_overheat_protection = motor_overheat_protection
    self._on_rack = on_rack
    self._cam_dist = 1.0
    self._cam_yaw = 0
    self._cam_pitch = -30
    self._hard_reset = True
    self._kd_for_pd_controllers = kd_for_pd_controllers
    self._last_frame_time = 0.0
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

    self._seed()
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

  def set_env_randomizer(self, env_randomizer):
    self._env_randomizer = env_randomizer

  def configure(self, args):
    self._args = args

  def _reset(self):
    if self._hard_reset:
      self._pybullet_client.resetSimulation()
      self._pybullet_client.setPhysicsEngineParameter(
          numSolverIterations=int(self._num_bullet_solver_iterations))
      self._pybullet_client.setTimeStep(self._time_step)
      plane = self._pybullet_client.loadURDF("%s/plane.urdf" % self._urdf_root)
      self._pybullet_client.changeVisualShape(plane,-1,rgbaColor=[1,1,1,0.9])
      self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_PLANAR_REFLECTION,0)
      self._pybullet_client.setGravity(0, 0, -10)
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
        self._cam_dist, self._cam_yaw, self._cam_pitch, [0, 0, 0])
    if not self._torque_control_enabled:
      for _ in range(100):
        if self._pd_control_enabled or self._accurate_motor_model_enabled:
          self.minitaur.ApplyAction([math.pi / 2] * 8)
        self._pybullet_client.stepSimulation()
    return self._noisy_observation()

  def _seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def _transform_action_to_motor_command(self, action):
    if self._leg_model_enabled:
      for i, action_component in enumerate(action):
        if not (-self._action_bound - ACTION_EPS <= action_component <=
                self._action_bound + ACTION_EPS):
          raise ValueError(
              "{}th action {} out of bounds.".format(i, action_component))
      action = self.minitaur.ConvertFromLegModel(action)
    return action

  def _step(self, action):
    """Step forward the simulation, given the action.
    Args:
      action: A list of desired motor angles for eight motors.
    Returns:
      observations: The angles, velocities and torques of all motors.
      reward: The reward for the current state-action pair.
      done: Whether the episode has ended.
      info: A dictionary that stores diagnostic information.
    Raises:
      ValueError: The action dimension is not the same as the number of motors.
      ValueError: The magnitude of actions is out of bounds.
    """
    if self._is_render:
      # Sleep, otherwise the computation takes less time than real time,
      # which will make the visualization like a fast-forward video.
      time_spent = time.time() - self._last_frame_time
      self._last_frame_time = time.time()
      time_to_sleep = self._action_repeat * self._time_step - time_spent
      if time_to_sleep > 0:
        time.sleep(time_to_sleep)
      base_pos = self.minitaur.GetBasePosition()
      camInfo = self._pybullet_client.getDebugVisualizerCamera()
      curTargetPos = camInfo[11]
      distance=camInfo[10]
      yaw = camInfo[8]
      pitch=camInfo[9]
      targetPos = [0.95*curTargetPos[0]+0.05*base_pos[0],0.95*curTargetPos[1]+0.05*base_pos[1],curTargetPos[2]]
           
           
      self._pybullet_client.resetDebugVisualizerCamera(
          distance, yaw, pitch, base_pos)
    action = self._transform_action_to_motor_command(action)
    for _ in range(self._action_repeat):
      self.minitaur.ApplyAction(action)
      self._pybullet_client.stepSimulation()

    self._env_step_counter += 1
    reward = self._reward()
    done = self._termination()
    return np.array(self._noisy_observation()), reward, done, {}

  def _render(self, mode="rgb_array", close=False):
    if mode != "rgb_array":
      return np.array([])
    base_pos = self.minitaur.GetBasePosition()
    view_matrix = self._pybullet_client.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=base_pos,
        distance=self._cam_dist,
        yaw=self._cam_yaw,
        pitch=self._cam_pitch,
        roll=0,
        upAxisIndex=2)
    proj_matrix = self._pybullet_client.computeProjectionMatrixFOV(
        fov=60, aspect=float(RENDER_WIDTH)/RENDER_HEIGHT,
        nearVal=0.1, farVal=100.0)
    (_, _, px, _, _) = self._pybullet_client.getCameraImage(
        width=RENDER_WIDTH, height=RENDER_HEIGHT, viewMatrix=view_matrix,
        projectionMatrix=proj_matrix, renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
    rgb_array = np.array(px)
    rgb_array = rgb_array[:, :, :3]
    return rgb_array

  def get_minitaur_motor_angles(self):
    """Get the minitaur's motor angles.
    Returns:
      A numpy array of motor angles.
    """
    return np.array(
        self._observation[MOTOR_ANGLE_OBSERVATION_INDEX:
                          MOTOR_ANGLE_OBSERVATION_INDEX + NUM_MOTORS])

  def get_minitaur_motor_velocities(self):
    """Get the minitaur's motor velocities.
    Returns:
      A numpy array of motor velocities.
    """
    return np.array(
        self._observation[MOTOR_VELOCITY_OBSERVATION_INDEX:
                          MOTOR_VELOCITY_OBSERVATION_INDEX + NUM_MOTORS])

  def get_minitaur_motor_torques(self):
    """Get the minitaur's motor torques.
    Returns:
      A numpy array of motor torques.
    """
    return np.array(
        self._observation[MOTOR_TORQUE_OBSERVATION_INDEX:
                          MOTOR_TORQUE_OBSERVATION_INDEX + NUM_MOTORS])

  def get_minitaur_base_orientation(self):
    """Get the minitaur's base orientation, represented by a quaternion.
    Returns:
      A numpy array of minitaur's orientation.
    """
    return np.array(self._observation[BASE_ORIENTATION_OBSERVATION_INDEX:])

  def is_fallen(self):
    """Decide whether the minitaur has fallen.
    If the up directions between the base and the world is larger (the dot
    product is smaller than 0.85) or the base is very low on the ground
    (the height is smaller than 0.13 meter), the minitaur is considered fallen.
    Returns:
      Boolean value that indicates whether the minitaur has fallen.
    """
    orientation = self.minitaur.GetBaseOrientation()
    rot_mat = self._pybullet_client.getMatrixFromQuaternion(orientation)
    local_up = rot_mat[6:]
    pos = self.minitaur.GetBasePosition()
    return (np.dot(np.asarray([0, 0, 1]), np.asarray(local_up)) < 0.85 or
            pos[2] < 0.13)

  def _termination(self):
    position = self.minitaur.GetBasePosition()
    distance = math.sqrt(position[0]**2 + position[1]**2)
    return self.is_fallen() or distance > self._distance_limit

  def _reward(self):
    current_base_position = self.minitaur.GetBasePosition()
    forward_reward = current_base_position[0] - self._last_base_position[0]
    drift_reward = -abs(current_base_position[1] - self._last_base_position[1])
    shake_reward = -abs(current_base_position[2] - self._last_base_position[2])
    self._last_base_position = current_base_position
    energy_reward = np.abs(
        np.dot(self.minitaur.GetMotorTorques(),
               self.minitaur.GetMotorVelocities())) * self._time_step
    reward = (
        self._distance_weight * forward_reward -
        self._energy_weight * energy_reward + self._drift_weight * drift_reward
        + self._shake_weight * shake_reward)
    self._objectives.append(
        [forward_reward, energy_reward, drift_reward, shake_reward])
    return reward

  def get_objectives(self):
    return self._objectives

  def _get_observation(self):
    self._observation = self.minitaur.GetObservation()
    return self._observation

  def _noisy_observation(self):
    self._get_observation()
    observation = np.array(self._observation)
    if self._observation_noise_stdev > 0:
      observation += (np.random.normal(
          scale=self._observation_noise_stdev, size=observation.shape) *
                      self.minitaur.GetObservationUpperBound())
    return observation

  if parse_version(gym.__version__)>=parse_version('0.9.6'):
    render = _render
    reset = _reset
    seed = _seed
    step = _step
```





## Minitaur robot `minitaur.py`

The robot is defined in [pybullet_envs/bullet/minitaur.py](https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/bullet/minitaur.py)

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

## minitaur.urdf

[minitaur.urdf](https://github.com/bulletphysics/bullet3/blob/master/data/quadruped/minitaur.urdf)

```xml
<?xml version="0.0" ?>
<!-- ======================================================================= -->
<!--LICENSE:                                                                 -->
<!--Copyright (c) 2017, Erwin Coumans                                        -->
<!--Google Inc.                                                              -->
<!--All rights reserved.                                                     -->
<!--                                                                         -->
<!--Redistribution and use in source and binary forms, with or without       -->
<!--modification, are permitted provided that the following conditions are   -->
<!--met:                                                                     -->
<!--                                                                         -->
<!--1. Redistributions or derived work must retain this copyright notice,    -->
<!--   this list of conditions and the following disclaimer.                 -->
<!--                                                                         -->
<!--2. Redistributions in binary form must reproduce the above copyright     -->
<!--   notice, this list of conditions and the following disclaimer in the   -->
<!--   documentation and/or other materials provided with the distribution.  -->
<!--                                                                         -->
<!--THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS  -->
<!--IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,-->
<!--THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR   -->
<!--PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR        -->
<!--CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,    -->
<!--EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,      -->
<!--PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR       -->
<!--PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF   -->
<!--LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING     -->
<!--NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS       -->
<!--SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.             -->

<robot name="quadruped">
  <link name="base_chassis_link">
    <visual>
      <geometry>
        <box size=".33 0.10 .07"/>
      </geometry>
      <material name="black">
        <color rgba="0.3 0.3 0.3 1"/>
      </material>
    </visual>
    <visual>
			<origin rpy="0 0 0" xyz="0 0.10 0"/>
      <geometry>
        <box size=".17 0.10 .05"/>
      </geometry>
      <material name="black">
        <color rgba="0.3 0.3 0.3 1"/>
      </material>
    </visual>
		<visual>
			<origin rpy="0 0 0" xyz="0 -0.10 0"/>
      <geometry>
        <box size=".17 0.10 .05"/>
      </geometry>
      <material name="black">
        <color rgba="0.3 0.3 0.3 1"/>
      </material>
    </visual>

    <collision>
      <geometry>
       <box size=".33 0.10 .07"/>
      </geometry>
    </collision>
    <collision>
			<origin rpy="0 0 0" xyz="0 0.10 0"/>
      <geometry>
       <box size=".17 0.10 .05"/>
      </geometry>
    </collision>
    <collision>
			<origin rpy="0 0 0" xyz="0 -0.10 0"/>
      <geometry>
       <box size=".17 0.10 .05"/>
      </geometry>
    </collision>

    <inertial>
      <mass value="3"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

  <link name="chassis_right">
    <visual>
			<origin rpy="0 0 0" xyz="0 0.035 0"/>
      <geometry>
       <box size=".34 0.01 .04"/>
      </geometry>
			<material name="grey">
        <color rgba="0.65 0.65 0.75 1"/>
      </material>    </visual>
    <visual>
			<origin rpy="0 0 0" xyz="0 -0.035 0"/>
      <geometry>
       <box size=".34 0.01 .04"/>
      </geometry>
			<material name="grey">
        <color rgba="0.65 0.65 0.75 1"/>
      </material>    </visual>
    <collision>
			<origin rpy="0 0 0" xyz="0 0.035 0"/>
      <geometry>
       <box size=".34 0.01 .04"/>
      </geometry>
    </collision>
    <collision>
			<origin rpy="0 0 0" xyz="0 -0.035 0"/>
      <geometry>
       <box size=".34 0.01 .04"/>
      </geometry>
    </collision>
    <inertial>
      <mass value=".1"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

  <joint name="chassis_right_center" type="fixed">
    <axis xyz="0 0 1"/>
    <parent link="base_chassis_link"/>
    <child link="chassis_right"/>
    <origin rpy="-0.0872665 0 0" xyz="0.0 -0.10 0.0"/>
    <limit effort="100" velocity="100"/>
    <joint_properties damping="0.0" friction="0.0"/>
  </joint>

  <link name="chassis_left">
    <visual>
			<origin rpy="0 0 0" xyz="0 0.035 0"/>
      <geometry>
       <box size=".34 0.01 .04"/>
      </geometry>
			<material name="grey">
        <color rgba="0.65 0.65 0.75 1"/>
      </material>    </visual>
    <visual>
			<origin rpy="0 0 0" xyz="0 -0.035 0"/>
      <geometry>
       <box size=".34 0.01 .04"/>
      </geometry>
			<material name="grey">
        <color rgba="0.65 0.65 0.75 1"/>
      </material>    </visual>
    <collision>
			<origin rpy="0 0 0" xyz="0 0.035 0"/>
      <geometry>
       <box size=".34 0.01 .04"/>
      </geometry>
    </collision>
    <collision>
			<origin rpy="0 0 0" xyz="0 -0.035 0"/>
      <geometry>
       <box size=".34 0.01 .04"/>
      </geometry>
    </collision>
    <inertial>
      <mass value=".1"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

  <joint name="chassis_left_center" type="fixed">
    <axis xyz="0 0 1"/>
    <parent link="base_chassis_link"/>
    <child link="chassis_left"/>
    <origin rpy="0.0872665 0 0" xyz="0.0 0.10 0.0"/>
    <limit effort="100" velocity="100"/>
    <joint_properties damping="0.0" friction="0.0"/>
  </joint>

  
  
  <link name="motor_front_rightR_link">
    <visual>
      <geometry>
        <mesh filename="tmotor3.obj" scale="1 1 1"/>
      </geometry>
      <material name="white">
        <color rgba="1 1 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.026" radius="0.0434"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.25"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>
  <joint name="motor_front_rightR_joint" type="continuous">
    <axis xyz="0 0 1"/>
    <parent link="chassis_right"/>
    <child link="motor_front_rightR_link"/>
    <origin rpy="1.57075 0 0" xyz="0.21 -0.025 0"/>
    <limit effort="100" velocity="100"/>
    <joint_properties damping="0.0" friction="0.0"/>
  </joint>


	<link name="motor_front_rightL_link">
    <visual>
      <geometry>
        <mesh filename="tmotor3.obj" scale="1 1 1"/>
      </geometry>
      <material name="white">
        <color rgba="1 1 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.026" radius="0.0434"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.25"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>
  <joint name="motor_front_rightL_joint" type="continuous">
    <axis xyz="0 0 1"/>
    <parent link="chassis_right"/>
    <child link="motor_front_rightL_link"/>
    <origin rpy="1.57075 0 3.141592" xyz="0.21 0.04 0"/>
    <limit effort="100" velocity="100"/>
    <joint_properties damping="0.0" friction="0.0"/>
  </joint>
  
  
  <link name="motor_front_leftL_link">
    <visual>
      <geometry>
        <mesh filename="tmotor3.obj" scale="1 1 1"/>
      </geometry>
      <material name="white">
        <color rgba="1 1 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.026" radius="0.0434"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.25"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>
  <joint name="motor_front_leftL_joint" type="continuous">
    <axis xyz="0 0 1"/>
    <parent link="chassis_left"/>
    <child link="motor_front_leftL_link"/>
    <origin rpy="1.57075 0 3.141592" xyz="0.21 0.025 0"/>
    <limit effort="100" velocity="100"/>
    <joint_properties damping="0.0" friction="0.0"/>
  </joint>


	<link name="motor_front_leftR_link">
    <visual>
      <geometry>
        <mesh filename="tmotor3.obj" scale="1 1 1"/>
      </geometry>
      <material name="white">
        <color rgba="1 1 1 1"/>
      </material>
    </visual>
    <inertial>
      <mass value="0.25"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>
  <joint name="motor_front_leftR_joint" type="continuous">
    <axis xyz="0 0 1"/>
    <parent link="chassis_left"/>
    <child link="motor_front_leftR_link"/>
    <origin rpy="1.57075 0 0" xyz="0.21 -0.04 0"/>
    <limit effort="100" velocity="100"/>
    <joint_properties damping="0.0" friction="0.0"/>
  </joint>
  
  
  <link name="motor_back_rightR_link">
    <visual>
      <geometry>
        <mesh filename="tmotor3.obj" scale="1 1 1"/>
      </geometry>
      <material name="white">
        <color rgba="1 1 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.026" radius="0.0434"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.25"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>
  <joint name="motor_back_rightR_joint" type="continuous">
    <axis xyz="0 0 1"/>
    <parent link="chassis_right"/>
    <child link="motor_back_rightR_link"/>
    <origin rpy="1.57075 0 0" xyz="-0.21 -0.025 0"/>
    <limit effort="100" velocity="100"/>
    <joint_properties damping="0.0" friction="0.0"/>
  </joint>


	<link name="motor_back_rightL_link">
    <visual>
      <geometry>
        <mesh filename="tmotor3.obj" scale="1 1 1"/>
      </geometry>
      <material name="white">
        <color rgba="1 1 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.026" radius="0.0434"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.25"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>
  <joint name="motor_back_rightL_joint" type="continuous">
    <axis xyz="0 0 1"/>
    <parent link="chassis_right"/>
    <child link="motor_back_rightL_link"/>
    <origin rpy="1.57075 0 3.141592" xyz="-0.21 0.04 0"/>
    <limit effort="100" velocity="100"/>
    <joint_properties damping="0.0" friction="0.0"/>
  </joint>
  
  
  <link name="motor_back_leftL_link">
    <visual>
      <geometry>
        <mesh filename="tmotor3.obj" scale="1 1 1"/>
      </geometry>
      <material name="white">
        <color rgba="1 1 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.026" radius="0.0434"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.25"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>
  <joint name="motor_back_leftL_joint" type="continuous">
    <axis xyz="0 0 1"/>
    <parent link="chassis_left"/>
    <child link="motor_back_leftL_link"/>
    <origin rpy="1.57075 0 3.141592" xyz="-0.21 0.025 0"/>
    <limit effort="100" velocity="100"/>
    <joint_properties damping="0.0" friction="0.0"/>
  </joint>


	<link name="motor_back_leftR_link">
    <visual>
      <geometry>
        <mesh filename="tmotor3.obj" scale="1 1 1"/>
      </geometry>
      <material name="white">
        <color rgba="1 1 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.026" radius="0.0434"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.25"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>
  <joint name="motor_back_leftR_joint" type="continuous">
    <axis xyz="0 0 1"/>
    <parent link="chassis_left"/>
    <child link="motor_back_leftR_link"/>
    <origin rpy="1.57075 0 0" xyz="-0.21 -0.04 0"/>
    <limit effort="100" velocity="100"/>
    <joint_properties damping="0.0" friction="0.0"/>
  </joint>

	<link name="upper_leg_front_rightR_link">
    <visual>
      <geometry>
        <box size=".01 0.01 .11"/>
      </geometry>
      <material name="grey">
        <color rgba="0.65 0.65 0.75 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size=".01 0.01 .11"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.05"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>
  <joint name="hip_front_rightR_link" type="fixed">
    <axis xyz="0 0 1"/>
    <parent link="motor_front_rightR_link"/>
    <child link="upper_leg_front_rightR_link"/>
    <origin rpy="-1.57075 0 0" xyz="0.0 0.06 -0.015"/>
    <limit effort="100" velocity="100"/>
    <joint_properties damping="0.0" friction="0.0"/>
  </joint>

	<link name="lower_leg_front_rightR_link">
	<contact>
		<stiffness value="10000"/>
		<damping value="10"/>
		<lateral_friction value="1"/>
  </contact>

    <visual>
    	<origin rpy="0.0 0 0" xyz="0 0 .1"/>
      <geometry>
        <box size=".01 0.01 .2"/>
      </geometry>
      <material name="grey">
        <color rgba="0.65 0.65 0.75 1"/>
      </material>
    </visual>
    <collision>
    	<origin rpy="0.0 0 0" xyz="0 0 .1"/>
      <geometry>
        <box size=".01 0.01 .2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.05"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>
  <joint name="knee_front_rightR_link" type="revolute">
    <axis xyz="0 1 0"/>
    <parent link="upper_leg_front_rightR_link"/>
    <child link="lower_leg_front_rightR_link"/>
    <origin rpy="0 0 0" xyz="0.0 0.01 .055"/>
    <limit effort="100" velocity="100"/>
    <joint_properties damping="0.0" friction="0.0"/>
  </joint>

    
  <link name="upper_leg_front_rightL_link">
    <visual>
      <geometry>
        <box size=".01 0.01 .11"/>
      </geometry>
      <material name="grey">
        <color rgba="0.65 0.65 0.75 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size=".01 0.01 .11"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.05"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>
  <joint name="motor_front_rightL_link" type="fixed">
    <axis xyz="0 0 1"/>
    <parent link="motor_front_rightL_link"/>
    <child link="upper_leg_front_rightL_link"/>
    <origin rpy="-1.57075 0 0" xyz="0.0 0.06 -0.015"/>
    <limit effort="100" velocity="100"/>
    <joint_properties damping="0.0" friction="0.0"/>
  </joint>

  

  
  <link name="lower_leg_front_rightL_link">
	<contact>
		<stiffness value="10000"/>
		<damping value="10"/>
		<lateral_friction value="1"/>
  </contact>

    <visual>
    	<origin rpy="0.0 0 0" xyz="0 0 .1"/>
      <geometry>
        <box size=".01 0.01 .198"/>
      </geometry>
      <material name="grey">
        <color rgba="0.65 0.65 0.75 1"/>
      </material>
    </visual>
    <collision>
    	<origin rpy="0.0 0 0" xyz="0 0 .1"/>
      <geometry>
        <box size=".01 0.01 .198"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.05"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>
  <joint name="knee_front_rightL_link" type="revolute">
    <axis xyz="0 1 0"/>
    <parent link="upper_leg_front_rightL_link"/>
    <child link="lower_leg_front_rightL_link"/>
    <origin rpy="0 0 0" xyz="0.0 0.01 .055"/>
    <limit effort="100" velocity="100"/>
    <joint_properties damping="0.0" friction="0.0"/>
  </joint>


	<link name="upper_leg_front_leftR_link">
    <visual>
      <geometry>
        <box size=".01 0.01 .11"/>
      </geometry>
      <material name="grey">
        <color rgba="0.65 0.65 0.75 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size=".01 0.01 .11"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.05"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>
  <joint name="hip_front_leftR_link" type="fixed">
    <axis xyz="0 0 1"/>
    <parent link="motor_front_leftR_link"/>
    <child link="upper_leg_front_leftR_link"/>
    <origin rpy="-1.57075 0 0" xyz="0.0 0.06 -0.015"/>
    <limit effort="100" velocity="100"/>
    <joint_properties damping="0.0" friction="0.0"/>
  </joint>


	<link name="lower_leg_front_leftR_link">
	<contact>
		<stiffness value="10000"/>
		<damping value="10"/>
		<lateral_friction value="1"/>
  </contact>

    <visual>
    	<origin rpy="0.0 0.0 0" xyz="0 0 .1"/>
      <geometry>
        <box size=".01 0.01 .2"/>
      </geometry>
      <material name="grey">
        <color rgba="0.65 0.65 0.75 1"/>
      </material>
    </visual>
    <collision>
    	<origin rpy="0.0 0.0 0" xyz="0 0 .1"/>
      <geometry>
        <box size=".01 0.01 .2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.05"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>
  <joint name="knee_front_leftR_link" type="revolute">
    <axis xyz="0 1 0"/>
    <parent link="upper_leg_front_leftR_link"/>
    <child link="lower_leg_front_leftR_link"/>
    <origin rpy="0 0 0" xyz="0.0 0.01 .055"/>
    <limit effort="100" velocity="100"/>
    <joint_properties damping="0.0" friction="0.0"/>
  </joint>



   <link name="upper_leg_front_leftL_link">
    <visual>
      <geometry>
        <box size=".01 0.01 .11"/>
      </geometry>
      <material name="grey">
        <color rgba="0.65 0.65 0.75 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size=".01 0.01 .11"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.05"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>
  <joint name="motor_front_leftL_link" type="fixed">
    <axis xyz="0 0 1"/>
    <parent link="motor_front_leftL_link"/>
    <child link="upper_leg_front_leftL_link"/>
    <origin rpy="-1.57075 0 0" xyz="0.0 0.06 -0.015"/>
    <limit effort="100" velocity="100"/>
    <joint_properties damping="0.0" friction="0.0"/>
  </joint>


  <link name="lower_leg_front_leftL_link">
	<contact>
		<stiffness value="10000"/>
		<damping value="10"/>
		<lateral_friction value="1"/>
  </contact>

    <visual>
    	<origin rpy="0.0 0.0 0" xyz="0 0 .1"/>
      <geometry>
        <box size=".01 0.01 .198"/>
      </geometry>
      <material name="grey">
        <color rgba="0.65 0.65 0.75 1"/>
      </material>
    </visual>
    <collision>
    	<origin rpy="0.0 0.0 0" xyz="0 0 .1"/>
      <geometry>
        <box size=".01 0.01 .198"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.05"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>
  <joint name="knee_front_leftL_link" type="revolute">
    <axis xyz="0 1 0"/>
    <parent link="upper_leg_front_leftL_link"/>
    <child link="lower_leg_front_leftL_link"/>
    <origin rpy="0 0 0" xyz="0.0 0.01 .055"/>
    <limit effort="100" velocity="100"/>
    <joint_properties damping="0.0" friction="0.0"/>
  </joint>
  
	<link name="upper_leg_back_rightR_link">
    <visual>
      <geometry>
        <box size=".01 0.01 .11"/>
      </geometry>
      <material name="grey">
        <color rgba="0.65 0.65 0.75 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size=".01 0.01 .11"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.05"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>
  <joint name="hip_rightR_link" type="fixed">
    <axis xyz="0 0 1"/>
    <parent link="motor_back_rightR_link"/>
    <child link="upper_leg_back_rightR_link"/>
    <origin rpy="-1.57075 0 0" xyz="0.0 0.06 -0.015"/>
    <limit effort="100" velocity="100"/>
    <joint_properties damping="0.0" friction="0.0"/>
  </joint>

	<link name="lower_leg_back_rightR_link">
	<contact>
		<stiffness value="10000"/>
		<damping value="10"/>
		<lateral_friction value="1"/>
  </contact>

    <visual>
    	<origin rpy="0.0 0 0" xyz="0 0 .1"/>
      <geometry>
        <box size=".01 0.01 .2032"/>
      </geometry>
      <material name="grey">
        <color rgba="0.65 0.65 0.75 1"/>
      </material>
    </visual>
    <collision>
    	<origin rpy="0.0 0 0" xyz="0 0 .1"/>
      <geometry>
        <box size=".01 0.01 .2032"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.05"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>
  <joint name="knee_back_rightR_link" type="revolute">
    <axis xyz="0 1 0"/>
    <parent link="upper_leg_back_rightR_link"/>
    <child link="lower_leg_back_rightR_link"/>
    <origin rpy="0 0 0" xyz="0.0 0.01 .055"/>
    <limit effort="100" velocity="100"/>
    <joint_properties damping="0.0" friction="0.0"/>
  </joint>

    
  <link name="upper_leg_back_rightL_link">
    <visual>
      <geometry>
        <box size=".01 0.01 .11"/>
      </geometry>
      <material name="grey">
        <color rgba="0.65 0.65 0.75 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size=".01 0.01 .11"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.05"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>
  <joint name="motor_back_rightL_link" type="fixed">
    <axis xyz="0 0 1"/>
    <parent link="motor_back_rightL_link"/>
    <child link="upper_leg_back_rightL_link"/>
    <origin rpy="-1.57075 0 0" xyz="0.0 0.06 -0.015"/>
    <limit effort="100" velocity="100"/>
    <joint_properties damping="0.0" friction="0.0"/>
  </joint>

  

  
  <link name="lower_leg_back_rightL_link">
	<contact>
		<stiffness value="10000"/>
		<damping value="10"/>
		<lateral_friction value="1"/>
  </contact>
  
    <visual>
    	<origin rpy="0.0 0 0" xyz="0 0 .1"/>
      <geometry>
        <box size=".01 0.01 .2"/>
      </geometry>
      <material name="grey">
        <color rgba="0.65 0.65 0.75 1"/>
      </material>
    </visual>
    <collision>
    	<origin rpy="0.0 0 0" xyz="0 0 .1"/>
      <geometry>
        <box size=".01 0.01 .2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.05"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>
  <joint name="knee_back_rightL_link" type="revolute">
    <axis xyz="0 1 0"/>
    <parent link="upper_leg_back_rightL_link"/>
    <child link="lower_leg_back_rightL_link"/>
    <origin rpy="0 0 0" xyz="0.0 0.01 .055"/>
    <limit effort="100" velocity="100"/>
    <joint_properties damping="0.0" friction="0.0"/>
  </joint>



	<link name="upper_leg_back_leftR_link">
    <visual>
      <geometry>
        <box size=".01 0.01 .11"/>
      </geometry>
      <material name="grey">
        <color rgba="0.65 0.65 0.75 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size=".01 0.01 .11"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.05"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>
  <joint name="hip_leftR_link" type="fixed">
    <axis xyz="0 0 1"/>
    <parent link="motor_back_leftR_link"/>
    <child link="upper_leg_back_leftR_link"/>
    <origin rpy="-1.57075 0 0" xyz="0.0 0.06 -0.015"/>
    <limit effort="100" velocity="100"/>
    <joint_properties damping="0.0" friction="0.0"/>
  </joint>

<link name="lower_leg_back_leftR_link">
	<contact>
		<stiffness value="10000"/>
		<damping value="10"/>
		<lateral_friction value="1"/>
  </contact>

    <visual>
    	<origin rpy="0.0 0 0" xyz="0 0 .1"/>
      <geometry>
        <box size=".01 0.01 .2032"/>
      </geometry>
      <material name="grey">
        <color rgba="0.65 0.65 0.75 1"/>
      </material>
    </visual>
    <collision>
    	<origin rpy="0.0 0 0" xyz="0 0 .1"/>
      <geometry>
        <box size=".01 0.01 .2032"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.05"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>
  <joint name="knee_back_leftR_link" type="revolute">
    <axis xyz="0 1 0"/>
    <parent link="upper_leg_back_leftR_link"/>
    <child link="lower_leg_back_leftR_link"/>
    <origin rpy="0 0 0" xyz="0.0 0.01 .055"/>
    <limit effort="100" velocity="100"/>
    <joint_properties damping="0.0" friction="0.0"/>
  </joint>
  
   <link name="upper_leg_back_leftL_link">
    <visual>
      <geometry>
        <box size=".01 0.01 .11"/>
      </geometry>
      <material name="grey">
        <color rgba="0.65 0.65 0.75 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size=".01 0.01 .11"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.05"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>
  <joint name="motor_back_leftL_link" type="fixed">
    <axis xyz="0 0 1"/>
    <parent link="motor_back_leftL_link"/>
    <child link="upper_leg_back_leftL_link"/>
    <origin rpy="-1.57075 0 0" xyz="0.0 0.06 -0.015"/>
    <limit effort="100" velocity="100"/>
    <joint_properties damping="0.0" friction="0.0"/>
  </joint>

 
  <link name="lower_leg_back_leftL_link">
	<contact>
		<stiffness value="10000"/>
		<damping value="10"/>
		<lateral_friction value="1"/>
  </contact>

    <visual>
    	<origin rpy="0.0 0 0" xyz="0 0 .1"/>
      <geometry>
        <box size=".01 0.01 .2"/>
      </geometry>
      <material name="grey">
        <color rgba="0.65 0.65 0.75 1"/>
      </material>
    </visual>
    <collision>
    	<origin rpy="0.0 0 0" xyz="0 0 .1"/>
      <geometry>
        <box size=".01 0.01 .2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.05"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>
  <joint name="knee_back_leftL_link" type="revolute">
    <axis xyz="0 1 0"/>
    <parent link="upper_leg_back_leftL_link"/>
    <child link="lower_leg_back_leftL_link"/>
    <origin rpy="0 0 0" xyz="0.0 0.01 .055"/>
    <limit effort="100" velocity="100"/>
    <joint_properties damping="0.0" friction="0.0"/>
  </joint>
 
</robot>
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


































