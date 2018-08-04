[TOC]

Figuring out how pybullet minitaur works

# Source

[paper](https://arxiv.org/abs/1804.10332)

[code](https://git.io/vp0V3)

# Code analysis

## Run the code

### Origin method

First, `cd` to `~/bullet3-master/examples/pybullet/gym/pybullet_envs/` this folder and then run the following command.
```bash
python3 -m pybullet_envs.agents.train_ppo --config=pybullet_pendulum --logdir=pendulum
```

### Rewrite with jupyter notebook

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

Every example is build on the default configuration.

# Algorithm

## PPO algorithm

PPO algorithm is the abbreviation of Proximal Policy Optimization algorithms, which is proposed in 2017. This paper can download from [arxiv](https://arxiv.org/abs/1707.06347).

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










































