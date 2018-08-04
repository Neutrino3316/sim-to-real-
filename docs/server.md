# How to run the code in the sever

Make sure that you're in the Lab's Intranet, and is able to `ping 192.168.1.115`

Then use `Git Bash` or some other software that supports `ssh` to connect the sever. The commands are as follows

```bash
ssh hadoop@192.168.1.115
```

## Using  Jupyter Notebook

Activate jupyter notebook

```bash
jupyter notebook --ip=192.168.1.115
```

It will return a link like this
> `http://192.168.1.115:8888/?token=050557c6bbe264a1c61d0dd5398a7ada372c0a91b73bfe5e`

Copy this url to your browser, and open it. Then the Jupyter Notebook will show up. Go for the `train_test_v0.1.0.ipynb` to run the code.

> `http://192.168.1.115:[port]/notebooks/zyc/bullet3-master/examples/pybullet/gym/pybullet_envs/agents/train_test_v0.1.0.ipynb`

For more imformation, please visit [Jupyter Notebook Documentation](https://jupyter.readthedocs.io/en/latest/)

## Using command line

```bash
ssh hadoop@192.168.1.115
cd ~/zyc/bullet3-master/examples/pybullet/gym
python3 -m pybullet_envs.agents.train_ppo --config=pybullet_pendulum --logdir=pendulum
python3 -m pybullet_envs.agents.train_ppo --config=pybullet_minitaur --logdir=minitaur
```
