# 如何連接到服務器

```bash
ssh hadoop@192.168.1.115
jupyter notebook --ip=192.168.1.115
cd ~/zyc/bullet3-master/examples/pybullet/gym
python3 -m pybullet_envs.agents.train_ppo --config=pybullet_pendulum --logdir=pendulum
```
