---
title: "代码运行指南"
permalink: /server/
last_modified_at: 2018-08-07T10:34:04+08:00
toc: true
toc_sticky: true
---

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

If you train the minitaur model, you may get something like this

> ```
> hadoop@gpu3:~/zyc/bullet3-master/examples/pybullet/gym$ python3 -m pybullet_envs.agents.train_ppo --config=pybullet_minitaur --logdir=minitaur
> pybullet build time: Jul 12 2018 10:40:53
> current_dir=/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_envs/bullet
> INFO:tensorflow:Start a new run and write summaries and checkpoints to minitaur/20180804T164348-pybullet_minitaur.
> urdf_root=/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_data
> options=
> argv[0]=
> urdf_root=/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_data
> options=
> argv[0]=
> urdf_root=/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_data
> options=
> argv[0]=
> urdf_root=/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_data
> options=
> argv[0]=
> urdf_root=/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_data
> options=
> argv[0]=
> urdf_root=/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_data
> options=
> argv[0]=
> urdf_root=/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_data
> options=
> argv[0]=
> urdf_root=/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_data
> options=
> argv[0]=
> urdf_root=/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_data
> options=
> argv[0]=
> urdf_root=/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_data
> options=
> argv[0]=
> urdf_root=/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_data
> options=
> argv[0]=
> urdf_root=/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_data
> options=
> argv[0]=
> urdf_root=/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_data
> options=
> argv[0]=
> urdf_root=/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_data
> options=
> argv[0]=
> urdf_root=/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_data
> options=
> argv[0]=
> urdf_root=/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_data
> options=
> argv[0]=
> urdf_root=/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_data
> options=
> argv[0]=
> urdf_root=/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_data
> options=
> argv[0]=
> urdf_root=/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_data
> options=
> argv[0]=
> urdf_root=/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_data
> options=
> argv[0]=
> urdf_root=/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_data
> options=
> argv[0]=
> urdf_root=/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_data
> options=
> argv[0]=
> urdf_root=/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_data
> options=
> argv[0]=
> urdf_root=/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_data
> options=
> argv[0]=
> urdf_root=/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_data
> options=
> argv[0]=
> urdf_root=/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_data
> options=
> argv[0]=
> urdf_root=/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_data
> options=
> argv[0]=
> urdf_root=/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_data
> options=
> argv[0]=
> urdf_root=/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_data
> options=
> argv[0]=
> urdf_root=/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_data
> options=
> argv[0]=
> 2018-08-04 16:43:49.152572: I tensorflow/core/platform/cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
> 2018-08-04 16:43:49.217256: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1212] Found device 0 with properties:
> name: GeForce GTX TITAN X major: 5 minor: 2 memoryClockRate(GHz): 1.076
> pciBusID: 0000:41:00.0
> totalMemory: 11.92GiB freeMemory: 445.69MiB
> 2018-08-04 16:43:49.217293: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1312] Adding visible gpu devices: 0
> 2018-08-04 16:43:49.394882: I tensorflow/core/common_runtime/gpu/gpu_device.cc:993] Creating TensorFlow device (/device:GPU:0 with 165 MB memory) -> physical GPU (device: 0, name: GeForce GTX TITAN X, pci bus id: 0000:41:00.0, compute capability: 5.2)
> WARNING:tensorflow:Variable /= will be deprecated. Use variable.assign_div if you want assignment to the variable value or 'x = x / y' if you want a new python Tensor object.
> INFO:tensorflow:Graph contains 52717 trainable variables.
> 2018-08-04 16:43:53.840545: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1312] Adding visible gpu devices: 0
> 2018-08-04 16:43:53.840895: I tensorflow/core/common_runtime/gpu/gpu_device.cc:993] Creating TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 224 MB memory) -> physical GPU (device: 0, name: GeForce GTX TITAN X, pci bus id: 0000:41:00.0, compute capability: 5.2)
> INFO:tensorflow:
> --------------------------------------------------
> Phase train (phase step 0, global step 0).
> return and value: [-0.000166058089][1.03854764]
> normalized advantage: [1.96078622e-06]
> value loss: [8.33006197e-06]
> policy loss: [-0.0333030187]
> current penalty: [1]
> kl change: [4.13230118e-05]
> decrease penalty [0]
> hadoop@gpu3:~/zyc/bullet3-master/examples/pybullet/gym$ python3 -m pybullet_envs.agents.train_ppo --config=pybullet_minitaur --logdir=minitaur
> pybullet build time: Jul 12 2018 10:40:53
> current_dir=/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_envs/bullet
> INFO:tensorflow:Start a new run and write summaries and checkpoints to minitaur/20180804T164815-pybullet_minitaur.
> urdf_root=/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_data
> options=
> argv[0]=
> urdf_root=/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_data
> options=
> argv[0]=
> urdf_root=/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_data
> options=
> argv[0]=
> urdf_root=/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_data
> options=
> argv[0]=
> urdf_root=/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_data
> options=
> argv[0]=
> urdf_root=/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_data
> options=
> argv[0]=
> urdf_root=/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_data
> options=
> argv[0]=
> urdf_root=/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_data
> options=
> argv[0]=
> urdf_root=/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_data
> options=
> argv[0]=
> urdf_root=/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_data
> options=
> argv[0]=
> urdf_root=/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_data
> options=
> argv[0]=
> urdf_root=/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_data
> options=
> argv[0]=
> urdf_root=/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_data
> options=
> argv[0]=
> urdf_root=/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_data
> options=
> argv[0]=
> urdf_root=/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_data
> options=
> argv[0]=
> urdf_root=/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_data
> options=
> argv[0]=
> urdf_root=/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_data
> options=
> argv[0]=
> urdf_root=/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_data
> options=
> argv[0]=
> urdf_root=/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_data
> options=
> argv[0]=
> urdf_root=/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_data
> options=
> argv[0]=
> urdf_root=/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_data
> options=
> argv[0]=
> urdf_root=/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_data
> options=
> argv[0]=
> urdf_root=/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_data
> options=
> argv[0]=
> urdf_root=/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_data
> options=
> argv[0]=
> urdf_root=/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_data
> options=
> argv[0]=
> urdf_root=/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_data
> options=
> argv[0]=
> urdf_root=/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_data
> options=
> argv[0]=
> urdf_root=/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_data
> urdf_root=/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_data
> options=
> options=
> argv[0]=
> argv[0]=
> urdf_root=/home/hadoop/zyc/bullet3-master/examples/pybullet/gym/pybullet_data
> options=
> argv[0]=
> 2018-08-04 16:48:16.220087: I tensorflow/core/platform/cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
> 2018-08-04 16:48:16.278230: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1212] Found device 0 with properties:
> name: GeForce GTX TITAN X major: 5 minor: 2 memoryClockRate(GHz): 1.076
> pciBusID: 0000:41:00.0
> totalMemory: 11.92GiB freeMemory: 474.06MiB
> 2018-08-04 16:48:16.278264: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1312] Adding visible gpu devices: 0
> 2018-08-04 16:48:16.452630: I tensorflow/core/common_runtime/gpu/gpu_device.cc:993] Creating TensorFlow device (/device:GPU:0 with 193 MB memory) -> physical GPU (device: 0, name: GeForce GTX TITAN X, pci bus id: 0000:41:00.0, compute capability: 5.2)
> WARNING:tensorflow:Variable /= will be deprecated. Use variable.assign_div if you want assignment to the variable value or 'x = x / y' if you want a new python Tensor object.
> INFO:tensorflow:Graph contains 52717 trainable variables.
> 2018-08-04 16:48:20.856283: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1312] Adding visible gpu devices: 0
> 2018-08-04 16:48:20.856604: I tensorflow/core/common_runtime/gpu/gpu_device.cc:993] Creating TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 224 MB memory) -> physical GPU (device: 0, name: GeForce GTX TITAN X, pci bus id: 0000:41:00.0, compute capability: 5.2)
> INFO:tensorflow:
> --------------------------------------------------
> Phase train (phase step 0, global step 0).
> return and value: [-0.000168203667][0.622981906]
> normalized advantage: [1.80435177e-06]
> value loss: [1.09974517e-05]
> policy loss: [-0.0353416912]
> current penalty: [1]
> kl change: [6.02173313e-05]
> decrease penalty [0]
> INFO:tensorflow:Score -1.4587786197662354.
> INFO:tensorflow:
> --------------------------------------------------
> Phase eval (phase step 0, global step 30000).
> INFO:tensorflow:Score -0.02941494807600975.
> INFO:tensorflow:
> --------------------------------------------------
> Phase train (phase step 30000, global step 60000).
> return and value: [-0.000168858955][-0.0566422977]
> normalized advantage: [1.22070309e-08]
> value loss: [8.63245361e-07]
> policy loss: [0.000396524789]
> current penalty: [0.666666687]
> kl change: [3.0424977e-05]
> decrease penalty [0]
> INFO:tensorflow:Score -1.3721096515655518.
> INFO:tensorflow:
> --------------------------------------------------
> Phase eval (phase step 30000, global step 90000).
> INFO:tensorflow:Score -0.03422568365931511.
> INFO:tensorflow:
> --------------------------------------------------
> Phase train (phase step 60000, global step 120000).
> return and value: [-0.000155343267][-0.0943794325]
> normalized advantage: [-2.03450523e-09]
> value loss: [9.55702603e-07]
> policy loss: [0.000348282192]
> current penalty: [0.444444478]
> kl change: [2.93298726e-05]
> decrease penalty [0]
> INFO:tensorflow:Score -1.4921305179595947.
> INFO:tensorflow:
> --------------------------------------------------
> Phase eval (phase step 60000, global step 150000).
> INFO:tensorflow:Score -0.02659527398645878.
> INFO:tensorflow:
> --------------------------------------------------
> Phase train (phase step 90000, global step 180000).
> return and value: [-0.000155730027][-0.109232821]
> normalized advantage: [3.05175774e-09]
> value loss: [6.48198636e-07]
> policy loss: [0.00039977656]
> current penalty: [0.296296328]
> kl change: [0.000105865205]
> decrease penalty [0]
> INFO:tensorflow:Score -1.4537789821624756.
> INFO:tensorflow:
> --------------------------------------------------
> Phase eval (phase step 90000, global step 210000).
> INFO:tensorflow:Score -0.025868888944387436.
> INFO:tensorflow:
> --------------------------------------------------
> Phase train (phase step 120000, global step 240000).
> return and value: [-0.000163117438][-0.0959276482]
> normalized advantage: [3.91642239e-08]
> value loss: [7.83652638e-07]
> policy loss: [0.000449127285]
> current penalty: [0.197530895]
> kl change: [3.13810342e-05]
> decrease penalty [0]
> INFO:tensorflow:Score -1.5147202014923096.
> INFO:tensorflow:
> --------------------------------------------------
> Phase eval (phase step 120000, global step 270000).
> INFO:tensorflow:Score -0.056419048458337784.
> INFO:tensorflow:
> --------------------------------------------------
> Phase train (phase step 150000, global step 300000).
> return and value: [-0.000167911407][-0.104914621]
> normalized advantage: [-3.56038421e-09]
> value loss: [9.24861354e-07]
> policy loss: [0.000407454383]
> current penalty: [0.131687269]
> kl change: [6.71114612e-05]
> decrease penalty [0]
> INFO:tensorflow:Score -1.5923177003860474.
> INFO:tensorflow:
> --------------------------------------------------
> Phase eval (phase step 150000, global step 330000).
> INFO:tensorflow:Score -0.03569361940026283.
> INFO:tensorflow:
> --------------------------------------------------
> Phase train (phase step 180000, global step 360000).
> return and value: [-0.000156877752][-0.108126894]
> normalized advantage: [-1.29699709e-08]
> value loss: [1.16474405e-06]
> policy loss: [0.000426398939]
> current penalty: [0.0877915174]
> kl change: [5.99894047e-05]
> decrease penalty [0]
> INFO:tensorflow:Score -1.4493634700775146.
> INFO:tensorflow:
> --------------------------------------------------
> Phase eval (phase step 180000, global step 390000).
> INFO:tensorflow:Score -0.04736465588212013.
> INFO:tensorflow:
> --------------------------------------------------
> Phase train (phase step 210000, global step 420000).
> return and value: [-0.000163537217][-0.121231847]
> normalized advantage: [-2.13623039e-08]
> value loss: [5.98159602e-07]
> policy loss: [0.000456030451]
> current penalty: [0.0585276783]
> kl change: [8.40434877e-05]
> decrease penalty [0]
> INFO:tensorflow:Score -1.4359354972839355.
> INFO:tensorflow:
> --------------------------------------------------
> ```

