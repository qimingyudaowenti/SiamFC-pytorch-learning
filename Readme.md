# SiamFC-pytorch-learning

This repository is the combination of two implementations of [SiamFC](https://www.robots.ox.ac.uk/~luca/siamese-fc.html) tracker:

- [bilylee/SiamFC-TensorFlow](https://github.com/bilylee/SiamFC-TensorFlow)
- [huanglianghua/siamfc-pytorch](https://github.com/huanglianghua/siamfc-pytorch)

This repository is helpful to learn SiamFC. For doing more research on SiamFC, fork repositories above.

## Why combine them?

Aiming at learning SiamFC, I search for a light-weighted python implementation of it until I find [huanglianghua/siamfc-pytorch](https://github.com/bilylee/SiamFC-TensorFlow). In huanglianghua's code, [GOT-10k toolkit](https://github.com/got-10k/toolkit) (the excellent tracking toolkit) is used. For speeding up training, the data is processed by bilylee's code.

## requirements

- python 3
- pytorch 1.0
- opencv-python 3

## Train

1. prepare data:

   See *data_prepare/Readme.md*.

2. run training script:

   ```
   mkdir -p saved/test_model_weights saved/training_resume_state

   python train.py
   ```

3. save weights for resuming training and testing:

   - the checkpoint for resuming training is saved in *saved/training_resume_state* (periodically / when Ctrl-c).
   - the weights for testing is saved in *saved/test_model_weights*.
   - script will auto select the newest weights.

## Test

run testing script:

```python
cd SiamFC-pytorch-learning

python test.py
```

## Learn more about SiamFC
[SiamFC 分析](http://geyao1995.com/SiamFC/)