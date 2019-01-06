#!/bin/bash
set -eux
#for e in Hopper-v2 Ant-v2 HalfCheetah-v2 Humanoid-v2 Reacher-v2 Walker2d-v2
#do
#    python run_expert.py experts/$e.pkl $e --num_rollouts=1
#done

for e in Hopper-v2
do
    #LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-396 LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-396/libGL.so python run_expert.py experts/$e.pkl $e --num_rollouts=1 --render
    python run_expert.py experts/$e.pkl $e --num_rollouts=1
done
