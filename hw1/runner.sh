#!/usr/bin/env bash
set -eux

#pipenv run bash ./demo.bash

for e in Ant-v2
#for e in Hopper-v2 Ant-v2 HalfCheetah-v2 Humanoid-v2 Reacher-v2 Walker2d-v2
do
#    pipenv run python run_expert.py experts/$e.pkl $e --num_rollouts=100
    #pipenv run python behavioral_cloning_eager.py $e
    pipenv run python behavioral_cloning_eager_eval.py $e #--render
    echo $e
done

#e='Hopper-v2'
#pipenv run python run_expert.py experts/$e.pkl $e --num_rollouts=5 --no_dump
#pipenv run python run_expert.py experts/$e.pkl $e --num_rollouts=1 --render
#pipenv run python behavioral_cloning.py $e

#pipenv run python behavioral_cloning_eager.py $e --print_graph
#pipenv run python behavioral_cloning_eager.py $e
#pipenv run python behavioral_cloning_eager_eval.py $e --render

#pipenv run python inspect_checkpoint.py

