#!/usr/bin/env bash
set -eux

#pipenv run bash ./demo.bash

e='Hopper-v2'
#pipenv run python run_expert.py experts/$e.pkl $e --num_rollouts=200
#pipenv run python behavioral_cloning.py $e
pipenv run python behavioral_cloning_eager.py $e
