#!/bin/bash
python3 collect_dataset.py --env-name HalfCheetah-v4 \
  --expert-path logs/sac/HalfCheetah-v4_1/HalfCheetah-v4.zip \
  --n-timesteps 500_000
