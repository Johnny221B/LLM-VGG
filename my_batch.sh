#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1            
#SBATCH -c 4
#SBATCH -p volta-gpu
#SBATCH --mem=5g          
#SBATCH -t 2:00:00      
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1

export PATH=/work/users/l/i/linyuliu/anaconda3/bin:$PATH

module purge
module load python/2.7.13
mudule load cuda/12.3
module load cudnn/8.9.7.29
module add tensorflow/1.14.0

source activate paleo
export PYTHONPATH=/work/users/l/i/linyuliu/anaconda3/envs/paleo/lib/python2.7/site-packages:$PYTHONPATH
export TF_USE_DEEP_CONV2D=0
export TF_CUDNN_USE_AUTOTUNE=0
export TF_CUDNN_WORKSPACE_LIMIT_IN_MB=4096
which python

python paleo/profiler.py "$@"