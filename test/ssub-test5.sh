#!/bin/bash
#SBATCH --job-name=ssub-test5
#SBATCH --partition=h20-nettr
#SBATCH --nodes=1
#SBATCH -n 8
#SBATCH --time=00-00:30:00
#SBATCH --output=ssub-test5.out
#SBATCH --error=ssub-test5.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zhangxin8069@qq.com
#SBATCH --gres=gpu:8
source /public/home/zhangxin/env.sh

mpirun -n 8 python ./test-test5.py
