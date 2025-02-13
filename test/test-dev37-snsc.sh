#!/bin/bash
#SBATCH --job-name=ssub
#SBATCH --partition=gpu-debug
#SBATCH --nodes=1
#SBATCH -n 2
#SBATCH --time=00-00:30:00
#SBATCH --output=ssub.out
#SBATCH --error=ssub.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zhangxin8069@qq.com
#SBATCH --gres=gpu:2
source /public/home/zhangxin/env.sh
pushd ../
bash ./install.sh
popd
mpirun --allow-run-as-root -n 2 python ./test-dev37.py
