#!/bin/bash
#DSUB --mpi hmpi
#DSUB -q q_hpcapp
#DSUB -n qcu
#DSUB -x job
#DSUB -nn 1
#DSUB -rpn 16
#DSUB -N 16
#DSUB -R "cpu=37;mem=8G"
#DSUB -oo output.txt
#DSUB -eo error.txt
#DSUB -T 600
source "${HOME}/env.sh"
lx2-stream-source >log.txt 2>&1
