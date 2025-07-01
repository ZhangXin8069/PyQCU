GPU_MODEL=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | head -n 1)
GPU_MODEL=$(echo $GPU_MODEL | xargs)
case "$GPU_MODEL" in
*"Tesla H100"*) export sm_arch="90" ;;
*"Tesla H20"*) export sm_arch="90" ;;
*"Tesla A100"*) export sm_arch="80" ;;
*"Tesla A800"*) export sm_arch="80" ;;
*"Tesla V100"*) export sm_arch="70" ;;
*"Tesla P100"*) export sm_arch="60" ;;
*"Tesla K80"*) export sm_arch="37" ;;
*"Tesla T4"*) export sm_arch="70" ;;
*"GeForce RTX 4060"*) export sm_arch="80" ;;
*"GeForce RTX 3080"*) export sm_arch="80" ;;
*"GeForce RTX 3090"*) export sm_arch="80" ;;
*"GeForce GTX 1080"*) export sm_arch="61" ;;
*"GeForce GTX 1060"*) export sm_arch="61" ;;
*"GeForce GTX 1070"*) export sm_arch="61" ;;
*"Quadro GV100"*) export sm_arch="70" ;;
*"Quadro RTX 8000"*) export sm_arch="75" ;;
*"Quadro P1000"*) export sm_arch="60" ;;
*)
    export sm_arch="80" # in snsc
    ;;
esac
echo "First GPU Model: $GPU_MODEL -> $sm_arch"
export mpi_home=$(dirname $(which mpirun))/..
if [ "$mpi_home" = "/usr/bin/.." ]; then
    export mpi_home="/usr/lib/x86_64-linux-gnu/openmpi"
fi
echo "MPI_HOME: $mpi_home"
