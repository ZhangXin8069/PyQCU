GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader)
for gpu in $GPU_INFO; do
    case "$gpu" in
    *"Tesla H100"*) sm="90" ;;
    *"Tesla H20"*) sm="90" ;;
    *"Tesla A100"*) sm="80" ;;
    *"Tesla A800"*) sm="80" ;;
    *"Tesla V100"*) sm="70" ;;
    *"Tesla P100"*) sm="60" ;;
    *"Tesla K80"*) sm="37" ;;
    *"Tesla T4"*) sm="70" ;;
    *"GeForce RTX 4060"*) sm="80" ;;
    *"GeForce RTX 3080"*) sm="80" ;;
    *"GeForce RTX 3090"*) sm="80" ;;
    *"GeForce GTX 1080"*) sm="61" ;;
    *"GeForce GTX 1060"*) sm="61" ;;
    *"GeForce GTX 1070"*) sm="61" ;;
    *"Quadro GV100"*) sm="70" ;;
    *"Quadro RTX 8000"*) sm="75" ;;
    *"Quadro P1000"*) sm="60" ;;
    *)
        sm="Unknown"
        ;;
    esac
    echo "$gpu -> sm_$sm"
done
export sm_arch=$sm
export mpi_home="/usr/lib/x86_64-linux-gnu/openmpi"
