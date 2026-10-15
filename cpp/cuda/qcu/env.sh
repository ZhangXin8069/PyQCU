GPU_MODEL=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader 2>/dev/null | head -n 1)
GPU_MODEL=$(printf '%s\n' "$GPU_MODEL" | xargs)
if [ -z "${sm_arch:-}" ]; then
    sm_arch=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '[:space:].')
fi
if ! printf '%s\n' "${sm_arch:-}" | grep -Eq '^[0-9]+$'; then
    # Fallback model table for drivers whose nvidia-smi has no compute_cap
    # query.  Keep exact architecture families before broad product families.
    case "$GPU_MODEL" in
        # Blackwell
        *"Jetson"*"Thor"*|*"Jetson"*"T4000"*|*"Jetson"*"T5000"*) export sm_arch="110" ;;
        *"GB10"*) export sm_arch="121" ;;
        *"B300"*|*"GB300"*) export sm_arch="103" ;;
        *"B100"*|*"B200"*|*"GB200"*) export sm_arch="100" ;;
        *"RTX PRO"*"Blackwell"*|*"Blackwell"*|*"GeForce RTX 50"*) export sm_arch="120" ;;

        # Hopper
        *"H100"*|*"H200"*|*"H800"*|*"H20"*|*"GH200"*) export sm_arch="90" ;;

        # Ada Lovelace
        *"Ada"*|*"GeForce RTX 40"*|*"L2"*|*"L4"*|*"L20"*|*"L40"*|*"L40S"*) export sm_arch="89" ;;

        # Ampere data-center parts
        *"A100"*|*"A800"*|*"A30"*|*"CMP 170HX"*) export sm_arch="80" ;;

        # Ampere consumer and workstation parts
        *"GeForce RTX 30"*|*"RTX A"*|*"A10"*|*"A16"*|*"A40"*|*"A2"*|*"CMP 40HX"*|*"CMP 50HX"*|*"CMP 90HX"*|*"MX570"*) export sm_arch="86" ;;

        # Turing
        *"GeForce RTX 20"*|*"GeForce GTX 16"*|*"Quadro RTX"*|*"TITAN RTX"*|*"Tesla T4"*|*"T400"*|*"T600"*|*"T1000"*|*"T1200"*|*"T2000"*|*"MX450"*|*"MX550"*|*"CMP 30HX"*) export sm_arch="75" ;;
        *"RTX 3000"*|*"RTX 4000"*|*"RTX 5000"*|*"RTX 6000"*|*"RTX 8000"*) export sm_arch="75" ;;

        # Volta
        *"V100"*|*"TITAN V"*|*"GV100"*) export sm_arch="70" ;;

        # Pascal (GP10x consumer, P4/P6/P40 and Quadro P series)
        *"GeForce GTX 10"*|*"TITAN Xp"*|*"TITAN X Pascal"*|*"Tesla P4"*|*"Tesla P6"*|*"Tesla P40"*|*"Quadro P"*|*"MX150"*|*"MX230"*|*"MX250"*|*"MX330"*|*"MX350"*) export sm_arch="61" ;;

        # Pascal GP100
        *"P100"*|*"GP100"*) export sm_arch="60" ;;

        # Maxwell (GM20x)
        *"GeForce GTX 9"*|*"TITAN X"*|*"Quadro M"*|*"Tesla M4"*|*"Tesla M6"*|*"Tesla M40"*|*"Tesla M60"*|*"GRID M"*) export sm_arch="52" ;;

        # Maxwell (GM10x)
        *"Tesla M10"*|*"GeForce GTX 745"*|*"GeForce GTX 750"*|*"GeForce GTX 850M"*|*"GeForce GTX 860M"*|*"GeForce MX110"*|*"GeForce MX130"*|*"GeForce 920MX"*|*"GeForce 930MX"*|*"GeForce 940MX"*|*"Quadro M500M"*|*"Quadro M520"*|*"Quadro M600M"*|*"Quadro M600"*|*"Quadro M620M"*|*"Quadro M620"*|*"Quadro M1000M"*|*"Quadro M1200M"*|*"Quadro M1200"*|*"Quadro K2200"*|*"Quadro K1200"*|*"Quadro K620"*) export sm_arch="50" ;;

        # Kepler
        *"Tesla K80"*) export sm_arch="37" ;;
        *"Tesla K40"*|*"Tesla K20"*|*"Quadro K6000"*|*"GTX 780"*|*"GTX TITAN"*|*"TITAN Z"*|*"TITAN Black"*|*"TITAN"*) export sm_arch="35" ;;
        *"Tesla K10"*|*"Quadro K"*|*"GRID K"*|*"GeForce GTX 6"*|*"GeForce GTX 7"*|*"GeForce GTX 870M"*|*"GeForce GTX 880M"*) export sm_arch="30" ;;

        # Fermi (legacy)
        *"GeForce GTX 460"*|*"GeForce GTX 550 Ti"*|*"GeForce GTX 560"*|*"GeForce GTS 450"*|*"GeForce GT 430"*|*"GeForce GT 440"*|*"GeForce GT 520"*|*"GeForce GT 530"*|*"GeForce GT 540M"*|*"GeForce GT 550M"*|*"GeForce GT 555M"*|*"GeForce GT 630M"*|*"GeForce GT 635M"*|*"Quadro 600"*|*"Quadro 700"*|*"Quadro 2000"*) export sm_arch="21" ;;
        *"Tesla M20"*|*"Tesla C20"*|*"Tesla S2050"*|*"GeForce GTX 470"*|*"GeForce GTX 480"*|*"GeForce GTX 570"*|*"GeForce GTX 580"*|*"GeForce GTX 590"*|*"Quadro 4000"*|*"Quadro 5000"*|*"Quadro 6000"*|*"Quadro 7000"*) export sm_arch="20" ;;
        *"GeForce GTX 4"*|*"GeForce GTX 5"*|*"GeForce 8"*|*"GeForce 9"*|*"GeForce 3"*") export sm_arch="20" ;;

        # Pre-Fermi (kept for recognition; old toolkits only)
        *"Tesla C1060"*|*"Tesla S1070"*|*"GTX 200"*|*"GTX 300"*|*"GT 2"*|*"Quadro FX 4800"*|*"Quadro FX 5800"*) export sm_arch="13" ;;
        *"GeForce 8"*|*"GeForce 9"*) export sm_arch="11" ;;
        *"Tesla C870"*|*"Tesla S870"*) export sm_arch="10" ;;

        # Jetson / Tegra
        *"Tegra X1"*|*"Jetson Nano"*|*"Jetson TX1"*) export sm_arch="53" ;;
        *"Tegra X2"*|*"Jetson TX2"*) export sm_arch="62" ;;
        *"Xavier"*) export sm_arch="72" ;;
        *"Jetson"*"Orin"*) export sm_arch="87" ;;
        *)
            export sm_arch="80" # fallback for the snsc host
        ;;
    esac
fi
export sm_arch
echo "First GPU Model: $GPU_MODEL -> $sm_arch"
export mpi_home=$(dirname $(which mpirun))/..
if [ "$mpi_home" = "/usr/bin/.." ]; then
    export mpi_home="/usr/lib/x86_64-linux-gnu/openmpi"
fi
echo "MPI_HOME: $mpi_home"
# export
# export LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH
