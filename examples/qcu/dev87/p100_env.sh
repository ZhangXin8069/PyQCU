#!/usr/bin/env bash
#
# Source this file from bash or zsh before running PyQCU on the two P100s.
# The physical V100 is intentionally hidden. CUDA logical device 0/1 therefore
# always mean the two Tesla P100-PCIE-16GB cards.

if [ -n "${BASH_VERSION:-}" ]; then
    _p100_env_script="${BASH_SOURCE[0]}"
elif [ -n "${ZSH_VERSION:-}" ]; then
    _p100_env_script="${(%):-%x}"
else
    _p100_env_script="$0"
fi

_p100_env_dir=$(CDPATH= cd -- "$(dirname -- "$_p100_env_script")" && pwd)
_p100_repo_root=$(CDPATH= cd -- "$_p100_env_dir/../../.." && pwd)

# shellcheck source=/dev/null
. "$_p100_repo_root/env.sh"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1

_p100_site="$_p100_repo_root/data/p100-site"
case ":${PYTHONPATH:-}:" in
    *":$_p100_site:"*) ;;
    *) export PYTHONPATH="$_p100_site${PYTHONPATH:+:$PYTHONPATH}" ;;
esac

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export QCU_P100_SITE="$_p100_site"
export QCU_STRICT_DEVICE=0
export QCU_STRICT_DEVICE_COUNT=2
export QCU_LOG_DIR="${QCU_LOG_DIR:-$_p100_repo_root/data/p100-logs}"
mkdir -p "$QCU_LOG_DIR"

unset _p100_env_script _p100_env_dir _p100_repo_root _p100_site
