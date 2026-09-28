#!/usr/bin/env bash

set -euo pipefail
export LC_ALL=C

repo_root=$(git rev-parse --show-toplevel 2>/dev/null) || {
    printf 'form-audit: 当前目录不属于 Git 仓库。\n' >&2
    exit 2
}

global_audit=${FORM_AUDIT_BIN:-/root/configure/skills/form/scripts/form-audit.sh}
if [[ ! -x "$global_audit" ]]; then
    printf 'form-audit: 通用审计器不可执行: %s\n' "$global_audit" >&2
    exit 2
fi

if ! findings=$("$global_audit" --root "$repo_root" --quiet); then
    printf 'form-audit: 通用审计器执行失败。\n' >&2
    exit 2
fi

residual=0
allowed=0
while IFS=$'\t' read -r severity rule path message; do
    [[ -n "${severity:-}" ]] || continue

    case "${severity}:${rule}:${path}" in
        ERROR:TRACKED_DATA_CONTENT:data/quda-*|\
        ERROR:TRACKED_DATA_CONTENT:data/mg_matrix_comprehensive_final_20260928/*|\
        ERROR:TRACKED_DATA_CONTENT:data/report_multigrid_comprehensive_20260928/*|\
        ERROR:TRACKED_DATA_CONTENT:data/report_multigrid_comprehensive_20260929/*|\
        ERROR:TRACKED_DATA_CONTENT:data/report_multigrid_optimized_20260927/*|\
        ERROR:TRACKED_DATA_CONTENT:data/mg_matrix_20260916_round2/*|\
        ERROR:TRACKED_DATA_CONTENT:data/strict_trace_stage_timing_20260906.csv|\
        ERROR:TRACKED_DATA_CONTENT:data/strict_trace_profile_20260906.csv|\
        ERROR:TRACKED_DATA_CONTENT:data/strict_trace_detailed_20260906.svg|\
        ERROR:TRACKED_DATA_CONTENT:data/strict_trace_detailed_20260906.csv|\
        ERROR:TRACKED_DATA_CONTENT:data/strict_trace_20260902_final.svg|\
        ERROR:TRACKED_DATA_CONTENT:data/multigpu_formal_20260902.svg|\
        ERROR:TRACKED_DATA_CONTENT:data/mg_small_trace_20260916_r2.svg)
            allowed=$((allowed + 1))
            ;;
        WARNING:DOC_EXTENSION:docs/.gitignore)
            allowed=$((allowed + 1))
            ;;
        WARNING:DOC_EXTENSION:docs/张鑫*)
            allowed=$((allowed + 1))
            ;;
        WARNING:DOC_EXTENSION:docs/High-Performance*|\
        WARNING:FILE_WHITESPACE:docs/High-Performance*)
            allowed=$((allowed + 1))
            ;;
        WARNING:TEST_LOCATION:skills/tag/tag-chain.test.sh)
            allowed=$((allowed + 1))
            ;;
        *)
            if [[ "${severity}:${rule}:${path}" == WARNING:LOG_EXTENSION:logs/* ]]; then
                case "${path##*.}" in
                    png|PNG|jpg|jpeg|gif|webp|svg|pdf|tex|md|stdout|jsonl|gz)
                        allowed=$((allowed + 1))
                        ;;
                    *)
                        printf '%s\t%s\t%s\t%s\n' "$severity" "$rule" "$path" "$message"
                        residual=$((residual + 1))
                        ;;
                esac
            else
                printf '%s\t%s\t%s\t%s\n' "$severity" "$rule" "$path" "$message"
                residual=$((residual + 1))
            fi
            ;;
    esac
done <<<"$findings"

if ((residual > 0)); then
    printf 'FAIL: form 审计仍有 %d 个未登记 finding。\n' "$residual" >&2
    exit 1
fi

printf 'PASS: form 审计通过；已按本地规则豁免 %d 个既有资产 finding。\n' "$allowed"
