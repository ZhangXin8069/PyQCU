#!/usr/bin/env bash

set -euo pipefail
export LC_ALL=C

repo_root=$(git rev-parse --show-toplevel 2>/dev/null) || {
    printf 'form-audit: 当前目录不属于 Git 仓库。\n' >&2
    exit 2
}

if [[ -n "${FORM_AUDIT_BIN:-}" ]]; then
    global_audit=$FORM_AUDIT_BIN
else
    configure_root=${CONFIGURE_ROOT:-${HOME:-}/configure}
    global_audit=$configure_root/skills/form/scripts/form-audit.sh
    if [[ ! -x "$global_audit" && -x /root/configure/skills/form/scripts/form-audit.sh ]]; then
        global_audit=/root/configure/skills/form/scripts/form-audit.sh
    fi
fi

if [[ ! -x "$global_audit" ]]; then
    printf 'form-audit: 通用审计器不可执行: %s\n' "$global_audit" >&2
    printf 'form-audit: 请设置 FORM_AUDIT_BIN，或设置 CONFIGURE_ROOT 指向 configure 仓库根目录。\n' >&2
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
        ERROR:TRACKED_DATA_CONTENT:data/report_multigrid_comprehensive_20260930/*|\
        ERROR:TRACKED_DATA_CONTENT:data/report_multigrid_comprehensive_20261001/*|\
        ERROR:TRACKED_DATA_CONTENT:data/report_multigrid_comprehensive_20261002/*|\
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
        WARNING:TEST_LOCATION:skills/pyqcu-tag/tag-chain.test.sh)
            allowed=$((allowed + 1))
            ;;
        WARNING:TEST_LOCATION:cpp/kupl/qcu/tests/benchmark_kupl.cpp|\
        WARNING:TEST_LOCATION:cpp/kupl/qcu/tests/test_canonical_layout.py|\
        WARNING:TEST_LOCATION:cpp/kupl/qcu/tests/check_abi.py|\
        WARNING:TEST_LOCATION:cpp/kupl/qcu/tests/test_failure_modes.py|\
        WARNING:TEST_LOCATION:cpp/kupl/qcu/tests/test_kupl.cpp|\
        WARNING:TEST_LOCATION:cpp/kupl/qcu/tests/test_multigrid.py|\
        WARNING:TEST_LOCATION:cpp/kupl/qcu/tests/test_new_abi.py)
            # KUPL tests remain inside the standalone CMake project so the
            # backend can be built and audited without importing PyQCU.
            allowed=$((allowed + 1))
            ;;
        WARNING:LOG_EXTENSION:logs/slurm-*.out_perf)
            # The cluster scheduler emits a tracked *_perf summary next to
            # each Slurm stdout file.  Preserve it as first-party resource
            # evidence rather than renaming the job's canonical output.
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

skills_dir=$repo_root/skills
skills_index=$skills_dir/AGENTS.md
skill_count=0

if [[ -d "$skills_dir" ]]; then
    skill_count=$(find "$skills_dir" -mindepth 1 -maxdepth 1 -type d -name 'pyqcu-*' | awk 'END {print NR + 0}')

    while IFS= read -r skill_dir; do
        skill_name=${skill_dir##*/}
        skill_file=$skill_dir/SKILL.md
        skill_agents=$skill_dir/AGENTS.md

        if [[ ! -f "$skill_file" ]]; then
            printf 'ERROR\tSKILL_STRUCTURE\tskills/%s\t缺少 SKILL.md\n' "$skill_name"
            residual=$((residual + 1))
        else
            declared_name=$(sed -n 's/^name:[[:space:]]*//p' "$skill_file" | head -n 1)
            if [[ "$declared_name" != "$skill_name" ]]; then
                printf 'ERROR\tSKILL_FRONTMATTER\tskills/%s/SKILL.md\tname=%s 与目录名不一致\n' \
                    "$skill_name" "${declared_name:-<missing>}"
                residual=$((residual + 1))
            fi
        fi

        if [[ ! -f "$skill_agents" ]]; then
            printf 'ERROR\tSKILL_STRUCTURE\tskills/%s\t缺少 AGENTS.md\n' "$skill_name"
            residual=$((residual + 1))
        fi
    done < <(find "$skills_dir" -mindepth 1 -maxdepth 1 -type d -name 'pyqcu-*' | sort)
else
    printf 'ERROR\tSKILL_STRUCTURE\tskills\t缺少技能目录\n'
    residual=$((residual + 1))
fi

if [[ -f "$skills_index" ]]; then
    index_count=$(sed -n 's/^| `\(pyqcu-[^`]*\)`.*/\1/p' "$skills_index" | awk 'END {print NR + 0}')
    if [[ "$index_count" -ne "$skill_count" ]]; then
        printf 'ERROR\tSKILL_INDEX_COUNT\tskills/AGENTS.md\t索引 %d 行，实际 %d 个目录\n' \
            "$index_count" "$skill_count"
        residual=$((residual + 1))
    fi

    while IFS= read -r skill_name; do
        [[ -n "$skill_name" ]] || continue
        printf 'ERROR\tSKILL_INDEX_MISSING\tskills/AGENTS.md\t缺少目录 %s 的索引行\n' "$skill_name"
        residual=$((residual + 1))
    done < <(
        comm -23 \
            <(find "$skills_dir" -mindepth 1 -maxdepth 1 -type d -name 'pyqcu-*' -exec basename {} \; | sort) \
            <(sed -n 's/^| `\(pyqcu-[^`]*\)`.*/\1/p' "$skills_index" | sort)
    )

    while IFS= read -r skill_name; do
        [[ -n "$skill_name" ]] || continue
        printf 'ERROR\tSKILL_INDEX_EXTRA\tskills/AGENTS.md\t索引项 %s 无对应目录\n' "$skill_name"
        residual=$((residual + 1))
    done < <(
        comm -13 \
            <(find "$skills_dir" -mindepth 1 -maxdepth 1 -type d -name 'pyqcu-*' -exec basename {} \; | sort) \
            <(sed -n 's/^| `\(pyqcu-[^`]*\)`.*/\1/p' "$skills_index" | sort)
    )

    if ! grep -Fq "当前 ${skill_count}/${skill_count}" "$skills_index"; then
        printf 'ERROR\tSKILL_INDEX_HEADER\tskills/AGENTS.md\t缺少当前 %d/%d 计数\n' \
            "$skill_count" "$skill_count"
        residual=$((residual + 1))
    fi

    if ! grep -Fq "## 技能表（${skill_count}/${skill_count}）" "$skills_index"; then
        printf 'ERROR\tSKILL_INDEX_HEADER\tskills/AGENTS.md\t缺少技能表 %d/%d 标头\n' \
            "$skill_count" "$skill_count"
        residual=$((residual + 1))
    fi
else
    printf 'ERROR\tSKILL_INDEX_MISSING\tskills/AGENTS.md\t缺少技能索引\n'
    residual=$((residual + 1))
fi

if ((residual > 0)); then
    printf 'FAIL: form 审计仍有 %d 个未登记 finding。\n' "$residual" >&2
    exit 1
fi

printf 'PASS: form 审计通过；%d 个技能目录与索引一致，豁免 %d 个既有资产 finding。\n' \
    "$skill_count" "$allowed"
