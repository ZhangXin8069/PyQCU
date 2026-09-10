#!/usr/bin/env bash

# Audit and, when explicitly requested, repair the follow chain of annotated
# stab/dev/bug/test tags.  The script changes tag objects only; it never moves
# the commit to which a tag peels.

set -u
set -o pipefail
export LC_ALL=C

usage() {
    cat <<'USAGE'
用法:
  tag-chain.sh [--check] [--repo PATH] [--remote NAME] [--root TAG]
  tag-chain.sh --repair [--dry-run] [--repo PATH] [--remote NAME]
  tag-chain.sh --repair --apply [--repo PATH]
  tag-chain.sh --repair --apply --remote NAME --confirm-remote-rewrite

选项:
  --check                    只读检查（默认）。
  --repair                   生成修正清单；不带 --apply 时不写入。
  --dry-run                  显式要求只生成清单。
  --apply                    应用可安全推导的本地 follow 前缀修正。
  --repo PATH                指定 Git 工作树或其子目录，默认当前目录。
  --remote NAME              同时核对远端 refs/tags；不代表会推送。
  --root TAG                指定历史根标签，覆盖自动识别的唯一 init 标签。
  --confirm-remote-rewrite  与 --apply、--remote 一起使用，明确授权远程标签重写。
  --help                     显示帮助。

排序与安全规则:
  1. 唯一的“<tag> init,”消息作为根；之后按 peeled commit 的拓扑顺序排列。
  2. 同一 peeled commit 内按 tagger 时间升序；时间并列、无法验证共同 follow 祖先的分支、缺失根、轻量/签名标签不自动改写。
  3. 远程应用使用 --force-with-lease，远端对象发生漂移时拒绝覆盖。
USAGE
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 2
}

is_managed_tag() {
    [[ "$1" =~ ^(stab|dev|bug|test)[0-9]+(_[0-9]+)*$ ]]
}

repo_path=''
remote_name=''
root_override=''
mode='check'
apply=0
confirm_remote=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --check)
            mode='check'
            ;;
        --repair)
            mode='repair'
            ;;
        --dry-run)
            mode='repair'
            ;;
        --apply)
            mode='repair'
            apply=1
            ;;
        --repo)
            shift
            [ "$#" -gt 0 ] || die '--repo 缺少 PATH'
            repo_path=$1
            ;;
        --remote)
            shift
            [ "$#" -gt 0 ] || die '--remote 缺少 NAME'
            remote_name=$1
            ;;
        --root)
            shift
            [ "$#" -gt 0 ] || die '--root 缺少 TAG'
            root_override=$1
            ;;
        --confirm-remote-rewrite)
            confirm_remote=1
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            die "未知选项: $1"
            ;;
    esac
    shift
done

if [ "$confirm_remote" -eq 1 ] && { [ "$apply" -ne 1 ] || [ -z "$remote_name" ]; }; then
    die '--confirm-remote-rewrite 必须与 --apply --remote NAME 一起使用'
fi

script_path=$0
case "$script_path" in
    /*) ;;
    *) script_path="$(pwd)/$script_path" ;;
esac

if [ -n "$repo_path" ]; then
    cd "$repo_path" || die "无法进入仓库路径: $repo_path"
fi

repo_root=$(git rev-parse --show-toplevel 2>/dev/null) || die '当前路径不在 Git 工作树中'
cd "$repo_root" || die "无法进入 Git 根目录: $repo_root"

tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/tag-chain.XXXXXX") || die '无法创建临时目录'
cleanup() {
    rm -rf "$tmp_dir"
}
trap cleanup EXIT HUP INT TERM

tab=$(printf '\t')
target_queries="$tmp_dir/target-queries"
target_results="$tmp_dir/target-results"
commit_roots="$tmp_dir/commit-roots"
commit_order="$tmp_dir/commit-order"
order_records="$tmp_dir/order-records"
sorted_order="$tmp_dir/sorted-order"
remote_refs="$tmp_dir/remote-refs"
: > "$target_queries"
: > "$commit_roots"

kv_set() {
    local map_name=$1
    local key=$2
    local value=${3-}
    printf '%s\t%s\n' "$key" "$value" >> "$tmp_dir/kv-$map_name"
}

kv_get() {
    local map_name=$1
    local key=$2
    local file="$tmp_dir/kv-$map_name"
    [ -f "$file" ] || return 1
    awk -F "$tab" -v key="$key" '
        $1 == key {
            value = $0
            sub(/^[^\t]*\t/, "", value)
            found = 1
        }
        END {
            if (found) {
                print value
            } else {
                exit 1
            }
        }
    ' "$file"
}

kv_value() {
    kv_get "$@" 2>/dev/null || true
}

kv_has() {
    kv_get "$@" >/dev/null 2>&1
}

for map_name in \
    tag_object_oid tag_object_type tag_created tag_subject tag_target \
    tag_target_type tag_payload_file tag_payload_loaded tag_rank \
    commit_rank ancestor_cache repair_expected_parent repair_kind \
    seen_parents remote_object_by_tag remote_target_by_tag repair_new_oid; do
    : > "$tmp_dir/kv-$map_name"
done

declare -a tag_names=()

tag_count=0
init_count=0
init_name=''
record_index=0

# for-each-ref supplies all inexpensive metadata in one pass.  The raw tag
# payload is loaded lazily only for a candidate that may actually be rewritten.
while IFS="$tab" read -r tag_name object_oid object_type created subject; do
    [ -n "$tag_name" ] || continue
    is_managed_tag "$tag_name" || continue

    tag_names+=("$tag_name")
    kv_set tag_object_oid "$tag_name" "$object_oid"
    kv_set tag_object_type "$tag_name" "$object_type"
    kv_set tag_created "$tag_name" "${created:-0}"
    kv_set tag_subject "$tag_name" "$subject"
    kv_set tag_payload_file "$tag_name" "$tmp_dir/payload-original-$record_index"
    kv_set tag_payload_loaded "$tag_name" 0
    printf 'refs/tags/%s^{}\n' "$tag_name" >> "$target_queries"
    record_index=$((record_index + 1))

    case "$subject" in
        "$tag_name init,"*)
            init_count=$((init_count + 1))
            init_name=$tag_name
            ;;
    esac
done < <(git for-each-ref \
    --format='%(refname:short)%09%(objectname)%09%(objecttype)%09%(creatordate:unix)%09%(contents:subject)' \
    refs/tags)

tag_count=${#tag_names[@]}

if [ "$tag_count" -eq 0 ]; then
    printf 'Repository: %s\n' "$repo_root"
    printf 'PASS: 未发现 stab/dev/bug/test 标签。\n'
    exit 0
fi

# Resolve every tag's peeled object with one batch request.  A non-commit
# result is retained for diagnostics but can never be an auto-repair target.
git cat-file --batch-check='%(objectname) %(objecttype)' < "$target_queries" > "$target_results" || \
    die '无法批量解析标签目标'
target_index=0
while IFS=' ' read -r target_oid target_type; do
    [ "$target_index" -lt "$tag_count" ] || break
    tag_name=${tag_names[$target_index]}
    if [ "$target_type" = 'missing' ]; then
        kv_set tag_target "$tag_name" ''
        kv_set tag_target_type "$tag_name" missing
    else
        kv_set tag_target "$tag_name" "$target_oid"
        kv_set tag_target_type "$tag_name" "$target_type"
    fi
    target_index=$((target_index + 1))
done < "$target_results"
while [ "$target_index" -lt "$tag_count" ]; do
    tag_name=${tag_names[$target_index]}
    kv_set tag_target "$tag_name" ''
    kv_set tag_target_type "$tag_name" missing
    target_index=$((target_index + 1))
done

if [ -n "$root_override" ]; then
    is_managed_tag "$root_override" || die "根标签名称不符合约定: $root_override"
    kv_has tag_object_oid "$root_override" || die "指定根标签不存在: $root_override"
    root_name=$root_override
elif [ "$init_count" -eq 1 ]; then
    root_name=$init_name
elif [ "$init_count" -eq 0 ]; then
    printf 'Repository: %s\n' "$repo_root"
    printf 'BLOCK: 未找到唯一 init 标签；请用 --root TAG 指定历史根。\n'
    exit 1
else
    printf 'Repository: %s\n' "$repo_root"
    printf 'BLOCK: 找到 %s 个 init 标签；请用 --root TAG 消除根歧义。\n' "$init_count"
    exit 1
fi

# A single reverse topological walk gives every target commit a stable rank.
# With --reverse, parents precede children.  Incomparable branches are kept in
# deterministic order, then accepted only when the tag message names a
# managed common ancestor explicitly (see is_branch_relation below).
for tag_name in "${tag_names[@]}"; do
    target_type=$(kv_value tag_target_type "$tag_name")
    target_oid=$(kv_value tag_target "$tag_name")
    if [ "$target_type" = 'commit' ] && [ -n "$target_oid" ]; then
        printf '%s\n' "$target_oid" >> "$commit_roots"
    fi
done

if [ -s "$commit_roots" ]; then
    git rev-list --topo-order --reverse --stdin < "$commit_roots" > "$commit_order" || \
        die '无法生成标签目标提交的拓扑顺序'
    commit_index=0
    while IFS= read -r commit_oid; do
        [ -n "$commit_oid" ] || continue
        if ! kv_has commit_rank "$commit_oid"; then
            kv_set commit_rank "$commit_oid" "$commit_index"
            commit_index=$((commit_index + 1))
        fi
    done < "$commit_order"
fi

: > "$order_records"
for tag_name in "${tag_names[@]}"; do
    [ "$tag_name" = "$root_name" ] && continue
    target_oid=$(kv_value tag_target "$tag_name")
    if [ -n "$target_oid" ] && kv_has commit_rank "$target_oid"; then
        kv_set tag_rank "$tag_name" "$(kv_value commit_rank "$target_oid")"
    else
        # Keep malformed/non-commit tags deterministic and after valid targets.
        kv_set tag_rank "$tag_name" 2147483647
    fi
    printf '%s\t%s\t%s\n' \
        "$(kv_value tag_rank "$tag_name")" "$(kv_value tag_created "$tag_name")" "$tag_name" >> "$order_records"
done
sort -t "$tab" -k1,1n -k2,2n -k3,3 "$order_records" > "$sorted_order" || die '标签拓扑排序失败'

declare -a ordered_tags=("$root_name")
while IFS="$tab" read -r rank created tag_name; do
    [ -n "$tag_name" ] || continue
    ordered_tags+=("$tag_name")
done < "$sorted_order"

printf 'Repository: %s\n' "$repo_root"
printf 'Root: %s\n' "$root_name"
printf 'Order: init root, then peeled-commit topology; same target tagger timestamp ascending\n'
printf 'Managed tags: %s\n' "$tag_count"

issue_count=0
blocking_count=0
repair_count=0
order_tie=0
previous_date=''
previous_name=''
previous_target=''

is_ancestor() {
    local parent=$1
    local child=$2
    local cache_key="${parent}:${child}"
    if kv_has ancestor_cache "$cache_key"; then
        [ "$(kv_value ancestor_cache "$cache_key")" = 1 ]
        return
    fi
    if git merge-base --is-ancestor "$parent" "$child" >/dev/null 2>&1; then
        kv_set ancestor_cache "$cache_key" 1
        return 0
    fi
    kv_set ancestor_cache "$cache_key" 0
    return 1
}

# Two adjacent tags may be legitimate siblings: both targets descend from the
# tag named by the current tag's explicit `follow` prefix, while neither target
# descends from the other.  This is the history shape used by the old dev37 /
# dev38 and dev72 / bug30 lines.  Accepting it preserves both published
# snapshots without pretending that one branch is an ancestor of the other.
is_branch_relation() {
    local previous_name=$1
    local current_name=$2
    local parent_name=$3
    local previous_target=$4
    local current_target=$5
    local parent_target

    [ -n "$parent_name" ] || return 1
    [ "$parent_name" != "$previous_name" ] || return 1
    [ "$parent_name" != "$current_name" ] || return 1
    kv_has tag_target "$parent_name" || return 1
    parent_target=$(kv_value tag_target "$parent_name")
    [ -n "$parent_target" ] || return 1
    [ -n "$previous_target" ] || return 1
    [ -n "$current_target" ] || return 1
    is_ancestor "$parent_target" "$previous_target" || return 1
    is_ancestor "$parent_target" "$current_target" || return 1
    is_ancestor "$previous_target" "$current_target" && return 1
    is_ancestor "$current_target" "$previous_target" && return 1
    return 0
}

declare -a repair_tags=()
queue_repair() {
    local name=$1
    local expected=$2
    local kind=$3
    local payload_file

    if [ "$(kv_value tag_object_type "$name")" != 'tag' ] || \
        [ "$(kv_value tag_target_type "$name")" != 'commit' ] || \
        [ -z "$(kv_value tag_target "$name")" ] || [ -z "$(kv_value tag_subject "$name")" ]; then
        printf '[BLOCK] %s 不是可安全改写的有效附注标签\n' "$name"
        blocking_count=$((blocking_count + 1))
        return
    fi

    payload_file=$(kv_value tag_payload_file "$name")
    if [ "$(kv_value tag_payload_loaded "$name")" -ne 1 ]; then
        kv_set tag_payload_loaded "$name" 1
        if ! git cat-file tag "$name" > "$payload_file" 2>/dev/null; then
            : > "$payload_file"
        fi
    fi
    if [ ! -s "$payload_file" ]; then
        printf '[BLOCK] %s 无法读取原始附注对象\n' "$name"
        blocking_count=$((blocking_count + 1))
    elif grep -q '^gpgsig ' "$payload_file"; then
        printf '[BLOCK] %s 含签名，自动改写会破坏签名\n' "$name"
        blocking_count=$((blocking_count + 1))
    else
        repair_tags+=("$name")
        kv_set repair_expected_parent "$name" "$expected"
        kv_set repair_kind "$name" "$kind"
        repair_count=$((repair_count + 1))
    fi
}

for tag_name in "${ordered_tags[@]}"; do
    object_type=$(kv_value tag_object_type "$tag_name")
    target_oid=$(kv_value tag_target "$tag_name")
    target_type=$(kv_value tag_target_type "$tag_name")
    target_type=${target_type:-missing}
    subject=$(kv_value tag_subject "$tag_name")

    if [ "$object_type" != 'tag' ]; then
        printf '[BLOCK] %s 不是附注标签（object type=%s）\n' "$tag_name" "$object_type"
        issue_count=$((issue_count + 1))
        blocking_count=$((blocking_count + 1))
    fi
    if [ "$target_type" != 'commit' ] || [ -z "$target_oid" ]; then
        printf '[BLOCK] %s 无法解析 peeled commit（target type=%s）\n' "$tag_name" "$target_type"
        issue_count=$((issue_count + 1))
        blocking_count=$((blocking_count + 1))
    fi

    if [ -z "$previous_name" ]; then
        case "$subject" in
            "$tag_name init,"*) ;;
            *)
                printf '[BLOCK] 根标签 %s 缺少“%s init,”前缀\n' "$tag_name" "$tag_name"
                issue_count=$((issue_count + 1))
                blocking_count=$((blocking_count + 1))
                ;;
        esac
    else
        actual_parent=''
        if [[ "$subject" =~ ^follow[[:space:]]+([^,]+), ]]; then
            actual_parent=${BASH_REMATCH[1]}
        fi

        branch_relation=0
        if [ -z "$previous_target" ] || [ -z "$target_oid" ]; then
            printf '[BLOCK] %s -> %s 缺少可验证的提交目标\n' "$previous_name" "$tag_name"
            issue_count=$((issue_count + 1))
            blocking_count=$((blocking_count + 1))
        elif [ "$previous_target" != "$target_oid" ] && ! is_ancestor "$previous_target" "$target_oid"; then
            if is_branch_relation "$previous_name" "$tag_name" "$actual_parent" \
                "$previous_target" "$target_oid"; then
                branch_relation=1
                printf '[BRANCH] %s -> %s：分叉目标均由 follow %s 派生，保留并行标签目标\n' \
                    "$previous_name" "$tag_name" "$actual_parent"
            else
                printf '[BLOCK] %s 的 peeled commit 不是 %s 的祖先，且没有可验证的共同 follow 祖先\n' \
                    "$previous_name" "$tag_name"
                issue_count=$((issue_count + 1))
                blocking_count=$((blocking_count + 1))
            fi
        elif [ "$previous_target" = "$target_oid" ] && [ "$previous_date" = "$(kv_value tag_created "$tag_name")" ]; then
            printf '[BLOCK] 同一 peeled commit 的 tagger 时间并列: %s 与 %s (%s)\n' \
                "$previous_name" "$tag_name" "$(kv_value tag_created "$tag_name")"
            issue_count=$((issue_count + 1))
            blocking_count=$((blocking_count + 1))
            order_tie=1
        fi

        expected_parent=$previous_name

        if [ -z "$actual_parent" ]; then
            printf '[FIX] %s: 缺少 follow 前缀 -> 补 follow %s\n' "$tag_name" "$expected_parent"
            issue_count=$((issue_count + 1))
            queue_repair "$tag_name" "$expected_parent" 'prepend'
        elif [ "$branch_relation" -eq 0 ] && [ "$actual_parent" != "$expected_parent" ]; then
            printf '[FIX] %s: follow %s -> follow %s\n' "$tag_name" "$actual_parent" "$expected_parent"
            issue_count=$((issue_count + 1))
            queue_repair "$tag_name" "$expected_parent" 'replace'
        fi

        if [ -n "$actual_parent" ] && kv_has seen_parents "$actual_parent"; then
            if [ "$branch_relation" -eq 1 ]; then
                printf '[BRANCH] %s 重复引用共同 follow %s，视为并行分支而非链回指\n' \
                    "$tag_name" "$actual_parent"
            else
                printf '[DUPLICATE] %s 重复引用 follow %s（随前缀修正一并消除）\n' "$tag_name" "$actual_parent"
                issue_count=$((issue_count + 1))
            fi
        fi
        if [ -n "$actual_parent" ]; then
            kv_set seen_parents "$actual_parent" 1
        fi
    fi

    previous_date=$(kv_value tag_created "$tag_name")
    previous_name=$tag_name
    previous_target=$target_oid
done

if [ "$order_tie" -eq 1 ]; then
    repair_count=0
fi

if [ -n "$remote_name" ]; then
    if ! git ls-remote --tags "$remote_name" > "$remote_refs" 2> "$tmp_dir/remote-error"; then
        printf '[BLOCK] 无法读取远端 %s: %s\n' "$remote_name" "$(sed -n '1p' "$tmp_dir/remote-error")"
        issue_count=$((issue_count + 1))
        blocking_count=$((blocking_count + 1))
    else
        # Parse remote refs once instead of running awk once per local tag.
        while read -r remote_oid remote_ref; do
            [ -n "$remote_ref" ] || continue
            [[ "$remote_ref" == refs/tags/* ]] || continue
            remote_tag=${remote_ref#refs/tags/}
            if [[ "$remote_tag" == *'^{}' ]]; then
                remote_tag=${remote_tag%'^{}'}
                kv_set remote_target_by_tag "$remote_tag" "$remote_oid"
            else
                kv_set remote_object_by_tag "$remote_tag" "$remote_oid"
            fi
        done < "$remote_refs"

        for tag_name in "${ordered_tags[@]}"; do
            remote_object=$(kv_value remote_object_by_tag "$tag_name")
            remote_target=$(kv_value remote_target_by_tag "$tag_name")
            object_oid=$(kv_value tag_object_oid "$tag_name")
            target_oid=$(kv_value tag_target "$tag_name")
            if [ -z "$remote_object" ]; then
                printf '[REMOTE] %s: 远端缺失\n' "$tag_name"
                issue_count=$((issue_count + 1))
                blocking_count=$((blocking_count + 1))
            elif [ "$remote_object" != "$object_oid" ] || [ "$remote_target" != "$target_oid" ]; then
                printf '[REMOTE] %s: local object/target=%s/%s, remote=%s/%s\n' \
                    "$tag_name" "$object_oid" "$target_oid" "$remote_object" "${remote_target:-missing}"
                issue_count=$((issue_count + 1))
                blocking_count=$((blocking_count + 1))
            fi
        done
    fi
fi

if [ "$mode" = 'check' ]; then
    if [ "$issue_count" -eq 0 ]; then
        printf 'PASS: follow 链、提交祖先关系%s均正常。\n' \
            "${remote_name:+，远端 $remote_name 对象一致}"
        exit 0
    fi
    printf 'FAIL: 发现 %s 个问题；可安全修正候选 %s 个。\n' "$issue_count" "$repair_count"
    exit 1
fi

if [ "$repair_count" -eq 0 ]; then
    if [ "$issue_count" -eq 0 ]; then
        printf 'PASS: 没有需要修正的 follow 链。\n'
        exit 0
    fi
    printf 'STOP: 没有可安全自动修正的候选。\n'
    exit 1
fi

printf 'Repair mode: %s candidate(s), %s blocking issue(s)\n' "$repair_count" "$blocking_count"
if [ "$apply" -ne 1 ]; then
    printf 'DRY-RUN: 未写入任何本地或远程标签。\n'
    exit 1
fi

if [ "$blocking_count" -ne 0 ]; then
    printf 'STOP: 存在阻断项，未应用任何修正。\n'
    exit 1
fi

if [ -n "$remote_name" ] && [ "$confirm_remote" -ne 1 ]; then
    printf 'STOP: 指定了远端但未提供 --confirm-remote-rewrite，未改写远端。\n'
    exit 1
fi

declare -a prepared_tags=()
candidate_index=0
for tag_name in "${repair_tags[@]}"; do
    candidate_index=$((candidate_index + 1))
    original_payload=$(kv_value tag_payload_file "$tag_name")
    new_payload="$tmp_dir/payload-new-$candidate_index"
    expected_parent=$(kv_value repair_expected_parent "$tag_name")
    kind=$(kv_value repair_kind "$tag_name")

    if grep -q '^gpgsig ' "$original_payload"; then
        printf 'STOP: %s 含签名，未应用任何修正。\n' "$tag_name"
        exit 1
    fi

    if ! awk -v parent="$expected_parent" -v kind="$kind" '
        BEGIN { in_body = 0; replaced = 0 }
        {
            if (!in_body) {
                print
                if ($0 == "") {
                    in_body = 1
                    if (kind == "prepend") {
                        printf "follow %s, ", parent
                    }
                }
                next
            }
            if (kind == "replace" && !replaced &&
                $0 ~ /^follow[[:space:]]*[^,]*,/) {
                line = $0
                sub(/^follow[[:space:]]*[^,]*,/, "follow " parent ",", line)
                print line
                replaced = 1
                next
            }
            print
        }
    ' "$original_payload" > "$new_payload"; then
        printf 'STOP: %s 无法生成新附注负载。\n' "$tag_name"
        exit 1
    fi

    new_oid=$(git mktag < "$new_payload" 2> "$tmp_dir/mktag-error-$candidate_index") || {
        printf 'STOP: %s 无法创建新附注对象: %s\n' \
            "$tag_name" "$(sed -n '1p' "$tmp_dir/mktag-error-$candidate_index")"
        exit 1
    }
    new_target=$(git rev-parse -q --verify "$new_oid^{}" 2>/dev/null) || new_target=''
    old_target=$(kv_value tag_target "$tag_name")
    if [ "$new_target" != "$old_target" ]; then
        printf 'STOP: %s 新附注对象的 peeled commit 改变（%s -> %s）\n' \
            "$tag_name" "$old_target" "${new_target:-missing}"
        exit 1
    fi
    printf '[PLAN] %s: object %s -> %s; peeled=%s\n' \
        "$tag_name" "$(kv_value tag_object_oid "$tag_name")" "$new_oid" "$old_target"
    prepared_tags+=("$tag_name")
    kv_set repair_new_oid "$tag_name" "$new_oid"
done

if [ -n "$remote_name" ]; then
    push_args=()
    refspecs=()
    for tag_name in "${prepared_tags[@]}"; do
        push_args+=("--force-with-lease=refs/tags/$tag_name:$(kv_value tag_object_oid "$tag_name")")
        refspecs+=("$(kv_value repair_new_oid "$tag_name"):refs/tags/$tag_name")
    done
    # The explicit remote confirmation above is the destructive-operation gate.
    # --atomic prevents a multi-tag repair from leaving a partially rewritten
    # remote when the server supports atomic receive-pack.
    if ! git push --atomic "${push_args[@]}" "$remote_name" "${refspecs[@]}"; then
        printf 'STOP: 远端标签推送失败，未更新本地 refs/tags。\n'
        exit 1
    fi
    printf 'REMOTE: 已用 force-with-lease 原子改写 %s 的 %s 个标签。\n' \
        "$remote_name" "${#prepared_tags[@]}"
fi

updates="$tmp_dir/updates"
: > "$updates"
for tag_name in "${prepared_tags[@]}"; do
    printf 'update refs/tags/%s %s %s\n' \
        "$tag_name" "$(kv_value repair_new_oid "$tag_name")" "$(kv_value tag_object_oid "$tag_name")" >> "$updates"
done
if ! git update-ref --stdin < "$updates"; then
    printf 'STOP: 远端已更新但本地 refs/tags 更新失败，请按清单手工执行 update-ref。\n'
    exit 1
fi
printf 'LOCAL: 已保留 peeled commit，改写本地 %s 个标签注释。\n' "${#prepared_tags[@]}"

check_args=(--check --repo "$repo_root")
[ -n "$root_override" ] && check_args+=(--root "$root_override")
[ -n "$remote_name" ] && check_args+=(--remote "$remote_name")
"$script_path" "${check_args[@]}"
