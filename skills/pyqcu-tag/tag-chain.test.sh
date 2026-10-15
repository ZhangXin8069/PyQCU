#!/usr/bin/env bash

set -u
set -o pipefail
export LC_ALL=C

script_dir=$(CDPATH= cd "$(dirname "$0")" && pwd)
checker="$script_dir/tag-chain.sh"
test_root=$(mktemp -d "${TMPDIR:-/tmp}/tag-chain-test.XXXXXX")

cleanup() {
    rm -rf "$test_root"
}
trap cleanup EXIT HUP INT TERM

fail() {
    printf 'FAIL: %s\n' "$*" >&2
    exit 1
}

assert_contains() {
    case "$1" in
        *"$2"*) ;;
        *) fail "输出缺少: $2" ;;
    esac
}

assert_not_contains() {
    case "$1" in
        *"$2"*) fail "输出不应包含: $2" ;;
        *) ;;
    esac
}

repo="$test_root/repo"
remote="$test_root/remote.git"
git init -q "$repo" || fail '初始化测试仓库失败'
git -C "$repo" config user.name 'Tag Chain Test'
git -C "$repo" config user.email 'tag-chain-test@example.invalid'

commit_file() {
    local message=$1
    local stamp=$2
    printf '%s\n' "$message" >> "$repo/history.txt"
    git -C "$repo" add history.txt || fail "暂存失败: $message"
    GIT_AUTHOR_DATE="$stamp" GIT_COMMITTER_DATE="$stamp" \
        git -C "$repo" commit -q -m "$message" || fail "提交失败: $message"
}

make_tag() {
    local tag_name=$1
    local stamp=$2
    local message=$3
    local target=$4
    GIT_COMMITTER_DATE="$stamp" git -C "$repo" tag -a "$tag_name" "$target" -m "$message" || \
        fail "创建标签失败: $tag_name"
}

commit_file init '2024-01-01T00:00:00Z'
root_commit=$(git -C "$repo" rev-parse HEAD)
# The root tagger date is intentionally later than its descendants.  init,
# rather than the timestamp, must anchor the chain.
make_tag stab0 '2030-01-01T00:00:00Z' 'stab0 init, 1. init; [test].' "$root_commit"
root_tag_date=$(git -C "$repo" for-each-ref --format='%(creatordate:unix)' refs/tags/stab0)

commit_file dev '2024-01-02T00:00:00Z'
dev_commit=$(git -C "$repo" rev-parse HEAD)
make_tag dev0 '2024-01-02T00:00:01Z' 'follow stab0, 1. dev; [test].' "$dev_commit"

commit_file stable '2024-01-03T00:00:00Z'
stable_commit=$(git -C "$repo" rev-parse HEAD)
make_tag stab1 '2024-01-01T00:00:01Z' 'follow stab0, 1. stable; [test].' "$stable_commit"

commit_file bug '2024-01-04T00:00:00Z'
bug_commit=$(git -C "$repo" rev-parse HEAD)
make_tag bug0 '2024-01-04T00:00:01Z' 'NVIDIA CUDA 默认线程块性能调优与显存回归; [test].' "$bug_commit"

commit_file multi '2024-01-05T00:00:00Z'
multi_commit=$(git -C "$repo" rev-parse HEAD)
make_tag test10_5 '2024-01-05T00:00:01Z' 'follow bug0, 1. multi; [test].' "$multi_commit"

commit_file multi-sub '2024-01-06T00:00:00Z'
multi_sub_commit=$(git -C "$repo" rev-parse HEAD)
make_tag test10_5_1 '2024-01-06T00:00:01Z' 'follow test10_5, 1. multi-sub; [test].' "$multi_sub_commit"

commit_file multi-sub-sub '2024-01-07T00:00:00Z'
multi_sub_sub_commit=$(git -C "$repo" rev-parse HEAD)
make_tag test10_5_1_1 '2024-01-07T00:00:01Z' 'follow test10_5_1, 1. multi-sub-sub; [test].' "$multi_sub_sub_commit"

commit_file multi-next '2024-01-08T00:00:00Z'
multi_next_commit=$(git -C "$repo" rev-parse HEAD)
make_tag test10_6 '2024-01-08T00:00:01Z' 'follow test10_5_1_1, 1. multi-next; [test].' "$multi_next_commit"

if output=$(bash "$checker" --check --repo "$repo" 2>&1); then
    fail '错误链检查意外通过'
fi
assert_contains "$output" '[FIX] stab1: follow stab0 -> follow dev0'
assert_contains "$output" '[FIX] bug0: 缺少 follow 前缀 -> 补 follow stab1'
assert_not_contains "$output" '[FIX] test10_6:'

old_stable_object=$(git -C "$repo" rev-parse refs/tags/stab1)
if output=$(bash "$checker" --repair --dry-run --repo "$repo" 2>&1); then
    fail 'dry-run 意外返回成功'
fi
assert_contains "$output" 'DRY-RUN: 未写入任何本地或远程标签。'
[ "$(git -C "$repo" rev-parse refs/tags/stab1)" = "$old_stable_object" ] || \
    fail 'dry-run 改写了本地标签'

bash "$checker" --repair --apply --repo "$repo" >/dev/null || fail '本地修正失败'
subject=$(git -C "$repo" for-each-ref --format='%(contents:subject)' refs/tags/stab1)
[ "$subject" = 'follow dev0, 1. stable; [test].' ] || fail "stab1 消息错误: $subject"
subject=$(git -C "$repo" for-each-ref --format='%(contents:subject)' refs/tags/bug0)
[ "$subject" = 'follow stab1, NVIDIA CUDA 默认线程块性能调优与显存回归; [test].' ] || fail "bug0 消息错误: $subject"
[ "$(git -C "$repo" for-each-ref --format='%(contents:subject)' refs/tags/test10_5_1_1)" = \
    'follow test10_5_1, 1. multi-sub-sub; [test].' ] || fail '多级标签消息错误'
[ "$(git -C "$repo" rev-parse 'stab1^{}')" = "$stable_commit" ] || fail 'stab1 peeled commit 被改变'
[ "$(git -C "$repo" rev-parse 'bug0^{}')" = "$bug_commit" ] || fail 'bug0 peeled commit 被改变'
[ "$(git -C "$repo" rev-parse 'test10_5_1_1^{}')" = "$multi_sub_sub_commit" ] || fail '多级标签 peeled commit 被改变'
[ "$(git -C "$repo" for-each-ref --format='%(creatordate:unix)' refs/tags/stab0)" = \
    "$root_tag_date" ] || fail '根标签 tagger 日期未保留'

local_refs_before=$(git -C "$repo" show-ref --tags)
bash "$checker" --repair --apply --repo "$repo" >/dev/null || fail '重复本地修正失败'
local_refs_after=$(git -C "$repo" show-ref --tags)
[ "$local_refs_after" = "$local_refs_before" ] || fail '重复修正改变了本地标签 refs'

commit_file remote '2024-01-05T00:00:00Z'
remote_commit=$(git -C "$repo" rev-parse HEAD)
make_tag dev1 '2024-01-05T00:00:01Z' 'follow stab0, 1. remote; [test].' "$remote_commit"
git init -q --bare "$remote" || fail '初始化测试远端失败'
git -C "$repo" remote add origin "$remote"
git -C "$repo" push -q --set-upstream origin HEAD
git -C "$repo" push -q origin --tags

if ! output=$(bash "$checker" --repair --apply --repo "$repo" --remote origin \
    --confirm-remote-rewrite 2>&1); then
    fail "远端修正失败: $output"
fi
remote_subject=$(git -C "$repo" for-each-ref --format='%(contents:subject)' refs/tags/dev1)
[ "$remote_subject" = 'follow test10_6, 1. remote; [test].' ] || fail "dev1 本地消息错误: $remote_subject"
[ "$(git -C "$repo" rev-parse 'dev1^{}')" = "$remote_commit" ] || fail 'dev1 peeled commit 被改变'
bash "$checker" --check --repo "$repo" --remote origin >/dev/null || fail '远端修正后的检查失败'

branch_repo="$test_root/branch-repo"
git init -q "$branch_repo" || fail '初始化分支目标测试仓库失败'
git -C "$branch_repo" config user.name 'Tag Chain Branch Test'
git -C "$branch_repo" config user.email 'tag-chain-branch@example.invalid'
printf '%s\n' root >> "$branch_repo/history.txt"
git -C "$branch_repo" add history.txt || fail '分支根提交暂存失败'
GIT_AUTHOR_DATE='2024-03-01T00:00:00Z' GIT_COMMITTER_DATE='2024-03-01T00:00:00Z' \
    git -C "$branch_repo" commit -q -m root || fail '分支根提交失败'
branch_root_commit=$(git -C "$branch_repo" rev-parse HEAD)
GIT_COMMITTER_DATE='2024-03-01T00:00:01Z' git -C "$branch_repo" tag -a stab0 "$branch_root_commit" \
    -m 'stab0 init, 1. init; [test].' || fail '分支根标签失败'

printf '%s\n' main >> "$branch_repo/history.txt"
git -C "$branch_repo" add history.txt || fail '主线提交暂存失败'
GIT_AUTHOR_DATE='2024-03-02T00:00:00Z' GIT_COMMITTER_DATE='2024-03-02T00:00:00Z' \
    git -C "$branch_repo" commit -q -m main || fail '主线提交失败'
branch_main_commit=$(git -C "$branch_repo" rev-parse HEAD)
GIT_COMMITTER_DATE='2024-03-02T00:00:01Z' git -C "$branch_repo" tag -a dev0 "$branch_main_commit" \
    -m 'follow stab0, 1. main; [test].' || fail '主线标签失败'

git -C "$branch_repo" checkout -q -b side "$branch_root_commit" || fail '创建分支失败'
printf '%s\n' side >> "$branch_repo/history.txt"
git -C "$branch_repo" add history.txt || fail '侧线提交暂存失败'
GIT_AUTHOR_DATE='2024-03-03T00:00:00Z' GIT_COMMITTER_DATE='2024-03-03T00:00:00Z' \
    git -C "$branch_repo" commit -q -m side || fail '侧线提交失败'
branch_side_commit=$(git -C "$branch_repo" rev-parse HEAD)
GIT_COMMITTER_DATE='2024-03-03T00:00:01Z' git -C "$branch_repo" tag -a test0 "$branch_side_commit" \
    -m 'follow stab0, 1. side; [test].' || fail '侧线标签失败'
branch_refs_before=$(git -C "$branch_repo" show-ref --tags)
if ! output=$(bash "$checker" --check --repo "$branch_repo" 2>&1); then
    fail "合法分支目标检查失败: $output"
fi
assert_contains "$output" '[BRANCH]'
assert_not_contains "$output" '[BLOCK]'
bash "$checker" --repair --apply --repo "$branch_repo" >/dev/null || fail '合法分支目标修正失败'
[ "$(git -C "$branch_repo" show-ref --tags)" = "$branch_refs_before" ] || fail '分支目标场景改写了本地标签'

tie_repo="$test_root/tie-repo"
git init -q "$tie_repo" || fail '初始化并列时间测试仓库失败'
git -C "$tie_repo" config user.name 'Tag Chain Tie Test'
git -C "$tie_repo" config user.email 'tag-chain-tie@example.invalid'
printf '%s\n' tie >> "$tie_repo/history.txt"
git -C "$tie_repo" add history.txt || fail '并列时间测试暂存失败'
GIT_AUTHOR_DATE='2024-02-01T00:00:00Z' GIT_COMMITTER_DATE='2024-02-01T00:00:00Z' \
    git -C "$tie_repo" commit -q -m tie || fail '并列时间测试提交失败'
tie_root_commit=$(git -C "$tie_repo" rev-parse HEAD)
GIT_COMMITTER_DATE='2024-02-01T00:00:01Z' git -C "$tie_repo" tag -a stab0 "$tie_root_commit" \
    -m 'stab0 init, 1. init; [test].' || fail '并列时间根标签失败'

printf '%s\n' child >> "$tie_repo/history.txt"
git -C "$tie_repo" add history.txt || fail '并列时间测试子提交暂存失败'
GIT_AUTHOR_DATE='2024-02-02T00:00:00Z' GIT_COMMITTER_DATE='2024-02-02T00:00:00Z' \
    git -C "$tie_repo" commit -q -m child || fail '并列时间测试子提交失败'
tie_child_commit=$(git -C "$tie_repo" rev-parse HEAD)
GIT_COMMITTER_DATE='2024-02-02T00:00:01Z' git -C "$tie_repo" tag -a dev0 "$tie_child_commit" \
    -m 'follow stab0, 1. first; [test].' || fail '并列时间 dev 标签失败'
GIT_COMMITTER_DATE='2024-02-02T00:00:01Z' git -C "$tie_repo" tag -a test0 "$tie_child_commit" \
    -m 'follow dev0, 1. second; [test].' || fail '并列时间 test 标签失败'
tie_refs_before=$(git -C "$tie_repo" show-ref --tags)
if output=$(bash "$checker" --repair --apply --repo "$tie_repo" 2>&1); then
    fail '并列 tagger 时间意外允许自动修正'
fi
assert_contains "$output" 'tagger 时间并列'
[ "$(git -C "$tie_repo" show-ref --tags)" = "$tie_refs_before" ] || fail '并列时间场景改写了本地标签'

printf 'PASS: tag-chain 自检、dry-run、本地修正和远端 lease 重写测试通过。\n'
