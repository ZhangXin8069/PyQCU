# AGENTS.md — skills/pyqcu-solver/references

`pyqcu-solver` 的按需算法参考；主流程与触发条件以父目录 `SKILL.md` 为准。

- `mg-cache-comparison.md`：Legacy 33-tensor cache 与 Strict runtime cache 的结构、
  identity 与复用边界。
- `legacy-mg-corrector-history.md`：2026-11-25 删除的 KUPL Legacy-MG 修正器路径
  （修正间隔、阻尼、`best_ratio`、MG/plain 倍数）；只在问历史时读。
- 更新 cache 结论时同步代码协议、统一接口说明和实测证据；不得把配置不匹配的 cache
  描述为可复用。
