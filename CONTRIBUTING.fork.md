# CONTRIBUTING.fork — 上游 PR 规范补充页（fork 内部使用）

> 本文件是 Lmy271828/FlashRT fork 的内部工作规范，浓缩自上游
> `CONTRIBUTING.md`、`docs/pr_review_checklist.md`、`docs/adding_new_model.md`
> 及 Jetson-PI 集成案例（issue #143 / PR #148 / #157）的实际评审行为。
> **本文件本身永远不进入任何上游 PR 的 diff。**

我们的上游化目标：Omega-QVLA GPTQ pack 消费层（E0M3 转换器 + per-step scale
消费 + 低秩 epilogue），目标硬件 Jetson AGX Thor (SM110)。

## 1. 工程规范（写代码时）

- kernel 命名必须带 ownership 前缀（模型/硬件/特性），禁止裸通用名，
  禁止 `_v4` / `_new` / `_fast` 后缀（pr_review_checklist:277-327）
- `CMakeLists.txt` 是唯一 kernel 注册点；arch-gated 的 `.cu` 必须配同条件
  gate 的 binding；未构建的符号必须有无条件 stub 抛清晰错误
  （先例：`bindings.cpp` 的 "was not built"）
- 大组模型专属 kernel 独立成模块 `flash_rt_<model>_kernels`
  （先例：`flash_rt_fp4`、`flash_rt_qwen3_vl_kernels`）
- frontend 禁止运行时 `if arch==` 分支，走 `_PIPELINE_MAP` 文件级路由
  （adding_new_model.md:23-64）
- pipeline 用指针接口；hot path 禁止 host sync、禁止动态分配
- legacy 名字是 ABI：替换实现必须保持 shape/数值契约
  （CONTRIBUTING.md:122-142，#30/#40 事故）

## 2. ABI 纪律（Jetson-PI 案例的核心教训）

- `frt_model_runtime_v1` 只增不改（additive only）
- 一个 runtime 只发布一个 execution authority——禁止并行 v2
- 新功能/provider 对 `runtime/`、`exec/`、`csrc/` 共享目录 zero diff
- 自定义 stage 图走 v1 generic OPAQUE plan，不另立接口

## 3. PR 规范（提交时）

- 单一目的、additive、opt-in、默认关闭、不动默认路径；小 PR 优先
- 先开 issue 对齐设计（#143 是样板：tracking thread 可以长期挂着）
- kernel/CMake/binding 类 = R3 风险级：需全路径回归证据
- 精度类变更必须附：cosine/参考对比 + 延迟前后数据
- 性能声明必须含：确切命令、硬件、迭代数，且配正确性证据
- PR 描述不带 `Co-authored-by:`

## 4. 证据等级纪律（作者的评审语言）

- wrapper parity ≠ 参考策略 parity（两种证据分开报）
- 定性测量 ≠ 性能声明（"qualification measurements, not production
  performance claims"）
- 能力声明 fail-closed：做不到的不 advertise（假分阶段教训：
  `context()` 跑全程 264ms + `action()` 拷缓存 0.15ms 是被作者实测戳穿的）
- 作者会拿真实 checkpoint 亲自复跑你的 PR——不留语义水分

## 5. AI 痕迹红线

- "AI-generated traces or comments" 是 PR blocker（pr_review_checklist:581-607）
- 提交前自查：注释风格改写为仓库惯例、删除模板化 docstring、
  去掉过度防御性注释、commit message 用人话
- 私有路径、密钥、本机信息不进 diff

## 6. 我们的 PR 切分计划

1. `tools/`：Omega-QVLA pack → E0M3+UE4M3 布局转换器 + 格式文档
   （纯 additive，最易收；参照作者 GGUF Q4_0/Q4_K 映射笔记的形态）
2. per-step scale 消费路径（flag-gated `weight_format` 分支，
   参照 `pipeline_thor.py` 现有三路路由）
3. SVDQuant 低秩分支 epilogue 融合
每个 PR 附：LIBERO 逐任务成功率配对 + action cosine + p50/p95 延迟对照。

## 7. 行为备忘

- 作者欢迎功能但会按契约重写实现，且保留贡献者 authorship——
  被重写不丢人，初版就把契约形态做对他更省事
- 礼貌但实质：他用提问代替命令，我们用数据代替辩解
