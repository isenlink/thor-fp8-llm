# 在 DRIVE Thor 上用 llama.cpp 启用 Qwen3.5-MoE 的 MTP 投机解码

> One-liner: A Qwen3.6-35B-A3B GGUF carries a built-in MTP (multi-token
> prediction) head, but `llama-server` **silently ignores it** unless you pass
> `--spec-type draft-mtp`. Enabling it on a DRIVE Thor board took decode from
> **~39.6 t/s to ~58.6 t/s (+48%)** with no quality change.
>
> 适用：DRIVE Thor（p3960-0010 / Tegra264），DriveOS 7.0.x，llama.cpp 0.4.0-dev

---

## 一、问题：模型里有 MTP，但推理时被丢弃

部署 Qwen3.6-35B-A3B（MoE，Q4_K_M）到 DRIVE Thor 后，加载日志里出现一串警告：

```
W model has unused tensor blk.40.nextn.eh_proj.weight            (size = 8912896 bytes) -- ignoring
W model has unused tensor blk.40.nextn.enorm.weight              (size = 8192 bytes)    -- ignoring
W model has unused tensor blk.40.nextn.hnorm.weight              (size = 8192 bytes)    -- ignoring
W model has unused tensor blk.40.nextn.shared_head_norm.weight   (size = 8192 bytes)    -- ignoring
```

同时模型元数据里明明声明了 MTP 层：

```
qwen35moe.nextn_predict_layers      ← 元数据声明有 MTP
```

**这组张量就是 MTP 头**（Multi-Token Prediction）。`unused ... ignoring` 表示推理时**它们被直接跳过了**——模型自带的投机解码能力完全没用上。

## 二、根因：不传参数就不加载 MTP 权重

查 llama.cpp 源码，Qwen3.5-MoE 的 MTP 支持**是完整实现的**：

```cpp
// src/models/qwen35.cpp:488-489
GGML_ASSERT(hparams.n_layer_nextn > 0 && "QWEN35 MTP requires n_layer_nextn > 0");
GGML_ASSERT(hparams.n_layer_nextn == 1 && "QWEN35 MTP currently only supports a single MTP block");

// src/models/qwen35.cpp:110-115  MTP 张量定义
layer.nextn.eh_proj          = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ, "weight", il), { 2 * n_embd, n_embd }, mtp_flags);
layer.nextn.enorm            = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM, "weight", il), { n_embd }, mtp_flags);
layer.nextn.hnorm            = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM, "weight", il), { n_embd }, mtp_flags);
// ...519-549 完整的 MTP 图构建（embed_tokens + hnorm + enorm + eh_proj 拼接）
```

关键在 `common/common.cpp`：

```cpp
// common/common.cpp:1714
mparams.load_mtp = std::find(params.speculative.types.begin(),
                             params.speculative.types.end(),
                             COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params.speculative.types.end();
```

**`load_mtp` 只在投机解码类型里包含 `draft-mtp` 时才为 true。** 不传 `--spec-type draft-mtp` → 不加载 MTP 权重 → 那些张量被当无用张量忽略。

**所以修复只需要加一个参数。**

## 三、启用方法

```bash
llama-server \
  -m /brand_data/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  -ngl 99 -fa on \
  --host 0.0.0.0 --port 8080 \
  -c 16384 -t 8 \
  --reasoning-effort low \
  --spec-type draft-mtp \      # ★ 就是这一行
  --temperature 0.2
```

启动日志确认 MTP 已挂载：

```
I common_speculative_init_result: creating MTP draft context against the target model '/brand_data/models/....gguf'
I srv  llama_server: model loaded
I srv  llama_server: listening on http://0.0.0.0:8080
```

对比：不传 `--spec-type draft-mtp` 时**没有** `creating MTP draft context` 这行，而是前面那些 `unused tensor ... ignoring`。

## 四、实测效果

同一提示词、同一服务，仅切换 MTP 开关：

| 指标 | 无 MTP | 有 MTP | 变化 |
|---|---|---|---|
| 完成 tokens | 1232 | 1216 | ~持平 |
| 墙钟耗时 | 31.5 s | **20.8 s** | **-34%** |
| **吞吐** | 39.6 t/s | **58.6 t/s** | **+48%** |

**输出质量无变化**——同一道编程题（回文判断函数）两次都给出一致的标准解：

```python
def is_palindrome(s: str) -> bool:
    cleaned = ''.join(c.lower() for c in s if c.isalnum())
    return cleaned == cleaned[::-1]
```

低温度配置（`--temperature 0.2 --top-k 20 --top-p 0.9`）下的补充观测：

| 任务 | tokens | 耗时 | 吞吐 |
|---|---|---|---|
| 算术 `17*23+45`（thinking 模型） | 592 | 9.1 s | ~65 t/s |
| 同题重跑 | 542 | 8.2 s | ~66 t/s |

两次均输出正确答案 `436`，一致性符合低温度预期。

## 五、相关参数说明

llama.cpp 里和"思考/投机"相关的参数（`llama-server --help` 实测）：

| 参数 | 作用 |
|---|---|
| `--spec-type draft-mtp` | ★ 启用 MTP 投机解码（同时触发 MTP 权重加载） |
| `--spec-draft-n-max N` | 每步投机草稿 token 数（默认 3） |
| `--spec-draft-p-min P` | 投机接受最小概率 |
| `--reasoning-effort LEVEL` | 思考强度：`minimal`/`low`/`medium`/`high`/`xhigh`/`max` |
| `--reasoning-budget N` | 思考 token 硬上限（-1 不限，0 立即结束） |
| `--reasoning-format FORMAT` | 思考内容放置：`deepseek`（放进 `reasoning_content`）等 |

### 关于 thinking 模型的注意事项

Qwen3.6-35B-A3B 是 **thinking 模型**，思考链会消耗 token 预算。若 `max_tokens` 给小了（如 300），会出现"思考完但正文为空"：

```
回答: (空)
思考: 1. **分析用户需求**： ... 17 * 23 +     ← 思考被截断，正文没机会输出
tokens: prompt=30 completion=300
```

规避：`max_tokens` 至少给 1000，或加 `--reasoning-budget` 限制思考长度。

⚠️ 实测 `--reasoning-effort low` 对 Qwen 模板的**约束力有限**——思考链仍有 1300-1500 字符。要真正压短思考，`--reasoning-budget 256`（硬限）比 effort 更有效。

## 六、适用性判断（怎么知道某个模型能不能开 MTP）

三步确认：

```bash
# 1. 模型元数据里有没有 nextn_predict_layers 之类的声明
llama-gguf model.gguf r 2>&1 | grep -iE 'nextn|mtp'

# 2. 有没有 nextn 张量（MTP 头的实体）
llama-gguf model.gguf r 2>&1 | grep -i 'nextn' | head

# 3. 推理引擎的模型实现里有没有 MTP 图构建
grep -rn "n_layer_nextn\|LLM_TENSOR_NEXTN" src/models/<your-model>.cpp
```

三个都有 → 可以开 `--spec-type draft-mtp`。
只有 1、2 没有 3 → 模型带 MTP 头但引擎还没接（等上游更新）。

**本次案例三个都满足**，所以启用即生效。

## 七、性能背景：为什么投机解码在这个平台特别重要

DRIVE Thor 的内存带宽约 **273 GB/s**，LLM decode 阶段是带宽受限的——每生成一个 token 都要把激活的权重读一遍。MoE 模型虽然每 token 只激活少量参数（Qwen3.6-35B-A3B 约 3B active），但 decode 的串行性仍然使其受限于内存延迟。

MTP 的价值在于：**一次前向生成多个 token**，把内存带宽的利用效率提上去。实测 +48% 符合投机解码的典型收益区间。

## 八、一页速查

```bash
# 启用 MTP（关键就这一行）
--spec-type draft-mtp

# 确认生效（启动日志）
grep -i 'MTP draft context' server.log        # 有 → 生效
grep -i 'unused tensor.*nextn' server.log     # 有 → 没生效，检查参数

# 检查模型是否带 MTP 头
llama-gguf model.gguf r 2>&1 | grep -i nextn

# 配合思考控制
--reasoning-effort low            # 软约束（效果有限）
--reasoning-budget 256            # 硬上限（更有效）
```

## 九、环境与版本

| 项 | 值 |
|---|---|
| 平台 | DRIVE Thor（p3960-0010，Tegra264） |
| 系统 | DriveOS 7.0.x |
| 计算能力 | sm_101（实测 `cudaDeviceGetAttribute`，非 sm_110） |
| 引擎 | llama.cpp 0.4.0-dev（ARM64，CUDA，交叉编译产出） |
| 模型 | Qwen3.6-35B-A3B-UD-Q4_K_M.gguf（35.5B 参数，Q4_K_M） |
| 上下文 | 16384（模型原生支持 262144） |
| GPU 内存池 | 46 GiB（`Device 0: Thor, compute capability 10.1, VMM: yes, VRAM: 47104 MiB`） |

---

## [整理者注] 已移除/脱敏内容清单

本文档由内部工作笔记脱敏改写：
- 板载数据分区路径统一改 `/brand_data/` 代称（DriveOS 板上该分区约 105G、与只读根分区独立，原路径名含车辆品牌字样；读者在自己板卡上执行 `ls /` 即可看到真实分区名）
- 设备 IP、序列号、账号名、主机代号、协作设施名称已移除
- 技术数据（性能数字、参数、源码行号、命令、模型规格）100% 保留
