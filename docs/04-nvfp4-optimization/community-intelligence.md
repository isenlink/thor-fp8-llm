# Thor NVFP4 优化情报汇总（停机期间研究）

> One-liner: Community intelligence digest for optimizing NVFP4 speculative
> decoding on bandwidth-constrained hardware — the research that led to our
> final config (fa on, n-max 12, p-min 0.6, parallel 1 → 26.48 tok/s).
>
> 背景：Thor 车机因水冷渗漏停机检修，利用停机窗口完成的社区经验调研。
> 目标：decode 25.89 tok/s（社区 QUASAR+MTP K7+Graph 水平）
> 调研时状态：QUASAR QAT-NVFP4 + MTP K3 + q8_0 KV @64K = 11.42 tok/s

---

## 一、核心发现：三条最有价值的经验来源

### 1. sudoingX/qwen38-mtp（社区实测宝典，40人53配置）
https://github.com/sudoingX/qwen38-mtp

**七条黄金规则（摘要）**：
1. n-max 甜点因卡而异：24GB卡=2，更强卡=3-4；换配置后要重扫
2. **`--spec-draft-p-min` 0.60-0.75 对带宽受限机型帮助大**（门控让深起草几乎免费）；快速卡上反而负收益
3. 收益随生成长度增长；400 token 以上才完全体现
4. 多卡先修 split-mode（tensor 比 layer 快）
5. **必须 --parallel 1 测量**，并行 4 以上投机优势消失
6. 先重编译最新 llama.cpp（上游每周都在优化 qwen3_5 路径，新构建裸速 +10-15%）
7. 共享桌面会悄悄吃掉一半显存带宽——确认权重真正常驻

**带宽受限机型（与我们最像）参考数据**：

| 机型 | 基线 | 调优后 | n-max | acceptance |
|---|---|---|---|---|
| Ryzen AI Max+ 395 (APU) | 11.5 | **23.7** | 12 | 0.95-1.0 |
| GMK EVO-X2 (Strix Halo) | 10.5 | **21.4-22.2** | 12 | 0.95-1.0 |
| RTX 3090 | 31.0 | 41.3 | 2 | 0.78 |
| RTX 4090 | 47.7 | 76.3 | 2 | 0.56 |

**APU 类（带宽受限）= n-max 大(12) + p-min 高门控 = 翻倍** ← 我们的钥匙

### 2. hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4（L4 24GB 深度剖析）
https://github.com/hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4

**L4 24GB 达成 37.4 tok/s 的配方**（降序价值）：

| 要素 | 累计 tok/s | 价值 |
|---|---|---|
| 原生 MTP 头 draft depth 2 + p-min 0.4 | 24.14 | 基础 |
| **MMQ kernel 路由**（`GGML_MMVQ_MAX=2`） | 28.05 | **+16.2%** |
| DFlash 2 独立 drafter（块扩散起草） | 35.99 | +15% |
| 2026-08-19 重建版权重（UD-Q4_K_XL v3） | 37.86 | +5.2% |
| GPU 端采样（`-bs`，免 248K logits 拷回主机） | 31.0 | +0.9% |

**关键机制洞察**（对 Thor 直接适用）：
- decode 是内存带宽瓶颈；带宽利用率 llama.cpp 只到 84%（252.8/300 GB/s）
- 缺的 16% 是**反量化的功耗税**：dequant 烧功耗 → 功耗墙压 SM 频率 → 总线填不满
- **这 16% 只能靠"少做 ALU 功"找回 = 换 kernel 家族，不是调参**
- 我们的 tcgen05 kernel 优化正好对应这个方向！

**成本模型**（可以直接套用到 Thor 分析）：
```
tok/s = mean_accepted_length / (verify_pass_cost + drafting_cost)
Thor: ~67ms 级别的 pass 成本需要实测，套用同款带宽探针
```

### 3. vLLM 官方 recipe（QUASAR 同款模型）
- QAT 版 MTP acceptance 0.897，unsloth 后量化版 0.788
- **我们的 acceptance 0.4-0.94 波动大**——QUASAR QAT 头质量应更好
- RTX 5090 上 NVFP4 需要 --enforce-eager（CUDA graph 显存挤不下）—— Thor 无此问题

---

## 二、与我们配置的直接对照

| 配置项 | 我们当前 | 社区最优（带宽受限类） | 动作 |
|---|---|---|---|
| flash attention | auto（未确认） | on | 显式 `-fa on` |
| n-max | 3（K7 试过，acceptance 18% 反而慢） | APU类=12 + p-min 门控 | **试 12 + p-min 0.6** |
| p-min | 未设 | 0.60-0.75（带宽受限类） | **设 0.6** |
| parallel | 未显式设 | 必须 1 | 显式 `--parallel 1` |
| KV | q8_0 @64K | q4_0（226K） | 保留 q8_0（64K 够用） |
| llama.cpp 版本 | 2026-09-02 | 持续更新 | 考虑更新（+10-15% 裸速） |
| kernel 路由 | 默认 MMVQ | GGML_MMVQ_MAX=2 全走 MMQ | **+16.2% 可试** |

**预期路径**：
```
当前 11.42
→ fa on + parallel 1 + n-max/p-min 调优:  期望 15-18
→ MMQ kernel 路由补丁:                    期望 18-21
→ 最新 llama.cpp 重编译:                  期望 20-23
→ tcgen05 kernel（自研主线）:             冲击 25.89+
```

---

## 三、其他要点

1. **DFlash 2 有了 llama.cpp 支持路径**：Jetson AI Lab 教程显示 `--spec-type draft-dflash` + `-hfd` 独立 drafter（n-max 15）——之前搁置的 DFlash2 头可能可以直接用！需要 GGUF 格式的 draft 头，Qwen3.8 的 DFlash GGUF 需要找/转
2. **量化对照**：4-bit Q4_K_M 质量损失可忽略（社区评测），1-bit 崩——我们 NVFP4 无虞
3. **Thor 专属数据点**：视频实测 Thor 上 Qwen3-coder prompt 880 tok/s + decode 49 tok/s（TensorRT-LLM？模型较小）；MLC issue 确认 Thor 上 llama.cpp decode 比 MLC 慢 2-3 倍——**说明 Thor 的 llama.cpp 路径还有很大 kernel 优化空间**
4. **Upstream 动向**：llama.cpp 的 qwen3_5 hybrid-attention kernel 还很年轻（"day-one speeds"），上游持续优化中——定期 rebase 值得做

---

## 四、恢复服务后的实验队列（按性价比排序）

1. 启动脚本改造版：+`-fa on --spec-draft-p-min 0.6 --parallel 1`（5 分钟出结果）
2. 若 n-max 12 + p-min 好 → 细扫 p-min {0.5,0.6,0.65,0.7} × n-max {8,10,12}
3. GGML_MMVQ_MAX=2 环境变量（零改动，直接测）
4. 拉 llama.cpp 最新 master 重编译，裸速对比
5. 深挖：tcgen05 prefill kernel 集成

---

> **实验结果见** [nvfp4-optimization-log.md](./nvfp4-optimization-log.md)：
> 实验队列第 1 项一次命中，decode 26.48 tok/s，超 25.89 目标 ✅
