# 13-阶段成果总结（2026-09-13）

> 作者：AI助手（x86主机） · 范围：Thor 板 Qwen3.8-27B 推理优化，从 9-08 到 9-13
> 前置文档：11-完整工作交接文档、A1-DECISION-2026-09-11、HANDOFF-2026-09-12-B3、
> b3-y-reuse-kernel-design.md、b3-takeover-correctness-2026-09-12.md

## 1. 最终成绩

模型 RadixArk-F8attn-v2.gguf + 修复后 NVFP4 MMVQ kernel（perj）+ MTP K12 p-min 0.5：

| 口径（decode，确定性配对 prompt） | 基线 | 最终 | 提升 |
|---|---|---|---|
| 2K（输入 1957 tok / 输出 384） | 25.72 t/s | **31.02 t/s** | **+20.6%** |
| 128K（输入 123925 tok / 输出 384） | 16.88 t/s | **19.61 t/s** | **+16.2%** |
| 128K prefill | 174.34 t/s | 174.38 t/s | 持平（符合预期） |

- 正确性：同配置链（baseline/gated/perj 三代 kernel）输出逐字一致；K12 输出因 verify batch 形状变化在 near-tie 处良性换述，文本连贯、任务行为一致
- **正式目标：KV ≥128K 口径 decode ≥30 t/s（用户 9-13 确认；网友同模型族 200K 已实测 31 t/s 作可行性锚点）。当前 128K = 19.61 t/s，差距 1.53×，理论上限约 45 t/s（带宽摊薄估算）；2K 的 31.02 t/s 是过程里程碑，不计入达标**
- 对比项目起点（9-08 老栈 QUASAR 口径）：API decode 12.5 t/s 级 → 当前同模型族新栈 31 t/s @2K

## 2. 思路演进（为什么走到这里）

### 2.1 起点与路线选择（9-08 ~ 9-10）
- 起点：老 build（官方镜像 b10373）+ QUASAR-NVFP4 + 外挂 MTP K7，200K 口径 12.32 t/s
- 社区配方逐一证伪：GGML_MMVQ_MAX=2（对 NVFP4 无效）、DFlash2（量化目标模型上 acceptance 崩到 12-32%）、K12（旧栈 200K 反转劣于 K7）——教训：**配方迁移必须核对"模型+栈"双重前提**
- 转向 RadixArk 模型（NVFP4 MLP + FP8 注意力的 ModelOpt 混合量化），自写转换器出 GGUF；F8attn-v2 定版（F8 attention 投影走自写 cuBLASLt shim，实测 246 GB/s ≈ 90% 带宽）

### 2.2 瓶颈定位方法论（9-11，A1 决策链）
关键不是猜，是分叉实验：
1. **R26 GPU 利用率分叉**：tegrastats 采样证明 decode 段 GPU ~94% 满载 → GPU-bound，排除 launch/graph 覆盖假设
2. **R27/R28/R30/R32 op 账本**：op_prof 去串行化 + 滚动窗口 + per-call 带宽核算，实锤**唯一低效路径 = NVFP4 MMVQ gemv（95 GB/s，同场景 F8 为 246 GB/s，Q8_0 上限 240）**
3. 排除法全部有实验依据：36B 非对齐不是瓶颈、寄存器压力不是瓶颈、缓存热度不是瓶颈

### 2.3 B3 y 复用 kernel（9-12）
- microbench3/4/5 流水线拆解：y（激活侧 Q8_1）在每个权重行循环内重复读取，R=8 时占 53% L2 流量；提到行循环外 + rows_per_block=8 → standalone 108→202 GB/s（+88%）
- 实现：mmvq.cu 三处改动 + vecdotq.cuh 新增 preload 变体；standalone 验证 ffn_gate +44%、lm_head +50%

### 2.4 正确性事故与修复链（9-12 晚 ~ 9-13，本轮核心）
1. **buggy 版**：preload 分支条件只查 `NVFP4 && rows>1`，M=2..8（MTP verify）误入；且 preload 指针缺 `j*stride_col_y` → 所有输出列错用第 0 列输入。M=1 通顺、MTP 下放大成重复循环
2. **gated 版**（9-12 22:35）：两处条件补 `ncols_dst == 1`，M=1 保留优化，多列回退原路径。全部验证通过
3. **perj 版**（9-12 23:43 构建，9-13 验证定案）：preload 移入列循环内、指针补 `j*stride_col_y` —— 根因修复，verify 路径（M=2..8）同样获得 y 复用。standalone 逐位一致 + CPU 参考全 PASS，生产配对 2K/128K 输出与基线逐字相同
4. perj 严格优于 gated（2K +1.8%、128K +2.3%）且数学上是根修 → 定为生产 binary

### 2.5 MTP 参数重扫（9-13，零编译）
kernel 变快改变了 verify/draft 成本结构 → 旧栈"K7 最优"结论失效，必须重扫：

| 配置 | 2K decode | 判定 |
|---|---|---|
| K7 p0.6（旧最优） | 28.38 | 基线 |
| K12 p0.6 | 30.41 | 突破 30 |
| **K12 p0.5** | **31.02** | 最优 |
| K12 p0.4 | 30.36 | 过深略亏 |
| K16 p0.5 | 27.21 | 过深反噬 |
| MMVQ_MAX=8 | 28.32 | 与 =2 打平，维持 =2 |

128K 确认：K12 p0.5 = 19.61 t/s（vs K7 18.19，+7.8%；mean len 4.39→5.94）。

## 3. 操作方式（复现手册）

### 3.1 kernel 构建（x86主机，免 qemu 快通道）
```bash
# mmvq.cu 单文件交叉编译（约 1-2 分钟；全量 cmake 约 37 分钟仅首次需要）
cd ~/work/thor-driveos && make -f <repo>/scripts/native-mmvq.mk
# 产物 mmvq.cu.o → 链接：
bash <repo>/scripts/link-server.sh   # 出 llama-server（手改 -o 目标名）
bash <repo>/scripts/link-check.sh    # 出 b3-correctness（batched vs 单列）
bash <repo>/scripts/link-check2.sh   # 出 b3-correctness2（CPU 反量化参考）
```

### 3.1.1 kernel 补丁（本仓库内）
- `patches/mmvq-nvfp4-y-reuse-b3.diff` —— mmvq.cu + vecdotq.cuh 的完整改动（y 预读复用 / rows_per_block=8 / preload 列内修正），`git apply` 到 llama.cpp 对应版本即可
- `scripts/native-mmvq.mk`、`scripts/link-*.sh` —— 单文件交叉编译与链接脚本（路径按己方环境改）

### 3.2 正确性门禁（ kernel 改动必过，不过不进全模型）
```bash
# 传板后：
./b3-correctness      # 要求 column_failures=0（M=1/2/3/4/8 每列 batched==single 逐位一致）
./b3-correctness2     # 要求 cpu_failures=0（k=128/5120 × M=1..8 对 CPU 反量化点积）
# 全模型门禁：固定请求 temperature=0，输出必须与基线逐字一致（batch 形状不变时）
# MTP 门禁：acceptance 不得塌（A2 教训：能出字≠正确，acc <65% 判失败）
```

### 3.3 生产启动（板上，脚本文件承载，防 pkill 连坐 ssh）
```bash
# /brand_data/ai_workspace/ai-assistant/restart-pmin.sh <K> <pmin> <logtag>
bash restart-pmin.sh 12 0.5 prod
# 等价手动命令：
GGML_CUDA_GRAPH_OPT=1 GGML_MMVQ_MAX=2 setsid ./llama-server-ai-assistant-perj \
  -m /brand_data/ai_workspace/models/RadixArk-F8attn-v2.gguf -ngl 99 -c 131072 -fa on \
  --cache-type-k f16 --cache-type-v f16 --parallel 1 --port 8080 \
  --spec-type draft-mtp --spec-draft-n-max 12 --spec-draft-p-min 0.5 \
  > prod.log 2>&1 < /dev/null &
```

### 3.4 确定性配对 bench（`scripts/bench.py`，板上同名）
```bash
python3 bench.py 2000 trial1     # 2K 口径，约 25 秒
python3 bench.py 124000 trial1   # 128K 口径，约 12-13 分钟
```
- prompt 由 /tokenize 精确计数构造（永不猜密度）；同一 trial 名 → 逐字节相同 prompt
- temperature=0、seed=123、n_predict=384、ignore_eos、cache_prompt=false
- 判定：同 trial 跨 binary 输出逐字一致 + timings.predicted_per_second 对比
- prefill 对比必须用同进度点的瞬时速率（早期 KV 小的暖区读数不可比）

### 3.5 运维纪律（踩坑沉淀）
- llama-server 有两个进程，TERM 后必须 ps 确认全灭再等 HugePages_Free 回到 23552
- pkill -f 的模式串会匹配 ssh 会话自身命令行 → 连坐自杀；杀进程逻辑必须放板上脚本文件
- 板上启动 server 必须 setsid + 重定向三件套，否则 ssh 断开带走子进程
- 温度红线 95°C；kernel 崩溃可能拖死 nvgpu 通道（只能重启板），microbench 先查代码

> 模型文件 19.57 GiB 不入库，SHA256 与获取/转换步骤见 [../03-model-conversion/README.md](../03-model-conversion/README.md)。

## 4. 失败/关闭路线台账（勿重复投入）

| 路线 | 结果 | 证据 |
|---|---|---|
| GGML_MMVQ_MAX=2 对老 QUASAR 栈 | 无效/-12% | 06-实验1 |
| DFlash2 块扩散起草 | 两个目标模型全慢，acc 崩 | 06-实验6/7 |
| q8_0 KV @长上下文 | 双重惩罚更慢 | 11-R3/R4 |
| A2：MLP NVFP4→F8 重编码 | decode -61%，acc 0.68% 全灭 | A1 §9.12 |
| K12 @旧栈 200K | 反转劣于 K7 | 11-R2 |
| 社区 #28514 TMA/FP4-MMA patch | Thor sm_101a 无 tcgen05，硬件不存在 | A1 §9.14 B1 |
| L2 预取扩展到 101a（R35） | kernel 中性，t/s 差异是 acc 噪声 | A1 §9.14 B2a |
| MMVQ_MAX=8（新 kernel 下复扫） | 与 =2 打平 | 本文件 §2.5 |
| K16 p0.5 | 过深反噬 -12% | 本文件 §2.5 |

## 5. 当前生产配置

- binary：板上 `/brand_data/ai_workspace/ai-assistant/llama-server-ai-assistant-perj`（SHA256 3dafb508…，源 = xbuild/llama-cpp-latest 当前工作区 mmvq.cu 的 perj 修复）
- 模型：RadixArk-F8attn-v2.gguf（SHA256 d15dac91…，板上与 x86主机 一致）
- 配置：GGML_CUDA_GRAPH_OPT=1、GGML_MMVQ_MAX=2、f16 KV、-fa on、-c 131072、MTP K12 p-min 0.5
- 证据：~/work/thor-ai-assistant/evidence/（全部配对 jsonl + server log）

## 6. 遗留与下一步

1. **128K → 30 t/s**：当前 19.61，缺口在 kernel 效率（NVFP4 LUT 算术化解码 ~17% 空间、draft 链开销、小算子 ~17%）与 KV 字节（物理边界，需架构级改动如 MLA；q8_0 KV 已证伪）
2. 多轮配对提升置信度（当前各口径单 trial）
3. perj 修复可向上游提 PR（NVFP4 MMVQ preload 路径）
4. DRIVE OS 7.2.5（CUDA 13.3）隔离评估——独立立项，见 12-研究报告
