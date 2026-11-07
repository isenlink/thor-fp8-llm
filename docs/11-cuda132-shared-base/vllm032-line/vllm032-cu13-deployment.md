# vLLM 0.32 · CUDA 13 部署实录（thor05 生产线）

> 定位：本文是 [CUDA 13.2 共用底座](README.md) 之上的 **vLLM 0.32 部署线**，补齐
> [09-vllm032-sm101](../09-vllm032-sm101/README.md)（CUDA 12.8 资格验证线）之后的
> 生产化环节——栈落盘布局、启动脚本、JIT 编译纪律、断电重启 runbook、验收基准。
> **2026-11-07 增量**：三臂投机（AR/MTP/DFlash2）+ 文字视觉全模态 + 256K 验收
> （现役形态见文末「11-07 增量」节）。
> 全部命令与数字来自 2026-10-24/25 在一块 DRIVE Thor（p3960 / Tegra264，sm_101，
> sm_101，14 SM，统一内存 ~58G）上的实机操作与验收，部署链**只依赖板上持久分区，不依赖
> 任何外置存储**（无 USB SSD 也能完整复现）。本文中 `sda1` 仅以**反例**出现
> （A3 自检与核对清单里「必须清零的外置盘引用」），部署链本身不含。
> CUDA 13 forward-compat 方法致谢 [souyou233 (EnkiMa)](https://github.com/souyou233)。

## 为什么需要这篇

- docs/09 解决了「vLLM 0.32 在 sm_101 上能不能跑、多快」（CUDA 12.8 资格线），
  但该线明确「未作生产升级」；
- docs/11 提供了 CUDA 13.2 共用底座（compat 驱动 + runtime），但只给了 llama.cpp
  线的部署脚本；
- 本文补中间层：**vLLM 0.32 + torch cu130 在共用底座上的完整部署与运维**，
  含三次真实事故（JIT 并发编译 OOM、符号链接迁移引发全量重编、cache 换家哈希失效）
  的机理与规避方法。

## 软件栈版本（venv 实测 `pip list`）

| 包 | 版本 |
|---|---|
| vllm | 0.32.0 |
| torch | 2.13.0+cu130 |
| triton | 3.7.1 |
| transformers | 5.19.0 |
| flashinfer | 0.7.0.post1（JIT 缓存目录可见） |
| safetensors | 0.8.0 |

GPU 侧只有 CUDA 12 时代的内核驱动——cu130 栈靠 docs/11 的 forward-compat
`libcuda.so.1`（放 `LD_LIBRARY_PATH` 最前）驱动，系统 `libnvrm_*` 保留不动。

## 落盘布局（全部在持久分区）

```
/zeekr_data/                          ← 板上持久分区（断电不丢）
├── cu13-vllm-20261024/               ← 栈家目录
│   ├── venvs/vllm032-cu13-clean/     ← venv（~15G，含 torch cu130 全家）
│   ├── toolchain/                    ← 交叉编译期搬上来的 nvcc/gcc 工具链
│   ├── compat_orin/                  ← CUDA13 forward-compat libcuda.so.1.1 等
│   └── cache/vllm-cu130-clean/       ← 全部 JIT/编译缓存（triton/flashinfer/cuda…）
└── ai_workspace/cu13-vllm-20261022/  ← 工作区：launch 脚本 + logs + pid
└──（模型权重在另一持久分区，如 /zeekr_hdmap/<model>）

/home、/tmp、/etc                      ← overlay，断电即丢——部署链严禁依赖
```

**为什么 cache 必须在持久分区**：flashinfer/triton 的 JIT 产物（`cached_ops/`
下的 `.so`）是按「环境指纹（sys.prefix、路径哈希、编译参数）」索引的。放在
tmp 类目录意味着每次断电重编，首次重编还会撞上「并发编译打爆宿主」事故（见下）。

## 部署步骤（每步带判据）

### A0 · 前置：大页池

KV cache 走 2MiB 大页。10-24/25 基线为 23552 页（=46G）；**11-07 起 256K 验收口径
升级为 25600 页（=50G）**——池即 CUDA 眼里的全部「显存」，256K KV 预算按 50G 算。
设置与持久化方法见 [hugepage-pool](../05-system-tuning/hugepage-pool.md)。

```bash
grep HugePages_Total /proc/meminfo   # 判据：= 25600（11-07 起；重启后为 0 就按上文重设）
```

### A1 · 栈落盘

把 venv + toolchain + compat_orin + cache 四件解包到 `/zeekr_data/cu13-vllm-20261024/`
（本文不随仓分发大二进制；构件清单与 sha256 见 ModelScope 仓
`navyyang/thor05-vllm032-sm101-deploy` 与 `navyyang/thor-cuda132-shared`）。

判据：`du -sh` venv ≈ 15G；`ls compat_orin/libcuda.so.1.1` 存在。

### A2 · 两个开发符号链接（缺了 JIT 链接失败）

venv 内的 CUDA 目录需要 lib64 与 libcudart 开发名：

```bash
CU13=/zeekr_data/cu13-vllm-20261024/venvs/vllm032-cu13-clean/lib/python3.12/site-packages/nvidia/cu13
ln -sfn lib "$CU13/lib64"
ln -sf libcudart.so.13 "$CU13/lib/libcudart.so"   # 按实际 .so.13.x 后缀调整
```

判据：`ls $CU13/lib64`、`ls $CU13/lib/libcudart.so` 都能解析。

### A3 · venv 可移植性自检（换机部署的头号坑）

venv 里的启动包装脚本（`bin/vllm`、`bin/python` 等）与 toolchain 的 `bin/cc`、
`bin/c++` 是**内嵌绝对路径的文本脚本**。任何「打包 → 换路径解包」之后都必须全量
核对，否则出现 `ninja: cannot find cc` 或 flashinfer 反复重编：

```bash
grep -rl '/media/sda1\|/old/path' /zeekr_data/cu13-vllm-20261024/venvs/vllm032-cu13-clean/bin/ | wc -l   # 判据：0
grep -rl '/media/sda1\|/old/path' /zeekr_data/cu13-vllm-20261024/toolchain/bin/cc /zeekr_data/cu13-vllm-20261024/toolchain/bin/c++ | wc -l  # 判据：0
```

**机理教训（10/24 实证）**：flashinfer 的 `build()` 每次启动按当前环境重写
`build.ninja` 并与 `.ninja_log` 比对哈希——**venv 路径一变哈希必失效必重编**。
软链骗不过 Python 的 `sys.prefix`（`realpath` 会解析出真实路径），所以
「sda1 软链回板上」这类做法无效，**实体必须在最终路径上**。cache 目录同理：
本次（10/25）把 cache 从外置盘挪回板上，指纹变化触发 fp8/fp4 两个内核全量重编
（约 30–60 分钟，`MAX_JOBS=1` 下无 OOM），重编一次后 `.ninja_log` 自洽、之后
启动回到 6–10 分钟。

### A4 · 启动脚本（生产在用原文，单文件即部署）

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT=/zeekr_data/ai_workspace/cu13-vllm-20261022
VENV=/zeekr_data/cu13-vllm-20261024/venvs/vllm032-cu13-clean
COMPAT=/zeekr_data/cu13-vllm-20261024/compat_orin
TOOLCHAIN=/zeekr_data/cu13-vllm-20261024/toolchain
MODEL=/zeekr_hdmap/RadixArk-Qwen3.8-27B-NVFP4
LOG=$ROOT/logs/server-cu130-clean.log
PID=$ROOT/server-cu130-clean.pid
CACHE=/zeekr_data/cu13-vllm-20261024/cache/vllm-cu130-clean
CUDA13=$VENV/lib/python3.12/site-packages/nvidia/cu13

mkdir -p "$ROOT/logs" "$CACHE"/{triton,torchinductor,cuda,vllm,hf,xdg,home,flashinfer}
echo $$ > "$PID"
exec > >(tee -a "$LOG") 2>&1

unset PYTHONPATH
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$COMPAT"          # ← forward-compat libcuda 必须最前
export CUDA_HOME="$CUDA13"
export CC="$TOOLCHAIN/bin/cc"
export CXX="$TOOLCHAIN/bin/c++"
export PATH="$VENV/bin:$TOOLCHAIN/usr/bin:$CUDA13/bin:$PATH"
export VLLM_GDN_DECODE_KERNEL=cuda
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_HUB_OFFLINE=1
export MAX_JOBS=1                          # ← JIT 并发纪律，见下
export TRITON_CACHE_DIR="$CACHE/triton"
export TRITON_HOME="$CACHE/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE/torchinductor"
export CUDA_CACHE_PATH="$CACHE/cuda"
export VLLM_CACHE_ROOT="$CACHE/vllm"
export HF_HOME="$CACHE/hf"
export XDG_CACHE_HOME="$CACHE/xdg"
export HOME="$CACHE/home"                  # ← flashinfer 缓存挂在 HOME 下
export FLASHINFER_WORKSPACE_DIR="$CACHE/flashinfer"

echo "START $(date -Is) pid=$$"

exec "$VENV/bin/vllm" serve "$MODEL" \
  --served-model-name qwen3.8-27b \
  --safetensors-load-strategy lazy \
  --host 0.0.0.0 \
  --port 8081 \
  --max-model-len 262144 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 2 \
  --gpu-memory-utilization 0.90 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4}' \
  --override-generation-config '{"temperature":0.3}' \
  --default-chat-template-kwargs '{"enable_thinking":true,"reasoning_effort":"medium"}' \
  --reasoning-parser deepseek_r1 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --enable-prefix-caching \
  --kernel-config '{"enable_jit_warmup":false}' \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,5,10]}' \
  --attention-backend TRITON_ATTN \
  --language-model-only
```

拉起方式（SSH 会话断连不杀进程的唯一正确姿势）：

```bash
cd /zeekr_data/ai_workspace/cu13-vllm-20261022
setsid bash launch-vllm-cu130.sh >/tmp/launch.out 2>&1 </dev/null &
disown
```

**坑：`nohup` 不够**——10/24 实测 SSH 断连时 nohup 进程仍被杀，`setsid` 才存活。

### A5 · 就绪判据

```bash
curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8081/v1/models   # 判据：200
grep HugePages_Free /proc/meminfo                                        # 判据：≈2470–2500（KV 已驻留）
grep -cE 'ERROR|Traceback' logs/server-cu130-clean.log 的本轮段           # 判据：0
```

时长：缓存热 **6–10 分钟**；指纹失效触发重编时 **30–60 分钟**（`MAX_JOBS=1`）。
判断在编还是在死锁：`ps` 看 `nvcc/ninja` 进程且 cache 目录有新 `.o` = 正常重编；
日志长时间停在 `Loading safetensors` 且无编译进程 = 停线取证。

### A6 · 冒烟验收（两处坑）

```bash
curl -s http://127.0.0.1:8081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"用一句话解释投机解码"}],"max_tokens":512}'
```

- **max_tokens 必须 ≥512**：启动脚本默认开 thinking（`enable_thinking:true`），
  16/64 token 预算会被思考段吃光 → `content` 为空，误判服务坏；
- 验 `finish_reason`（`stop`=正常收束 / `length`=截断）与中文答案正确性。

### A7 · 断电重启 runbook（全部命令已验证可盲跑）

```
① preflight 认板（SN→串口）→ ② ping/ssh 验网络（IP 丢 → 串口恢复链，修复只走串口）
③ grep HugePages_Total = 23552（0 → 按 hugepage 文档重设）
④ setsid 拉起 launch 脚本 → ⑤ A5 就绪判据 → ⑥ A6 冒烟
```

重启后 JIT 缓存因在持久分区而**不重编**（10/24 实证：重启后 6 分钟就绪、零重编）。

## JIT 编译纪律（事故链定案）

| 事故 | 机理 | 定案 |
|---|---|---|
| 首次 JIT 编译打爆宿主（10/24 晨） | flashinfer 缺省 12 路并发 nvcc，宿主 58G RAM 被 cudafe OOM 打爆 → 引擎死 + sshd 失联 | `MAX_JOBS=1`（2 也曾在 2×nvcc 时 OOM-137） |
| 缓存焐热后自愈 | `.ninja_log` 哈希自洽后永久 no-work-to-do | 预热 fp8/fp4 两个内核 `.so` 后交付 |
| 路径/环境变化全量重编（10/25 实证） | build.ninja 指纹含 sys.prefix 与绝对路径；软链欺骗无效 | 实体落最终路径；接受一次 30–60 分钟重编 |

## 实测基准（10-24/25 基线，MTP4，批=1；11-07 现役形态见下节）

| 项 | 实测 | 备注 |
|---|---|---|
| prefill @8K | 1167.6 tok/s | 10-24 run6 |
| prefill @32K | 577.2 tok/s | |
| prefill @96K | 216.0 tok/s | 448.0s |
| decode 短上下文 | 42.9 / 45.3 / 44.9 tok/s | run6 + 10-25 run7 两轮复测 |
| decode @96K KV | 3.9 tok/s | 长上下文带宽瓶颈，线性衰减 |
| 100K 上下文检索 | PASS 434.1s | 与资格线 10-22 交接一致 |
| 192K+ 长 prefill | **死锁**（3 配置复现） | 已知栈缺陷：同板同模型 llama.cpp 150K/190K 全过；仅留档不在验收口径 |

同栈 A/B（10-25 run7，同题同参同 MTP）：RadixArk decode 45.3 tok/s vs
标准 HF 打包 40.6 tok/s（快 ~10%），prefill 打平——现役选型不换。

## 11-07 增量：三臂投机 + 全模态 + 256K 验证（现役形态）

2026-11-07 全流程重部署并扩展验收（交付包 = 模型库 `qwen38-27b-nvfp4-serve-20261107/`，
含 DEPLOY.md + AOT 模块 + jit-packages 回填包 + 起服/测试脚本 + testdata 全套原始数据）。
与 10-24/25 基线的四个变化：**投机从 MTP4 扩为三臂（AR / MTP n=4 / DFlash2 q=7）、
去掉 `--language-model-only`（文字+视觉全模态）、前缀缓存关 + AutoTuner 关、
池 46G→50G**。AOT 直载（铁律：禁板上 JIT）：flashinfer `fp8_gemm_cutlass.so` /
`fp4_gemm_cutlass.so` 预编译注入 `flashinfer/data/aot/`，`try_load` 命中即不进 JIT。

### 口径（与 10-30 基线对齐）

50G 大页池（25600×2MiB）/ util 0.90 / temp=0 / ignore_eos / max_tokens=128 /
裸补全流式 / 单发 / 板上本地跑（LAN 客户端会被中间设备掐断长空闲流，超长点必须板上本地）。

### KV cache（起服日志 `GPU KV cache size` 实测）

| 臂 | 全模态 KV | 达标线 |
|---|---|---|
| mtp（n=4） | 618,542 tokens | ≥550,000 |
| dflash（DFlash2 q=7） | 368,028 tokens | ≥350,000 |

vision 塔成本：mtp −4,418 tok（−0.16G，几乎无损）；dflash −66,418 tok（−2.46G，
草稿栈放大）。全模态前（语言 only）KV：mtp 622,960 / dflash 434,446。

### decode t/s 全表（temp=0，ignore_eos，128 tok；短点 3 轮/中点 2 轮/长点 1 轮取均值）

| 实测 ctx | AR | MTP n=4 | DFlash2 q=7 |
|---|---|---|---|
| 1,016 | 13.86 | 37.20 | 35.66 |
| 7,976 | 13.71 | 40.38 | 38.73 |
| 31,989 | 13.19 | 32.55 | 31.35 |
| 63,947 | 12.55 | 32.73 | 29.05 |
| 127,864 | 11.45 | 30.04 | 24.53 |
| 199,842 | 10.42 | 24.13 | **29.75** |
| 255,754 | 未测 | 24.24 | **28.19** |

prefill：1K ≈1340–1861 t/s；200K ≈380–411 t/s；256K ≈315–331 t/s
（TTFT 200K ≈486–526s；256K ≈773–812s）。

**长上下文结论：128K 以内 MTP 最快；≥200K DFlash2 反超（200K +23%、256K +16.3%）
且 prefill 双轴全胜（256K：dflash 331 vs mtp 315 t/s）。10-24 的 192K 死锁缺陷
在 11-07 栈上未复现：dflash 192K 单点跑通（411 t/s prefill / 30.95 t/s decode），
两臂 200K 多轮与 256K 全过，旧死锁所在的 prefill 中段已被长上下文连续覆盖。**

### 投机解码健康

- mtp acceptance length ≈ 3.34（n=4，≈84% 接受率，1K ctx）
- dflash 冷启动首读 1.00 是假象，连发数请求后再读：1K ctx ≈3–4 → 长 ctx ≈4.57
  （HF H200 官方 5.46 的 84%）
- 三臂无损性：greedy checksum 严格相等在 tuner 开的 boot 间不成立（tactic→浮点归约序
  分叉）；无损性以投机接受机制（数学无损）+ 臂内自洽为准；tuner 关闭后此分歧源消除

### 多模态验收（文字+视觉）

发真实照片 + 中文提问（temp=0，max_tokens=80），两臂均准确描述人物着装/坐姿/发型；
`prompt_tokens` ≈1098 两臂一致（视觉编码确定性 ✓）。判分看描述内容——模型输出常带
思考前缀（「用户要求……我要……」），属正常行为。

### 11-07 配置坑位（新增三条，其余沿用上文坑位表）

| 症状 | 根因 | 处置 |
|---|---|---|
| 起服 Errno 28 /home 满 | Triton **视觉**内核缓存写默认路径（去 language-model-only 后激活） | 第四缓存根 `TRITON_CACHE_DIR` 指持久分区 |
| 起服 40–50 分钟 + 连环 OOM | AutoTuner 画像风暴；且其缓存持久化机制在 0.32 wrapper × flashinfer 0.7 API 下**静默失效**（hash 目录建了从不写 json） | `--kernel-config '{"enable_flashinfer_autotune": false}'` 永久关闭，放弃缓存路线 |
| prefill 超线性变慢/超时（256K 点位 >40min timeout 实测） | prefix cache 缓存块与大请求互赶 | `--no-enable-prefix-caching` |

256K 两轮失败 → 第三轮（tuner 关 + 前缀缓存关）双臂全过。**起服 5 分钟内完成**
（对比 tuner 开的 boot ~50min 画像风暴）。

### 与 10-30 基线的对照（速度提升归因）

10-30 基线（同板同模型旧配置：tuner 开 + 前缀缓存开 + JIT kernel 缺失）：
prefill 1534→109、decode 36.3→10.6（长上下文崩塌）。11-07 线 prefill **3.8×**
（JIT sm_110a kernel 补齐，AOT 直载）、200K decode **2.3×**（投机长上下文崩塌修复：
seqs=1 / 前缀缓存关 / KV fp8）。

## 与其它线的边界

- 本线 `VLLM_GDN_DECODE_KERNEL=cuda` 配 `qwen3_5` 架构（RadixArk/HF 打包）；
  `qwen3_next` 架构（thor04 线 dense 混合检查点）需要 triton 内核 + 出树插件做
  dense/非 mRoPE/视觉键残留兼容，见 docs/09 的「已验证的组成」节，两线勿混；
- 端口契约：8081 = 本线（10-24/25 基线形态）；11-07 现役三臂/全模态形态在 8080
  面向局域网（`--host 0.0.0.0`）；8082 = llama.cpp 对照工具线（可停）；
- 验收口径：10-24/25 基线 = 100K；**11-07 起 = 256K（dflash/mtp 双臂 255,754 tok
  完整跑通）**；10-24 的 192K 死锁为旧栈缺陷，11-07 栈上已不复现。

## 复现核对清单（给照做的人）

- [ ] `grep HugePages_Total` = 25600（11-07 起；10-24/25 基线为 23552）
- [ ] venv/toolchain/bin 包装脚本内嵌路径与目标机布局一致（A3 判据 0 命中）
- [ ] cu13/lib64 与 libcudart.so 开发链接在位（A2 判据）
- [ ] 启动用 `setsid`，不用 nohup
- [ ] 就绪 = `/v1/models` 200 + 大页 free ≈2475 + 本轮日志 0 ERROR
- [ ] 冒烟 max_tokens ≥512 且看 `finish_reason` 与 content 非空
- [ ] 部署链零外置存储依赖：`grep -r sda1` 于启动脚本与栈目录 = 0 命中
- [ ] （11-07 形态）AOT gemm 两 `.so` 已注入 `flashinfer/data/aot/`（缺了会进板上 JIT，
      违反禁 JIT 铁律）；`TRITON_CACHE_DIR` 等四缓存根全指持久分区；
      `enable_flashinfer_autotune=false` 且日志无 `[AutoTuner]: Tuning` 行；
      `--no-enable-prefix-caching` 在位
