# Thor Automotive Domain Controller NVFP4 Inference Optimization — Complete Engineering Log

> A complete engineering log of optimizing NVFP4-quantized LLM inference on an NVIDIA DRIVE Thor automotive domain controller, covering FP8→GGUF conversion obstacles, numpy/torch ABI pitfalls, and speculative-decoding tuning that raised decode throughput from 11.42 to 26.48 tok/s.

# Thor 域控板 NVFP4 推理优化——完整操作实录与复盘

> 日期：2026-09-08（全天，含停机检修窗口）
> 参与：作者 + AI 助手协作
> 关联文档：[硬件档案](../01-hardware-recon/hardware-archive.md)、[社区情报调研](./community-intelligence.md)
> 目标：NVFP4 量化路线 decode ≥ 25.89 tok/s（社区网友水平）
> **结果：26.48 tok/s，目标达成 ✅**

---

## 一、时间线总览

```
上午        硬件档案定稿、NFS 贯通、42G 池固化、交叉编译链就绪
下午        依赖链攻坚（清华源）→ FP8→GGUF 转换打通（3 层障碍连破）
17:00前后   QUASAR 调研确认 → 下载 → 转换（numpy ABI 大坑）→ 上板基准
17:55-18:15 QUASAR 下载 + 三模型基准 + 主力换血
18:30前后   MTP K7 实验（负收益）→ 对外 API 上线
~19:00      64K 上下文上线（q8_0 KV）→ 用户质疑速度 → 网络调研
~19:30      水冷渗漏紧急停载 → 用户断电检修
19:30-21:30 停机窗口：社区经验深挖（sudoingX/qwen38-mtp 等）
21:30前后   修复完成 → EXP1 实验一次命中 → 26.48 tok/s ✅
22:40前后   NFS 自动挂载 systemd 化 → 全部闭环
```

---

## 二、完整操作过程

### 阶段 1：转换环境攻坚（转换主机）

**背景**：llama.cpp 的 `convert_hf_to_gguf.py` 需要 torch 全套依赖。转换主机上用 systemd-run 跑依赖安装（unit 名 `convert-deps2`），遇到清华源 + requirements 钉版本的连环坑。

**关键坑与解法**：

| 坑 | 现象 | 解法 |
|---|---|---|
| pytorch.org 官方源假死 | 79MB/196MB 零速停滞 | 切清华源 `pypi.tuna.tsinghua.edu.cn/simple` |
| requirements 钉死 torch==2.11.0+cpu | 清华装了 2.14 全量版后 requirements 又要回 pytorch.org 下 190MB，56KB/s 假死 | 停 unit，前台手动装；**torch 版本不必跟 requirements 死磕** |
| **numpy/torch ABI 错配（最大坑）** | `torch.from_numpy` 报 `TypeError: expected np.ndarray (got numpy.ndarray)`——三个 torch 版本（2.14/2.11×2）+ 重装 numpy 全部同样错误 | 根因：torch cp312 轮子是 numpy 2.x ABI 编译的，运行时 numpy 必须也是 2.x。**正确组合：numpy 2.5.3 + torch 2.11.0+cpu**。中途降级 numpy 到 1.26 是画蛇添足反而制造错误 |
| torch 轮子下载慢 | 清华主索引无 +cpu 变体，pytorch.org 被墙 | **SJTU 镜像直连**：`https://mirror.sjtu.edu.cn/pytorch-wheels/cpu/` 下载 10MB/s+，20 秒 190MB |

**教训**：
- pip 装完必须用最小用例验证（`torch.from_numpy(np.arange(10))`），不能只看 import 成功
- 报错文案 "expected X (got X)" 字面矛盾时 = 类型对象来自不同 ABI 编译，查编译侧与运行时库版本配套

### 阶段 2：FP8 → GGUF 转换打通（转换主机）

模型：`~/models/dflash2_fp8/base`（Qwen3.8-27B FP8 safetensors，66 分片）

**三层障碍连破**：

1. **0 张量导出（只有 10MB 元数据）**
   - `config.json` 的 `architectures = Qwen3_5ForConditionalGeneration`
   - 该名字在 llama.cpp 新版注册表同时注册给了 `Qwen3VLVisionModel`（mmproj 视觉投影类）和 `Qwen3_5TextModel`
   - 多模态分发器把 ConditionalGeneration 分给了 VisionModel → 纯文本权重全被跳过
   - **解法**：patch config.json → `Qwen3_5ForCausalLM`（纯文本路径）

2. **patch 后仍然 0 张量**
   - llama.cpp `conversion/base.py` 硬编码 `prefix = "model"` 找分片，我们分片叫 `layers-*.safetensors`
   - **解法**：66 个分片 symlink 成 `model-XXXXX-of-00066.safetensors` 标准命名 + 重写 `model.safetensors.index.json` 的 weight_map（Python 脚本 1 分钟搞定）

3. **转换成功**：n_tensors = 866, 产出 54.66GB BF16 全精度 GGUF（FP8 反量化翻倍）
   - 母版存转换主机 `/mnt/sdb2/models/qwen3.8-27b/`（根盘 96% → 84% 解放）
   - 按"NVFP4 主线优先"决策，q8_0 量化搁置，母版留作对照储备

**注意**：QUASAR 模型是原生标准命名（model-000XX-of-00005），不需要 symlink 步骤；只需 config patch。

### 阶段 3：QUASAR QAT-NVFP4 转换与部署（主线正军）

**模型情报**：
- `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4`（hf-mirror 可拉）
- QAT 量化感知训练版 NVFP4——训练时模拟量化，精度损失远小于后量化
- 256K 训练上下文，vLLM 官方 MTP 支持，QAT 版 MTP acceptance 0.897（unsloth 后量化版仅 0.788）
- 同款 `Qwen3_5ForConditionalGeneration` 架构 → 复用 config patch 套路

**转换流程**（可复用脚本化）：
```bash
# 1. 下载（hf-mirror.com，systemd-run 断连免疫）
curl -sL -C - --retry 3 -o model-0000$i-of-00005.safetensors \
  "https://hf-mirror.com/QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4/resolve/main/..."

# 2. config patch
python3 -c "import json; c=json.load(open('config.json')); \
  c['architectures']=['Qwen3_5ForCausalLM']; json.dump(c,open('config.json','w'),indent=2)"

# 3. 转换（含 base.py 的 mmap patch，见下）
~/.venv_convert/bin/python convert_hf_to_gguf.py ~/models/quasar_nvfp4 \
  --outfile ~/models/quasar_nvfp4/QUASAR-NVFP4-text.gguf --outtype f16
```

**额外坑**：NVFP4 真量化权重触发 lazy 路径 mmap bug
- `base.py:2653` `byteswap_tensor(tensor.mmap_bytes(), ...)` → memmap 类型错误
- patch：`np.asarray(tensor.mmap_bytes())`（实际由 ABI 修复解决，patch 保留无害）

**产物**：`QUASAR-NVFP4-text.gguf` = 19.65GB（**真 NVFP4 4bit 保持**，1858 张量）→ scp 上板（~35MB/s，6 分钟）

### 阶段 4：三模型基准测试（板上）

```
llama-bench -m <model> -p 128 -n 64 -ngl 99
```

| 模型 | 体积 | pp128 (prefill) | tg64 (decode) | 特点 |
|---|---|---|---|---|
| COMPACT-LOW | 14.1G | 209.83 t/s | 10.78 t/s | 体积小省 KV 空间 |
| HIGHEST | 21.6G | **232.32 t/s** | 9.67 t/s | prefill 快（量化分组规整）|
| QUASAR-QAT | 18.3G | 204.64 t/s | 9.48 t/s | **质量最优**（QAT）|

**结论**：裸速三家同水平（±5%）——网友 25.89 的优势来自**运行时调优**而非模型本身。物理本质：
- prefill = 计算密集型（tensor core 吃满，232 t/s 合理）
- decode = 带宽密集型（每 token 读全量权重，体积大 53% → 慢 10%，完全符合带宽模型）

### 阶段 5：对外 API 标准化

最终启动命令（v2 服务脚本 `/brand_data/ai_workspace/tools/thor_service.sh`）：

```bash
/brand_data/ai_workspace/tools/llama-server \
  -m /brand_data/ai_workspace/models/QUASAR-NVFP4-text.gguf \
  --alias quasar-nvfp4-27b \
  -ngl 99 -c 65536 \
  -fa on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --spec-type draft-mtp --spec-draft-n-max 12 --spec-draft-p-min 0.6 \
  --parallel 1 \
  --host 0.0.0.0 --port 8080
```

- **API**：`http://192.168.1.101:8080/v1`，OpenAI 兼容，无 KEY（任意/不填均可）
- **模型 ID**：`quasar-nvfp4-27b`（--alias 简化）
- **上下文**：64K（FP16 KV 装不下 64K——42G 池 - 19.65G 权重 = 22G 可用，64K×0.5MB=32G 超限；q8_0 KV 16G ✓）
- 管理命令：`./thor_service.sh {start|stop|status|bench}`

### 阶段 6：水冷渗漏紧急停机

- 立即 `pkill -f llama-server`，确认进程清零、端口关闭、后台任务为零
- 温度回落：tj 67.3°C 且持续下降
- **数据零风险**：全部配置持久化（42G 池 sysctl/DHCP/账号/NFS），断电零损失
- 停机窗口用于网络调研（见阶段 7）

### 阶段 7：停机期间社区经验调研（胜负手）

**信息源 1：sudoingX/qwen38-mtp**（40 人 53 配置社区实测库）
https://github.com/sudoingX/qwen38-mtp

七条黄金规则中与我们直接相关的：
- **规则 2**：`--spec-draft-p-min` 0.60-0.75 置信门控对带宽受限机型帮助巨大（起草前先看置信度，低置信跳过深起草）；快速卡上反而负收益
- **规则 5**：必须 `--parallel 1` 测量（并行 4+ 投机优势消失）
- **规则 1**：n-max 甜点因卡而异；**带宽受限 APU 类甜点是 12**（RTX 24GB 卡是 2）
- 规则 6：上游 llama.cpp 每周都在优化 qwen3_5 路径，新构建裸速 +10-15%

**决定性对标数据**：Ryzen AI Max+ 395（带宽受限 APU，与我们同款瓶颈）：
```
基线 11.5 tok/s → 调优后 23.7 tok/s（翻倍）
配方：n-max 12 + p-min 0.6-0.75 门控
我们起点 11.42 tok/s —— 几乎一模一样
```

**信息源 2：hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4**（L4 24GB 深度剖析）
- decode 瓶颈机制实锤：反量化烧功耗 → 功耗墙压 SM 频率 → 总线填不满（带宽利用率仅 84%）
- **这 16% 只能靠"少做 ALU 功"找回 = 换 kernel 家族（tcgen05 方向正确）**
- MMQ kernel 路由 `GGML_MMVQ_MAX=2`：+16.2%（零代码，环境变量级）
- 成本模型：`tok/s = mean_accepted_length / (verify_pass_cost + drafting_cost)`

**信息源 3**：vLLM 官方 recipe（QUASAR acceptance 基线 0.897）

完整调研见 [community-intelligence.md](./community-intelligence.md)。

### 阶段 8：修复后实验矩阵（一次命中目标）

| 实验 | 配置 | 高可预测任务 | acceptance | 创作任务 |
|---|---|---|---|---|
| 基线（K3 默认） | n-max 3, 无门控 | 11.42 t/s | 40.6% | 11.42 t/s |
| K7（更早实验） | n-max 7, 无门控 | 7.24 t/s ❌ | 18% | — |
| **EXP1** | **n-max 12, p-min 0.6, fa on, parallel 1** | **26.48 t/s ✅** | **85.9%** | 11.02 t/s |
| EXP2 | n-max 8, p-min 0.5 | ~21.2 t/s | 81.1% | 11.27 t/s |

**EXP1 深度解析**：
- mean accepted length：2.19 → **9.34**（一次验证出 9 个 token，带宽效率 ×4）
- p-min 门控的作用：低置信任务自动退化浅投机，**垃圾预测不再浪费验证算力**
- 创作任务（temperature 0.8）acceptance 自然掉到 55%——这是 MTP 头预测力的物理上限，EXP1 配置下不掉速（11.02 ≈ 基线），等价于"免费期权"
- K7 早期失败原因：无门控时深起草 = 大量低置信 draft 浪费验证算力

### 阶段 9：NFS 自动挂载 systemd 化

**根因链**：fstab 条目一直在（且有重复两行）→ 但 `mount.nfs` helper 只在 `/home/user/bin/`（rootfs 只读放不进 `/sbin`）→ 开机 fstab 挂载找不到 helper → `nofail` 静默跳过

**❌ 错误方案（踩坑重犯）**：`ln -sf ... /sbin/mount.nfs`（只读）+ `remount,rw`（车载安全设计，恒只读，不该碰）

**✅ 正确方案**：systemd service 单元（放可写的 `/etc/systemd/system/`），ExecStart 全路径调用：

```ini
# /etc/systemd/system/models-mount.service
[Unit]
Description=Mount NFS models repository
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
ExecStart=/home/user/bin/mount.nfs <x86主机IP>:/mnt/sdb2/models /media/models -o vers=4
Restart=on-failure
RestartSec=15

[Install]
WantedBy=multi-user.target
```

启用：`sudo systemctl daemon-reload && sudo systemctl enable --now models-mount.service`

**验证结果**：enabled ✓ / ExecStart exit 0 ✓ / 挂载在线 ✓ / fstab 重复行清理 ✓
（oneshot 成功后显示 inactive 是正常状态；彻底闭环待下次自然重启验证）

---

## 三、最终状态清单

| 项 | 状态 |
|---|---|
| 对外 API | `http://192.168.1.101:8080/v1`，无 KEY，OpenAI 兼容 |
| 模型 ID | `quasar-nvfp4-27b` |
| decode 性能 | **26.48 tok/s**（高可预测任务，超 25.89 目标）|
| 创作任务 | 11.02 tok/s（= 基线不掉速，MTP 头预测力上限）|
| 上下文 | 64K（q8_0 KV；FP16 物理 44K 上限）|
| 服务管理 | `./thor_service.sh {start\|stop\|status\|bench}` |
| NFS 自动挂载 | systemd unit，开机自启+失败重试 ✓ |
| CUDA Graph | 默认开启（日志 `graphs reused` 确认）|
| 模型资产 | QUASAR GGUF 19.65G 在板；HIGHEST 23.2G 在板；COMPACT-LOW 15.2G 在板；FP8 母版 54.66G 在转换主机 |

## 四、踩坑汇总（按严重度）

1. **numpy/torch ABI 错配**：`from_numpy` 报 "expected np.ndarray (got numpy.ndarray)" = 两侧 numpy ABI 不一致。正确组合 numpy 2.5.3 + torch 2.11.0+cpu。**装完必跑最小用例验证**
2. **transformer 架构注册分发**：`Qwen3_5ForConditionalGeneration` 被 VLM 类抢走 → 纯文本模型 patch 成 `ForCausalLM`
3. **非标准分片命名**：llama.cpp 只认 `model*` 前缀分片，其他命名需 symlink + 重写 index
4. **MTP 调参反直觉**：K7 无门控反而慢 30%（acceptance 18%）；**p-min 门控是深起草的前提**
5. **测量污染**：`--parallel` 非 1 时基线虚低 20%，A/B 结论全错
6. **监控盯错目标**：journal 日志里 heredoc 回显会污染 grep 完成标记（QUASAR 下载误报"完成"一次）；盯 unit 状态 + 文件数才是真指标
7. **rootfs 只读再次踩坑**：/sbin 链接方案不能提（第二次犯了）；正确思路永远是"放可写区 + 全路径引用"
8. **网上数据要问条件**：网友 25.89 的 K7 是 vLLM 树形投机栈，llama.cpp 链式直接抄参数会负收益；但**机型瓶颈画像一致时配方可迁移**

## 五、后续路线（按优先级）

1. **创作类任务突破**：DFlash2 块扩散起草——llama.cpp 已有 `draft-dflash` 类型（已编译进现有二进制），需要 Qwen3.8 的 DFlash GGUF draft 头（找现成的或从 DFlash2 的 2.1G FP8 头转换）
2. **MMQ kernel 路由**：`GGML_MMVQ_MAX=2` 环境变量（L4 实测 +16.2%，零改动可试）
3. **上游重编译**：转换主机拉最新 llama.cpp master 交叉编译（预期裸速 +10-15%）
4. **交接主机**：tcgen05 prefill kernel 集成 llama.cpp 主线（prefill ×2 潜力）；交接主机 = 2× Xeon 170hx（48C/96T）+ 128G RAM
5. **下次自然重启**：验证 NFS 自动挂载闭环
6. **64K 以下不测**（用户指示）：一切基准从 64K 起

## 六、经验沉淀（给后来人）

1. **"模型一样，速度差 2 倍"先查运行时**：投机解码的 p-min 门控、parallel 设置、flash-attn 开关，比换模型影响大
2. **带宽受限机型的黄金配方**：深起草（n-max 8-12）+ 高置信门控（p-min 0.6）+ fa on + parallel 1
3. **报错字面矛盾 = ABI 问题**："expected np.ndarray (got numpy.ndarray)" 这种自我矛盾报错，查编译侧与运行时库版本
4. **社区实测库 > 官方文档**：sudoingX 的 53 配置表 10 分钟解决的调参问题，自己扫要半天
5. **长任务测量铁律**：盯对 unit 名/文件数等硬指标；journal 日志回显会污染 grep
6. **先分析后操作**：所有 sudo/只读区操作前查档案（本案：/sbin 踩坑重犯的教训）

---

[整理者注] 以下内容已按脱敏规则移除/替换（供复核）：

- **内网 IP**：对外 API 地址（板端 192.168.1.101，保留）；NFS 服务器地址改写为 `<x86主机IP>` 占位（非本机，未记录）
- **机器代号**：2 个内部机器代号（转换用主机 ×1、后续交接 Xeon 主机 ×1）→ 分别替换为"转换主机"/"交接主机"；相关 systemd unit 名与挂载点同步更名去除代号
- **账号名/路径**：含真实账号名的 home 路径 1 处 → `/home/user/`；参与人登录名（2 个）→ 统一为 `user`
- **人名**：1 位真实人名 + 所属部门 → "作者"；AI 助手产品名 → "AI助手"
- **疑似内部项目代号**：后续路线中 1 处内部项目代号 → 以模型系列名" DFlash2"指代
- **本文件中未发现**以下类型敏感内容，故无需对应处理：板卡 SN 号、密码/token/凭据（未加 "[已移除凭据]" 标记）、渠道商/解锁/刷机/锁机相关段落
- **路径代称说明**：`/brand_data/` 为板载数据分区挂载点在本笔记中的**代称**。DriveOS 板上该分区（约 105G，板载 vblkdev，与只读根分区独立）的原路径名含车辆品牌字样，为保持品牌中立统一写作 `/brand_data/`。读者在自己板卡上执行 `ls /` 即可看到真实分区名；文中所有 `/brand_data/ai_workspace/...` 对应"数据分区下的 AI 工作区"。GitHub/镜像站等公开 URL 原样保留。
