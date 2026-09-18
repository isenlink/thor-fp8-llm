# llama-server 在 DRIVE Thor 上的部署配置速查（Q4_K_M MoE 实例）

> One-liner: A working, tuned `llama-server` configuration for a 35B MoE
> GGUF on DRIVE Thor — every flag explained, with the measured throughput
> each one buys, plus the sampling/thinking knobs and their real effects.
>
> 适用：DRIVE Thor（p3960-0010 / Tegra264），DriveOS 7.0.x，llama.cpp 0.4.0-dev
> 模型实例：Qwen3.6-35B-A3B-UD-Q4_K_M.gguf（35.5B 参数，MoE，K-quant 4-bit）

---

## 一、完整启动命令

```bash
llama-server \
  -m /brand_data/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  -ngl 99 \
  -fa on \
  --host 0.0.0.0 --port 8080 \
  -c 16384 \
  -t 8 \
  --spec-type draft-mtp \
  --reasoning-effort low \
  --temperature 0.2 \
  --top-k 20 \
  --top-p 0.9
```

后台常驻（注意板端 `nohup` 的日志重定向）：

```bash
cd /brand_data/llama-bin/bin
nohup ./llama-server <上面的参数> > /brand_data/llama-bin/server.log 2>&1 &
```

模型加载约 16 秒，日志出现这行即为就绪：

```
I srv  llama_server: listening on http://0.0.0.0:8080
```

## 二、参数逐项说明

| 参数 | 含义 | 为什么这么配 |
|---|---|---|
| `-m <path>` | 模型文件 | Q4_K_M GGUF，21.1 GiB |
| `-ngl 99` | 全部层放 GPU | Thor 统一内存，无 PCIe 拷贝开销 |
| `-fa on` | Flash Attention | 长上下文显著省显存、提速 |
| `--host 0.0.0.0` | 监听所有网卡 | 便于局域网其他主机调用 |
| `--port 8080` | 服务端口 | OpenAI 兼容接口在 `/v1/*` |
| `-c 16384` | 上下文长度 | 见第四节（模型原生支持 262144） |
| `-t 8` | CPU 线程数 | 板上 12 核，留余量给系统与采样 |
| `--spec-type draft-mtp` | ★ MTP 投机解码 | **+48% 吞吐**，见第三节 |
| `--reasoning-effort low` | 思考强度 | 压低思考链长度（软约束，效果有限） |
| `--temperature 0.2` | 采样温度 | 低温度 → 输出确定性高、思考链更短 |
| `--top-k 20` / `--top-p 0.9` | 采样截断 | 配合低温度收紧分布 |

## 三、每个开关买到的性能

同一提示词、同一板卡，逐项实测：

| 配置 | 吞吐 | 相对基线 |
|---|---|---|
| 基线（无 MTP，temp 0.8） | 39.6 t/s | — |
| **+ MTP**（temp 0.6） | **58.6 t/s** | **+48%** |
| + MTP + temp 0.2（思考链更短） | ~65 t/s | +64% |

**MTP 是收益最大的一项**——确认方法与原理见 `mtp-speculative-decoding.md`。

## 四、上下文长度怎么定

模型原生支持 **262144**（256K）上下文，但实际开多大要看 KV 缓存占用：

```
KV 每 token ≈ 0.078 MB     (40 层, 16 attn heads, 2 KV heads, head_dim 256, fp16)

16K  →  1.25 GB
32K  →  2.50 GB
64K  →  5.00 GB
```

板卡 GPU 可见内存池 **47104 MiB**（46 GiB），模型占 21.1 GiB：

| 上下文 | 模型 + KV | 池内余量 | 结论 |
|---|---|---|---|
| 16K | 22.4 GB | ~24 GB | ✅ 宽裕 |
| 32K | 23.6 GB | ~23 GB | ✅ 宽裕 |
| 64K | 26.1 GB | ~20 GB | ✅ 可行 |
| 128K | 31.1 GB | ~15 GB | ✅ 可行 |

**KV 缓存量化（`--cache-type-k q8_0`）实测无收益**——该模型 KV 本身就小，q8_0 反而损失约 1.1% 吞吐（692.22 vs 694.72 t/s @ 16K）。不值得开。

## 五、thinking 模型的采样注意点

Qwen3.6-35B-A3B 是 **thinking 模型**：回答前会先生成思考链，思考也计入 `max_tokens`。

**坑：`max_tokens` 给小 → 思考完没正文**

```json
// 请求 max_tokens=300 时的实际返回
{ "content": "",                      ← 正文空
  "reasoning_content": "1. **分析用户需求**： ... 17 * 23 +" }   ← 思考被截断
// usage: prompt=30 completion=300
```

规避方式（任选）：

| 方法 | 命令/参数 | 效果 |
|---|---|---|
| 加大预算 | 请求里 `max_tokens: 2000+` | 简单可靠，首选 |
| 硬限思考 | `--reasoning-budget 256` | 强制思考在 256 token 内结束 |
| 软约束 | `--reasoning-effort low` | **实测效果有限**，思考链仍有 1300-1500 字符 |
| 关思考 | `-rea off` (`--reasoning off`) | 完全关闭，见下 |

> ⚠️ 实测 `--reasoning-effort low` 对 Qwen 模板的约束力不明显。要真正压短思考，
> **`--reasoning-budget N` 比 effort 靠谱**——它是 token 硬上限。

## 六、调用示例

```bash
curl http://<board-ip>:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "你的问题"}],
    "max_tokens": 2000
  }'
```

健康检查与模型信息：

```bash
curl http://<board-ip>:8080/health          # {"status":"ok"}
curl http://<board-ip>:8080/v1/models       # 含 ftype / n_params / n_ctx_train
```

`/v1/models` 返回的 `meta` 字段很有用，能直接读到量化格式和上下文上限：

```json
"meta": {
  "n_vocab": 248320,
  "n_ctx": 16384,
  "n_ctx_train": 262144,      ← 模型原生上限
  "n_embd": 2048,
  "n_params": 35505251456,
  "size": 22652396032,
  "ftype": "Q4_K - Medium"    ← 量化格式实证
}
```

## 七、运维坑

### 1. 板端没有 curl

Thor 板上默认**不带 `curl`**，`curl: command not found`。调试方式：

- 从局域网其他主机直接访问板卡 IP:端口
- 或在板上用 `python3` 调 `urllib`

### 2. 重启服务要先杀干净

```bash
pkill -f llama-server
sleep 5
pgrep -fc llama-server      # 确认为 0 再起新的
```

残留进程会占着 GPU 内存池不放，新实例会加载失败或行为异常。

### 3. 换参数需重启（服务端默认值不可热改）

`--temperature` / `--top-k` 等作为**服务端默认值**只在启动时生效。要改默认值必须重启服务；
单次请求可以在 JSON body 里覆盖。

### 4. 服务不是持久化的

`nohup` 起的进程**重启板卡即消失**。板卡根分区是只读虚拟块设备、`/etc` 是会被格式化
的 overlay，所以别指望 systemd unit 能活过断电（详见 `overlay-power-loss-recovery.md`）。
需要常驻就用外部守护主机重新拉起。

## 八、一页速查

```bash
# 启动（含 MTP + 低温度）
./llama-server -m model.gguf -ngl 99 -fa on --host 0.0.0.0 --port 8080 \
  -c 16384 -t 8 --spec-type draft-mtp --reasoning-effort low \
  --temperature 0.2 --top-k 20 --top-p 0.9 &

# 就绪判据
grep 'listening on' server.log

# 健康 + 元数据
curl http://<ip>:8080/health
curl http://<ip>:8080/v1/models

# 重启流程
pkill -f llama-server; sleep 5; pgrep -fc llama-server   # 必须为 0

# 读不到 curl 时改从别的主机访问
```

## 九、环境

| 项 | 值 |
|---|---|
| 平台 | DRIVE Thor（p3960-0010，Tegra264） |
| 系统 | DriveOS 7.0.x |
| 计算能力 | sm_101（实测，非 sm_110） |
| 引擎 | llama.cpp 0.4.0-dev（ARM64 + CUDA，交叉编译产出） |
| GPU 可见显存 | 47104 MiB（统一内存池） |
| 模型 | Qwen3.6-35B-A3B-UD-Q4_K_M.gguf，35.5B 参数，ftype `Q4_K - Medium` |

---

## [整理者注] 已移除/脱敏内容清单

本文档由内部工作笔记脱敏改写：
- 板载数据分区路径统一改 `/brand_data/` 代称（同仓库其他文档约定）
- 设备 IP、序列号、账号名、主机代号已移除（`<board-ip>` 占位）
- 技术数据（性能数字、参数值、KV 容量计算、命令）100% 保留
