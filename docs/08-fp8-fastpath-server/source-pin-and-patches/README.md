# DRIVE Thor · 自编件源码 pin + 完整补丁集 + 可公开编译件（2026-09-19）

这个目录解决三个问题：
1. **这个二进制是哪来的** —— 见 `SOURCE-PIN.md`（上游 pin）与 `BUILD.md`（构建配方）。
2. **怎么自己编一份一样的** —— `patches/`（我们的完整补丁集）+ `patches/apply.sh`（一键应用）。
3. **可以公开带走的编译件** —— `bin/`（服务端 + 两个 NVFP4 数值自检工具）。

> 与它配套的是另一个包：`thor-prebuilt-2026-09-19`（开箱即用的启动脚本、实测读数、9 t/s 排障清单、精度自检、可选起草器）。

---

## 0. 两个 tar 包

| 包 | 内容 | 谁需要 |
|---|---|---|
| `thor-source-pin-and-patches-2026-09-19.tar.gz` | pin + 补丁集 + 构建配方 + 官方件链接 | 想自己编译 / 想审计的人 |
| `thor-binaries-2026-09-19.tar.gz` | `bin/` 三个可执行件 | 只想直接跑的人 |

两个包都可独立使用；`SHA256SUMS.txt` 是两个包的校验和。

## 1. `patches/` 里到底有什么（诚实口径）

| 目录 | 内容 | 与随包二进制的关系 |
|---|---|---|
| `patches/01-as-built/` | 12 个 tracked 文件的 diff + 4 个新增 `.cu/.cuh` | **构建前就已定稿**（文件时间戳早于构建时间）⇒ 这是随包二进制的性能路径 |
| `patches/POST-BUILD-DRIFT.tsv` | 构建**之后**才改动的文件清单（含时间戳） | **不在**随包二进制里；含后续实验件（新内核等），我们**没有**随包发布 |

**为什么不全发**：构建之后我们又做了一批实验（新 NVFP4 内核等），其中有的**精度结论未定论**。按"宁可慢也不能错"的原则，本包只发与随包二进制**一一对应**的那一层；未定论的东西不发。

**验收标准（重要）**：`SOURCE-PIN.md` 的 pin + `patches/01-as-built/` 应用后编译出的件，应与随包二进制**行为一致**（读数落在同一档、数值自检 `cpu_failures=0`）。
逐字节 sha256 相同**不是必需**：件是 `not stripped` 的，DWARF 里带源码/sysroot 绝对路径，路径不同 sha256 必不同（要做到逐字节相同见 `BUILD.md` §4）。

## 1.1 能复现到什么程度（用二进制反查过，不是推测）

判定方法：把每个改动里的"高信号标记"（环境变量名、带扩展名的文件名）拿去在随包二进制里逐个找，
并**剔除上游本身就有的名字**（命中它们证明不了任何事）；结果落在 `patches/PATCH-FIDELITY.tsv`（可复跑：`patch_fidelity.py`）。

| 判定 | 数量 | 含义 | 本包发不发 |
|---|---|---|---|
| `IN_CERTAIN` | 16（12 改动 + 4 新增源文件） | 文件 mtime 早于构建 ⇒ 内容就是构建时那一版 | **发**（`patches/01-as-built/`） |
| `IN_EVIDENCE` | 2（`ggml-cuda.cu`、`mmvq.cuh`） | 该 diff 的标记在件里命中 ⇒ 这部分当时在，但之后又改过，当前 diff 是**超集** | **发**（`patches/02-in-binary-post-modified/`，已标注） |
| `OUT_EVIDENCE` | 8（含 `mmq.cu`/`mmvq.cu`/`t4_mmq.*`/`speculative.cpp`/`qwen35.cpp`…） | 标记全不命中 ⇒ 这些改动**不在**件里 | 不发 |
| `UNKNOWN` | 7（`common.cuh`、`mmq.cuh`、`fattn.cu`、`ssm-conv.cu`、`models.h`…） | 没有可判定的标记，此法判不了 | 不发（表里列名） |

**结论**：
- **行为可复现**：pin + 上面发的补丁 → 同一性能档（验收三条见 `BUILD.md`）；
- **逐字节复现做不到**，两个具体原因写在 `BUILD.md` §4；
- 顺带更正一处：`GGML_NVFP4_WIDE_MAX` 只出现在构建后的 `mmvq.cu` 里，**随包二进制不读它**；
  这件真正会读的是 `GGML_CUDA_GRAPH_OPT`、`GGML_MMVQ_MAX`（证据 = 二进制里的字符串）。启动脚本里那条 export 是无害空操作。

## 2. 最小复现路径

```bash
git clone https://github.com/ggml-org/llama.cpp            # 见 SOURCE-PIN.md 的 pin
cd llama.cpp && git checkout 72797e89198ab564fd0e6baa54ab196e8dd1d884
bash /path/to/patches/apply.sh "$PWD" --check   # 先 dry-run，确认每个补丁都能干净应用
bash /path/to/patches/apply.sh "$PWD"           # 再真应用
bash /path/to/toolchain/build-thor-cuda128.sh "$PWD"    # 需要 aarch64 交叉工具链 + CUDA 12.8 sbsa
bash /path/to/toolchain/verify.sh bin/llama-server      # 验收（建议做，别只看编译成功）
```

## 3. 官方件名字与链接

见 `UPSTREAM-LINKS.md`（上游仓库与 commit、CUDA 工具链归档、模型权重仓库、起草器 PR 与社区库）。
凡不是我们产出的文件，都在那里写明**名字 + 具体链接**；本包内不再重复分发它们。

## 4. 本包不含什么

- 模型权重（target 与起草器）——体积与许可自行解决，来源见 `UPSTREAM-LINKS.md`;
- 构建后的实验件（新内核/未定论改动）——见 `patches/POST-BUILD-DRIFT.tsv`;
- 任何凭据、内网地址、账号名、内部主机/板卡代号；内部绝对路径已做**与二进制内一致的等长替换**（替换表本身不列出，避免反向泄漏）。
