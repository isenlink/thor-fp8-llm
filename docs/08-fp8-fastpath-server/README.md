# 08 · 预编译服务端（sm_101a · NVFP4）— 源码补丁与部署文档

> 本目录内容来自 2026-09-19 的对外整理包（含完整补丁集 + 构建配方 + 启动脚本 + 实测读数）。
> **预编译二进制不在本仓库**（体积原因），通过网盘分发，链接见下。

## 网盘下载（预编译二进制）

| 文件 | 大小 | sha256 |
|---|---|---|
| `llama-server-aarch64-sm101a-nvfp4` | 74.7 MiB | `a7feb90f5d8dd95feb09a8da42e8e1d1fd5126b341a54a80ecaea322a25e40a1` |
| `b3-correctness`（NVFP4 数值自检工具） | 59.7 MiB | `712cb1bcb3c9be2a478f4bb699c898d11463ea541b41ae592fb253e286f6296d` |
| `b3-correctness2`（同上，第二版） | 59.7 MiB | `fa46218bbefb004dd08736533b752cdf18c1acbb3229896237d84705b31e19ad` |

> 🔗 **百度网盘链接：待补**（发布时填入）

## 目录结构

```
source-pin-and-patches/   自编译所需全部材料（可复现）
  SOURCE-PIN.md           上游 pin：llama.cpp @ 72797e89（2026-09-10）
  BUILD.md                构建配方（交叉编译 + CUDA 12.8 sbsa）与三条验收标准
  UPSTREAM-LINKS.md       官方件/权重/起草器的名字与链接（本仓不再分发）
  patches/
    01-as-built/          12 个补丁 + 4 个新增 FP8 源文件（确定在随包二进制里）
    02-in-binary-post-modified/  2 个补丁（标记命中但为超集，已标注）
    PATCH-FIDELITY.tsv    保真度判级（逐补丁标注：在件里/超集/不在/无法判定）
    POST-BUILD-DRIFT.tsv  构建后改动清单（这部分不随包发布：精度未定论，"宁可慢也不能错"）
  toolchain/              工具链文件 + 构建脚本 + 验收脚本
prebuilt-docs/            直接跑预编译件所需的全部文档
  README.md               ★ 主文档：读数口径、逐参数说明、9 t/s 排障清单、精度自检
  02-launcher/            启动/停止/宿主机准备脚本
  03-reference/           实测读数、负结果黑名单、精度自检清单、9 t/s 排障
```

## 快速开始

**想自己编译**（完整审计与可复现）：
```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && git checkout 72797e89198ab564fd0e6baa54ab196e8dd1d884
bash /path/to/patches/apply.sh "$PWD" --check   # dry-run
bash /path/to/patches/apply.sh "$PWD"           # 应用
bash /path/to/toolchain/build-thor-cuda128.sh "$PWD"    # 需 aarch64 交叉工具链 + CUDA 12.8 sbsa
bash /path/to/toolchain/verify.sh bin/llama-server      # 验收
```

**直接跑**（下载网盘二进制后）：
```bash
export MODEL=/path/to/target-nvfp4.gguf
export DRAFT=/path/to/dflash2-draft.gguf
export SPEC=dflash2 CTX=131072 SPEC_N=7
bash prebuilt-docs/02-launcher/start-server.sh
```
详见 [prebuilt-docs/README.md](prebuilt-docs/README.md)（逐参数说明 + 排障清单 + 精度自检）。

## 实测读数（口径：看 ms/步，不看 t/s）

| 上下文 | t/s | ms/步 | 接受率 |
|---|---|---|---|
| 128K cold | 22.178 | 234.5 | 0.604 |
| 200K cold | 19.698 | 253.8 | 0.573 |

- 200K 目标 30 t/s 未到（差 21%），本件是"能用的最好一件"，不是"最快的一件"——诚实口径。
- tcgen05 内核路线更快（8K 41.8 / 128K 27.4 / 200K 24.1 t/s），但属另一条线，见 `scripts/tcgen05-nvfp4-gemv/`。
- 精度：150 题 × 8 方向（chat 口径、temp=0）32K 与 200K 均 150/150 全对、0 空回答。

## 与本仓其他部分的关系

- 投机草稿配方（MTP/DFlash2 深度甜点）→ [docs/04-nvfp4-optimization/speculative-drafting-recipes-2026-09-16.md](../04-nvfp4-optimization/speculative-drafting-recipes-2026-09-16.md)
- KV 预算 / 256K 长上下文 → [docs/05-system-tuning/](../05-system-tuning/)
- 基准方法论（口径陷阱）→ [docs/06-benchmarks/benchmark-methodology-2026-09-16.md](../06-benchmarks/benchmark-methodology-2026-09-16.md)
- tcgen05 内核（更快的实验线，代码级整理包）→ [scripts/tcgen05-nvfp4-gemv/](../../scripts/tcgen05-nvfp4-gemv/)；该内核线的完整源码与构建材料在 `tcgen05` 分支（整理中）
