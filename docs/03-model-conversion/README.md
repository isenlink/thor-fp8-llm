# 模型转换：FP8 → GGUF

> FP8 safetensors → GGUF 转换过程中连续破解的三层障碍实录，完整过程
> （含逐层排障日志与 patch 细节）见：
>
> **[../04-nvfp4-optimization/nvfp4-optimization-log.md](../04-nvfp4-optimization/nvfp4-optimization-log.md) 阶段 2**

## 三层障碍速览

### 障碍 1：0 张量导出（只有 10MB 元数据）

`config.json` 的 `architectures = Qwen3_5ForConditionalGeneration`，这个名字在
llama.cpp 新版注册表同时注册给了 `Qwen3VLVisionModel`（mmproj 视觉投影类）和
`Qwen3_5TextModel`。多模态分发器把 ConditionalGeneration 分给了 VisionModel →
纯文本权重全被跳过。

**解法**：patch config.json → `Qwen3_5ForCausalLM`（纯文本路径）。

### 障碍 2：patch 后仍然 0 张量

llama.cpp `conversion/base.py` 硬编码 `prefix = "model"` 找分片，而我们的分片
命名为 `layers-*.safetensors`。

**解法**：适配分片命名（详见完整实录）。

### 障碍 3：numpy/torch ABI 错配（最大坑）

`torch.from_numpy` 报 `TypeError: expected np.ndarray (got numpy.ndarray)`——
字面矛盾的报错。三个 torch 版本 + 重装 numpy 全部同样错误。

**根因**：torch cp312 轮子是 numpy 2.x ABI 编译的，运行时 numpy 必须也是 2.x。
**正确组合：numpy 2.5.3 + torch 2.11.0+cpu**。中途降级 numpy 到 1.26 反而制造错误。

## 教训

- pip 装完必须用最小用例验证（`torch.from_numpy(np.arange(10))`），不能只看 import 成功
- 报错文案 "expected X (got X)" 字面矛盾时 = 类型对象来自不同 ABI 编译，查编译侧与运行时库版本配套

## 复现材料（2026-09-14 补）

### 模型文件（体积原因不入库）
| 文件 | 大小 | SHA256 | 获取方式 |
|---|---|---|---|
| `RadixArk-F8attn-v2.gguf` | 19.57 GiB | `d15dac916823f590fa8e584ae18a37c62d61947ce33df943a658c83766d2e7e9` | 由 HF 源经下述步骤转换产出；需单独分发（HuggingFace / 网盘），下载后**必须核对 SHA256** |

> 19.57 GiB 超过仓库承载，故只入库转换器与步骤；拿到文件后用 `sha256sum` 核对上表 hash 即可确认与生产一致。

### 转换步骤
1. 取 HF 源（ModelOpt FP8+NVFP4 混合量化权重，含 `model_mtp.safetensors`）。
2. 用打上 F8 支持补丁的 `convert_hf_to_gguf.py` 转 GGUF（三层障碍见上：architectures 改名、分片命名适配、numpy/torch ABI 配套）。
3. 运行 `scripts/gguf-blk64-to-f8.py` 把 `blk.64`（MTP draft 层）8 个二维权重重编码为 F8_E4M3 + per-tensor scale；其余张量与元数据原样拷贝。
   ```bash
   python3 scripts/gguf-blk64-to-f8.py <输入.gguf> <输出.gguf>
   ```
4. 校验：张量审计应为 FP8×208 / NVFP4×193 / scale×995 / 未变×798。
