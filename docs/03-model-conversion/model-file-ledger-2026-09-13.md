# 14-模型文件台账（2026-09-13 整理）

> 范围：Thor 项目产生/使用的全部模型文件。状态分：✅生产 / 📦备份归档 / ❌已证伪 / 🗑建议清理。
> hash 为 SHA256（除注明 md5 外）。

## 1. 生产在用

| 文件 | 位置 | 大小 | SHA256 | 说明 |
|---|---|---|---|---|
| RadixArk-F8attn-v2.gguf | 板 `/ai_workspace/models/` | 19.57 GiB | `d15dac916823f590fa8e584ae18a37c62d61947ce33df943a658c83766d2e7e9` | ✅ **生产模型**。NVFP4 MLP（193 张量）+ F8_E4M3 注意力/GDN 投影（208 张量），arch=qwen35，内置 MTP 头 |
| 同上（主备份） | x86主机 `~/work/thor-driveos/models/` | 同上 | 同上（已核对一致） | 📦 板上文件的同源备份 |

- 格式来源：radixark/ HF 源（ModelOpt FP8+NVFP4 混合）经自写转换器 C3 产出
- 张量审计：FP8×208 / NVFP4×193 / scale×995 / 未变×798，verdict=pass（radixark/tensor-audit.json）
- v1→v2 文件差仅 17.6KB（元数据级）；v2 是经过全部配对验证的生产定版，v1 未再使用

## 2. 板上对照资产

| 文件 | 位置 | 大小 | SHA256 | 状态 |
|---|---|---|---|---|
| QUASAR-NVFP4-text.gguf | 板 `/ai_workspace/models/` | 18.30 GiB | `5358cf07df4779e1bf14c518d488b4ee3dca1ae00f98106428db248e83623408` | 📦 老栈对照模型（R1-R8 基准用）。已被 RadixArk 路线取代，保留作回归对照 |

## 3. x86主机 历史产物（~/work/thor-driveos/models/）

| 文件/目录 | 大小 | 状态 | 说明 |
|---|---|---|---|
| RadixArk-F8attn.gguf（v1） | 19.57 GiB | 📦 | F8 attention 首版，被 v2 取代 |
| RadixArk-Qwen3.8-27B-NVFP4.gguf | 26.3 GiB | 📦 | C3 首版转换（FP8 投影升 BF16），R10-R13 测试用；acceptance 81.8% 证据即来自此版 |
| RadixArk-F8all-v3.gguf | 26.5 GiB | ❌🗑 | **A2 失败产物**（MLP→F8 重编码，median 3% 误差致 MTP acc 0.68% 全灭）。仅留作反面教材，可删 |
| Qwen3.8-27B-Q4_K_M.gguf | 0 字节 | 🗑 | 空占位文件，删除 |
| Qwen3.8-27B-Q4_K_M/ | 空目录 | 🗑 | 删除 |
| qwen3.8-27b-nvfp4/ | HF 目录 | 📦 | RadixArk NVFP4 HF 源（含 model_mtp.safetensors） |
| qwen3-30b-a3b-awq/、qwen3.6-35b-a3b-fp8/ | HF 目录 | 📦 | 早期 MoE 探索，与主线无关 |

## 4. x86主机 出板归档（~/work/thor-driveos/models-offboard/，共 36 GiB）

| 文件 | 大小 | 状态 | 说明 |
|---|---|---|---|
| Qwen3.8-27B-DFlash2-Q4_K_M.gguf | 1.07 GiB | ❌ | DFlash2 draft 头（md5 958589e3…已校验）；路线已关闭（06-实验6/7） |
| Qwen3.8-27B-NVFP4-MTP-COMPACT-LOW.gguf | 14.1 GiB | 📦 | williamliao 系 NVFP4，未胜出 |
| Qwen3.8-27B-NVFP4-MTP-HIGHEST.gguf | 21.6 GiB | 📦 | 同上 |

## 5. x86主机 radix-gguf/ 与 radixark/

| 文件 | 大小 | 状态 | 说明 |
|---|---|---|---|
| radix-gguf/Qwen3.8-27B-NVFP4-Quality-v2.gguf | 14.95 GiB | ❌ | SHA256 `6007d4b0151bff…`（11-文档有记录）。A4 路线：MTP 0% acceptance + 新 build 数值错误，已排除 |
| radixark/*.safetensors（3 shards） | 20.3 GiB | 📦 | RadixArk HF 源，F8attn 系列的转换源头；含 conversion-manifest.json / tensor-audit.json |

## 6. 板上曾用已清理

- Qwen3.8-27B-MTP-Q4_K_M.gguf（16.8G，md5 d4f2e047…）：官方 Q4_K_M + MTP，A4-3/4 对照用，已出板
- RadixArk-F8all-v3.gguf：A2 失败后已从板上清理（x86主机 仍有副本，见 §3）

## 7. 站外归档

- Windows 归档机共享目录 `Temp/Models/`（284GB，已分类整理）：LLM/Image/Video/ASR 全族，详见 BFWIN10-MODELS-ORGANIZED.md
- 遗留：`20260814-88` 子目录 NTFS 权限未继承，3 项内容待移（AWQ 残件 / MiniMax-H3 / faster-whisper）

## 8. 处置建议（待用户确认后执行）

1. 删 x86主机 `models/Qwen3.8-27B-Q4_K_M.gguf`（0 字节）与同名空目录
2. RadixArk-F8all-v3.gguf（26.5G）移至 models-offboard/ 或直接删——A2 结论已落盘，文件本身无复用价值
3. 板上只保留两个模型（生产 + QUASAR 对照）现状合理， 余量 57G 健康
