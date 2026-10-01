# Thor01 T4/ARES 部署包上线（2026-10-01 定稿发布）

> 本目录 = `tcgen05` 实验分支（T4 tcgen05 MMQ 内核 + ARES A 常驻）的**定案发布版**：
> 精度已定论、与 Thor01 现行生产二进制逐字一致（sha256 `2066e681…`）、三次独立复测通过。
> 原 `tcgen05` 分支的过程记录（未定论时代）保留不动。

- **主文档**：[README.md](README.md)（环境 / 模型清单 / 优化方式 / 构建 / 部署 / 测试 / 实测 / 已知边界）
- **源码**：`source/`（上游 pin `72797e89` + workingtree-diff.patch 104KB / 26 文件 + 工具链 + 板端 deploy.sh）
- **测试**：`test/`（bench8091.py + 三次 200K 原始 jsonl + 短提示三次实测 jsonl）
- **预编译二进制**：[ModelScope · navyyang/thor01-qwen38-27b-dflash2-t4ares-deploy](https://www.modelscope.cn/models/navyyang/thor01-qwen38-27b-dflash2-t4ares-deploy)（13 件，全件 SHA256SUMS）

关键读数：**200K decode 21.9–22.5 t/s**（三次），**短提示 34.5–34.8 t/s**；
draft acceptance 59.8%（短）/ 51.0%（200K）。
