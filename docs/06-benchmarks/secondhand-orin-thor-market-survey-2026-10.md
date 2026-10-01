# 闲鱼捡漏新能源车自动驾驶算力模块（Orin-X / Thor）跑千问大模型——情报汇总

> One-liner: 二手车规域控（Orin-X / DRIVE Thor 拆车件）捡漏指南与社区实测汇总——
> 丝印料号识别、搜索黑话、到手验证命令、千问系模型实测速度与可行性边界。
>
> 整理日期：2026-10-01
> 来源：[无忧网联 5ULINK《闲鱼捡漏新能源车自动驾驶算力模块：用Orin-X/Thor跑千问大模型实测》](https://www.5ulink.com/archives/1590)（2026-09 发布，信息整理帖，汇总自 linux.do 等社区实测与 NVIDIA 官方文档）
> 定位：与 `community-field-results.md`（同款 Thor 板主的部署经验）互补——本文是**买板前的市场与硬件识别视角**。

## 为什么这波火了

- 存储超级周期（2025 下半年起 DRAM/NAND 涨价，预期贯穿 2026）+ 显卡缺货，把"本地跑大模型"的显存门槛推高
- 车企淘汰的智驾域控（Orin-X / Thor，本质是车规版 Tegra，统一内存 32~128GB、自带 CUDA、低功耗）经事故车拆解、工程样机处置流入二手市场，价格远低于同显存消费卡
- 2026-07-21 NVIDIA 上调 Jetson 全线价格 57%~101%（AGX Orin 32GB 模块翻倍、AGX Thor 开发者套件 $3499→$5499）且官网缺货，进一步抬升拆车件心理价位

## 硬件识别：看丝印，不看标题

车载域控芯片无对外文档，**丝印料号是唯一可信线索**：

| 丝印 / 料号 | 真实身份 | 统一内存 | 带宽 | 算力（INT8） |
|---|---|---|---|---|
| TA990SA-A1 | Orin-X 量产版 | 32GB LPDDR5 | 204.8 GB/s | 254 TOPS |
| TA990SA-QS-A1 | Orin-X QS/工程相关版 | 32GB | 204.8 GB/s | 254 TOPS |
| TA990SA-CS-A1 | Orin-X CS 版（小鹏域控上见过） | 32GB | 204.8 GB/s | 254 TOPS |
| TA1080SA / TA1080 | Thor-U | 常见 64GB 级 | 273 GB/s | 700 TOPS 级 |
| TA1080 + ENG | Thor-U 工程样品 | 同上 | 273 GB/s | 同上 |
| TA1090SA / TA1090 | Thor-X | 64~128GB 级 | 273 GB/s | 1000~2000 TOPS 级 |
| TA1090 + CUSTSAMPLE | Thor-X 客户样品 | 同上 | 273 GB/s | 同上 |

注意事项：

- Orin-X 的 254 TOPS（INT8）/约 200GB/s 带宽与 NVIDIA 官方 DRIVE AGX Orin 文档对得上；Thor 官方口径为相对 AGX Orin 约 7.5× AI 算力、128GB 版最高 2070 FP4 TFLOPS
- ⚠️ 车规 Thor 各子版本（X/U/S/Z）算力公开口径不一致（一说 X=1000T/U=700T；另有资料称 S=700T/U=500T/Z=300T），**以实物丝印和到手实测为准**
- Jetson 民用款（T4000/T5000）与车规 DRIVE Thor 同宗但产品定义不同，不要画等号

## 闲鱼怎么搜：搜料号和黑话，不搜"AI算力板"

卖家往往不知道手里是什么，标题都是"汽车拆车件"，直搜"AI 算力板"搜不到货：

**料号直搜**（最容易出漏）：
`TA990` / `TA990SA` / `TA1080` / `TA1080SA` / `TA1080 ENG` / `TA1090` / `TA1090SA` / `CUSTSAMPLE`

**行业黑话直搜**：
- 智己 周视视觉控制开发板（已验证案例：1199 元淘到 Orin-X 32G 工程板）
- 周视视觉控制器 / 视觉控制开发板 / 智驾控制器 工程板 / 智驾电脑 工程样机
- 域控测试板 / 汽车计算板 / 汽车 AI 控制器 / MPD 智驾控制器（上汽系 Orin-X 域控）

**车型+部件搜**（等事故车/拆车件流入）：仰望 / 理想 L9·L8 / 小米 SU7 / 小鹏 G9 / 蔚来 ET7 / 极氪 9X / 领克 900·08·10 等"智驾电脑/控制器"；"水冷车机/水冷控制器"是高算力域控的强信号。

**工程样机标记**（优先级最高）：`ENG` / `ES` / `QS` / `DV` / `EVT` / `DVT` / `SAMPLE` / `CUSTSAMPLE` / 工程样机 / 测试件——工程样机通常没有量产件的 Secure Boot 限制。

## 车型 ↔ 域控芯片速查

| 车型 | 域控芯片 | 内存 | 备注 |
|---|---|---|---|
| 小米 SU7 Max（2024~25 款） | 2× Orin-X (TA990SA-A1) | 64GB 总 | 水冷，副驾手套箱下方 |
| 小米 YU7 / 2026 款 SU7 | DRIVE AGX Thor (Thor-U) | — | 700 TOPS，全系标配 |
| 比亚迪仰望 | 2× Orin-X (TA990SA-A1) | 64GB LPDDR5 | +128GB UFS×2 + 32GB eMMC×2 |
| 蔚来 ET7 | 2× Orin-X | — | 换电站也用 4× Orin-X |
| 小鹏 G9 | 2× Orin-X (TA990SA-CS-A1) | — | |
| 智己 L7 / LS6 | Orin-X | 32GB | 周视视觉控制器工程板已验证可跑 Linux |
| 零跑 C10 / 极氪 007 | Orin-X 单/双 | — | |
| 理想 L6~L9（2025+ AD Max） | Thor-U ×1 | — | 由双 Orin 升级而来 |
| 理想 i6 / i8 | Thor-U ×1 | — | 官网明示 |
| 极氪 9X H7 / H9 | Thor-U ×1 / ×2 | — | H9 双 Thor-U，>1400 TOPS |
| 领克 900 H7 / 08 EM-P / 10 EM-P | Thor(-U) | — | |
| Robotaxi 领域 | Thor-X（如联想 AD1 双 Thor-X 2000T） | — | 工程样机值得盯 |

## 问卖家的 5 个关键问题

1. 能不能通电进系统？可以 SSH 吗？——能，价值翻倍；答"不知道，拆车的"就当盲盒砍价
2. 量产拆车件还是工程/测试样机？——工程样机（ENG/ES/QS/SAMPLE）没有 Secure Boot 困扰，优先
3. 芯片丝印能拍特写吗？——找 TA990 / TA1080 / TA1090 / NVIDIA 字样
4. 有没有泡水/事故？来源是整车拆件还是试验车？
5. 内存颗粒几颗、什么型号？——可反推总容量（如 8 颗美光 MT62F1G64 = 64GB）

## 到手验证命令

```bash
cat /proc/device-tree/model
cat /proc/device-tree/compatible
uname -a
nvidia-smi   # 或 tegrastats
lspci
```

对照：`TA990` 家族 → Orin-X（Tegra Orin），`TA1090` → Thor-X。

## 量产件 vs 工程样机：系统能不能进去是两回事

- **量产车规域控**：DRIVE OS（Hypervisor 上跑 QNX 或 Linux）+ Secure Boot + OEM 签名 + 加密固件 + 安全 MCU——"显卡即插即用"的期待不成立，且绕过安全机制涉及法律风险（文章有专节法律声明，此处不展开）
- **流出的工程样机**：本来就是 Ubuntu（社区实测原话："一个 ubuntu 小主机"），点亮、装环境、跑大模型不碰安全启动那一层
- **Docker 不用担心**：DRIVE OS Linux 6 起 Container Runtime 烧录进根文件系统；Jetson（JetPack）原生带 Docker + `nvidia-container-runtime`。卡人的不是 Docker，是能不能先拿到系统访问权限

## 实测数据：跑千问系模型

### Orin-X 32G 拆车域控

| 硬件 | 模型 | 实测结果 |
|---|---|---|
| 闲鱼拆机 Orin-X 32G | Qwen 35B MoE | 点亮进 Linux + root 成功；有效带宽实测仅 155~163 GB/s（标称 204.8） |
| 同上 | Qwen 27B 稠密 INT4 | 单并发 8~11 tokens/s；2000 字代码上下文首字延迟约 3 秒 |
| Jetson AGX Orin 64G（开发者论坛） | Qwen3.5-35B-A3B AWQ-4bit | vLLM 跑通，但间歇性崩溃，需物理断电重启 |

**结论：Orin-X 的墙是带宽墙，不是算力墙**——实际有效带宽只有标称的 75%~80%，稠密 27B 级体验一般；MoE（激活参数小）体验明显好于同标称参数的稠密模型。

### Jetson AGX Thor（T5000，128G）官方/媒体实测

| 模型 | Thor 实测 | 对比 AGX Orin |
|---|---|---|
| Qwen3.5-35B-A3B（MoE） | 35 tokens/s | — |
| Qwen3-30B-A3B | 226.4 tok/s | 76.7（约 2.95×） |
| Qwen3-32B（稠密） | 79.1 tok/s | 16.8（约 4.7×） |
| DeepSeek-R1-Distill-Qwen-32B | 82.6 tok/s | 17.0（约 4.87×） |
| Qwen2.5-VL-3B / 7B | 356.9 / 252 tok/s | 216 / 154 |
| Qwen3-8B FP8 | 298 tok/s 吞吐，TTFT 23ms | 需 MAXN 130W + `jetson_clocks` 锁频 |

调优要点：`nvpmodel` MAXN（130W）+ `jetson_clocks` 锁频后，Qwen3-8B 单 token 耗时 68~124ms → 稳定 38~42ms（约 −43%）；长时负载核心温度压 85℃ 以内，散热跟不上性能会掉。

### 可行性边界

- Thor T5000 128G **装不下千问旗舰**：Qwen3.8-Max 是 2.4T 参数级超大模型，别被版本号误导；128GB 远不够
- Thor 舒适区：Qwen3-32B 稠密 / Qwen3.5-35B-A3B 这类 MoE
- Orin-X 32G：MoE 远好于同参数稠密（带宽墙），别指望流畅跑 30B+ 稠密

## 价格行情与炒作脉络（2026-09）

行情：Orin-X 32G 拆机板约 **1300~1500 元**（有实锤成交+点亮案例）；智己周视视觉开发板约 **1199 元**（已验证）；Thor-U/Thor-X 域控约 **4000 元**（传闻价，波动大）。

脉络：2022 圈内小范围流通（Orin 单芯片套件 5000 元）→ 2024~2025 B 站 AGX Orin 跑 70B/QwQ-32B 视频（主角仍是 Jetson 开发板）→ 2025 下半年存储超级周期启动 → 2026 年初内存 18 个月涨约 331%、显卡缺货 → 2026-07-21 NVIDIA 官方涨价 57%~101% → 2026-08 智己 1199 元案例流传 → 2026-09-05 主流媒体报道 Orin-X 32G 1500 元，进入大众视野。

## 冷静提示

- 带宽是 Orin-X 硬伤（有效带宽仅标称 75%~80%），稠密大模型别抱期待
- 拆车板多来自事故车/泡水车，暗病自担，确认收货后基本无法退货
- 原车水冷拆掉后要自解决散热：CNC 均热板+堆风扇 / 桶装水+透明水管简易循环（社区有实例）/ 电竞主机水冷（注意接口与流量匹配）
- 纯推理性价比，同价位 AMD MI50 32GB（1TB/s HBM2）能快约 5 倍——车载算力板的价值在**统一内存大 + 低功耗 + 折腾乐趣**的组合，不在 raw 速度
- 优先找工程样机；量产件可能有 Secure Boot + OEM Key + 加密固件 + 安全 MCU
- ⚠️ 法律边界（原文有专节）：拆解/解锁量产域控处于法律灰色地带，破解/转售风险加重，可能触及破坏计算机信息系统、非法获取数据等罪名及知识产权问题；只研究来源合法的硬件，不出售改装成品

## 参考链接（原文引用）

- [NVIDIA DRIVE AGX Orin 平台文档](https://developer.nvidia.com/drive/agx)
- [NVIDIA Jetson Thor 产品页](https://www.nvidia.com/en-sg/autonomous-machines/embedded-systems/jetson-thor/)
- [NVIDIA 上调 Jetson 价格报道（cnx-software）](https://www.cnx-software.com/2026/07/22/nvidia-increases-the-price-of-jetson-modules-and-devkits-by-up-to-101/)
- [DRIVE OS 平台软件栈文档](https://developer.nvidia.com/docs/drive/drive-os/archives/6.0.4/linux/sdk/common/topics/archi/PlatformSoftwareStacks1.html)
- [DRIVE AGX Orin 直接跑 Docker（NVIDIA 官方博客）](https://developer.nvidia.com/zh-cn/blog/running-docker-containers-directly-on-nvidia-drive-agx-orin/)
- [linux.do：TA990SA-A1 跑 Qwen 实测帖](https://linux.do/t/topic/2705285)
- [linux.do：周视视觉控制器讨论](https://linux.do/t/topic/2760383/14)
- [linux.do：Orin-X 开发板简易水冷实测](https://linux.do/t/topic/2841665)

---

**本文与实测的关系**：本文数字均转引自来源文章（其汇总自社区实测帖与官方文档），本仓库未独立复现；本仓库自己的 Thor 实测数据见 [README 关键实测结果](../../README.md) 与 [community-field-results.md](community-field-results.md)。
