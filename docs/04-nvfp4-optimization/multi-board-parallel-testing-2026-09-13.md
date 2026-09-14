# Thor 多板并行测试准备（2026-09-13）

> Summary: Readiness snapshot, deployment checklist, and parallel work-split for adding two more DRIVE Thor boards to the optimization test fleet (first board = frozen baseline anchor).
>
> 背景：作者 09-13 通知当天可能上线 板3、板4 两台新主机，后续优化测试
> 要在新机器上并行跑以提速。本文档 = 当前状态快照 + 新板部署清单 + 并行分工方案。
> 接手/部署前必读。

## 1. 当前状态快照（首板 板1，11:40 实测）

- **板上运行中**：`llama-server-ai-assistant-spectime`（10:36 启动，端口 8080，K12 p0.5）
  - 注意：spectime 是带 speculative 计时插桩的**测量变体**，不是生产 binary
  - 生产 binary = `llama-server-ai-assistant-perj`（SHA256 3dafb508…），板上同在
- **AI助手活跃中**：11:26 刚构建 b3-correctness2 / b3-correctness-fixed / b3-microbench7
  （x86 主机 `~/work/thor-ai-assistant/` 同步有产物）——**动板前先看板上有没有新进程/新 log**
- 板健康：GPU 54.6°C、tj 56.4°C、load ~5、/brand_data 余 57G、HugePages 23552（server 占用中）
- 最新 128K 配对（spectime3，11:13）：decode **19.64 t/s**（与 perj 19.61 一致，插桩无性能偏差）

> `/brand_data/` 为板载数据分区挂载点在本笔记中的**代称**。DriveOS 板上该分区（约 105G，板载 vblkdev，与只读根分区独立）的原路径名含车辆品牌字样，为保持品牌中立统一写作 `/brand_data/`。读者在自己板卡上执行 `ls /` 即可看到真实分区名。

## 2. 项目主线状态（详见 [takeover-2026-09-13.md](takeover-2026-09-13.md)）

| 口径 | 基线 | 当前（perj K12 p0.5） | 目标 |
|---|---|---|---|
| 2K decode | 25.72 | 31.02 t/s | —（不计入达标） |
| 128K decode | 16.88 | **19.61 t/s** | **≥30 t/s（作者 09-13 确认的正式目标）** |

- 生产配置：RadixArk-F8attn-v2.gguf（SHA256 d15dac91…）+ f16 KV + -fa on + -c 131072
  + GGML_CUDA_GRAPH_OPT=1 GGML_MMVQ_MAX=2 + MTP K12 p-min 0.5
- 下一步杠杆（按优先级，见 takeover 文档"给下一会话的路线图"）：
  0. R30/R32 账本重测（perj 后旧账本过期）——**AI助手今早已做**，结论在
     `~/work/thor-ai-assistant/ANALYSIS-2026-09-13-openai-rerun.md`
     （128K 尾段：FLASH_ATTN ~28-31%、NVFP4 MMVQ ~41%、F8 GEMM ~12-14%）
  1. NVFP4 LUT 算术化解码（standalone 已测 LUT 成本 17%，microbench7 在迭代）
  2. draft 链开销（12 次串行 draft forward ≈ 50-60ms/step）
  3. 小算子 ~17%（GDN 门控 launch 主导）
  4. 多轮配对提升置信度（当前各口径单 trial）

## 3. 新板（板3/04）部署清单

按 `thor-deploy-checklist` 顺序（前两步是首板踩坑教训，**必做**）：

1. **第 0 步**：三账号免密 sudo（`user` 等，/etc/sudoers.d/99-nopasswd）
2. **第 1 步**：USB 网卡禁 autosuspend（RTL8153 0bda:8153 + Genesys hub 05e3，udev 规则持久化）
   —— 不做 = 掉线数小时"假死"，首板实锤
3. **推理栈**：x86 主机 `<workspace>/thor-driveos/deploy/thor-llama-official/README.md`（官方栈 1.1GB 部署包，每台 ~5 分钟）
4. **HugePages**：扩到 23552 页（46G 池，参考 [05-system-tuning/hugepage-pool.md](../05-system-tuning/hugepage-pool.md)）
5. **资产落板**（从 x86 主机局域网 scp，板无外网）：
   - 模型：`RadixArk-F8attn-v2.gguf`（19.57 GiB，SHA256 d15dac91…，落板后核对）
   - binary：`llama-server-ai-assistant-perj`（生产）+ `b3-correctness` / `b3-correctness2`（门禁）
     —— 位置：首板 `/brand_data/ai_workspace/ai-assistant/` 或 x86 主机 `~/work/thor-ai-assistant/`
   - bench：`bench.py`（确定性配对，/tokenize 精确计数构造 prompt）
   - 启动脚本：`start-server.sh` / `stop-ai-assistant-server.sh`（板上脚本文件承载，防 pkill 连坐 ssh）
6. **落板验证**：b3-correctness（column_failures=0）+ b3-correctness2（cpu_failures=0）
   + 2K 配对 bench 与首板数字对齐（2K 应 ~31 t/s，偏差 >3% 先查板差异再跑长测）
7. **温度监控**：红线 95°C（作者 09-11 定），tegrastats 采样；108°C 硬件自动关机

## 4. 并行分工方案（三板）

原则：**首板 = 基准锚点不动**（所有配对对照的基准数据都在它上面产生），
新板跑实验组，避免基准漂移。

| 板 | 角色 | 跑什么 |
|---|---|---|
| 板1（首板） | 基准锚点 + AI助手主力 | 保持生产配置；kernel 改动后的**基准侧**对照 |
| 板3 | 实验组 A | 零编译类：MTP 参数邻域（K11/K13 p0.4-0.6）、MMVQ_MAX 复扫、多 trial 置信度 |
| 板4 | 实验组 B | kernel 类：LUT 算术化解码、draft 链优化、小算子融合的新 binary 验证 |

并行纪律：
- 每板独立端口/独立 log 前缀（`<板名>-<实验>-<trial>.log`），evidence 落
  `~/work/thor-ai-assistant/evidence/<板名>/` 分目录
- 跨板对比前提：同模型 hash + 同 binary hash + 同 trial prompt（bench.py 同 trial 名逐字节相同）
- 新板首次 2K 配对与首板对齐后才允许跑 128K 长测（128K 单次 ~12 分钟，别浪费在坏板上）
- 温度/频率差异要记录（tegrastats 全程采样），跨板 t/s 对比附温度曲线
- 板上写操作纪律不变：杀进程用板上脚本文件、TERM 后 ps 确认全灭 + HugePages_Free 回 23552

## 5. 作者确认（09-13 11:45）

- [x] 新板**已就绪**，可直接使用
- [x] 账号沿用 `user` 等既有账号；IP 以作者实际通知为准
- [x] 分工按本文档第 4 节执行
- [ ] 散热水冷方案待板子到手后规划整理（影响温度/持续负载安排，128K 长测排期等水冷定案）

## 6. 部署包（已备齐，09-13 11:45）

x86 主机 `<workspace>/thor-driveos/deploy/board3-4-bundle/`：
- `deploy-board3-4.sh <板IP> <board3|board4>` — 一键部署（连通性→USB禁休眠→HugePages→传包→传模型+SHA校验→门禁→启动）
- 生产 binary 三件套（hash 已核对与首板一致：3dafb508…/1cb7ea8…/2c1ee40…）+ 启动/停止脚本 + bench.py
- 模型源：x86 主机 `<workspace>/thor-driveos/models/RadixArk-F8attn-v2.gguf`（19.57 GiB，SHA256 d15dac91…）
- 板子通电联网后：`bash deploy-board3-4.sh <板IP> board3` 即可，全程 ~10 分钟（模型传输占大头）

---

[整理者注] 本文档由工作笔记 板3-4-PARALLEL-TESTING-READINESS.md 脱敏改写：
内网 IP 保留（板端 192.168.1.101、同段 192.168.1.0/24，非敏感信息）；账号名统一为 `user`；
人名与 AI 助手名已中性化（作者/AI助手）；板载分区路径改 `/brand_data/` 代称（见 §1 说明）；
x86 主机本地路径改 `~/work/` 与 `<workspace>` 占位；"解锁"类表述已删除。
技术数据（性能数字、hash、参数、命令）100% 保留。
