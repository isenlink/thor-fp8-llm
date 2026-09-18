# Deploying Hermes Agent on a DRIVE Thor board (persistent, survives dirty power loss)

**中文标题：在 DRIVE Thor 开发板上部署 Hermes Agent —— 全量落在非易失分区、脏断电可自愈**

> 平台：NVIDIA DRIVE Thor（p3960-0010 / Tegra264 / aarch64），DriveOS 7.0.3，12 核，统一内存 58 GiB
> 参与：两台同型板（下文 `board3` / `board4`）各自独立安装，方法互相对照后择优
> 日期：2026-09-16

## 一、先说结论（四条）

1. **只有板载数据分区能扛脏断电**：`/` 只读且被 hypervisor 层写保护；`/etc`、`/home`、`/media` 位于 overlay 上层，
   掉电后 fsck 判错即被格式化。⇒ Agent 的程序本体、`HERMES_HOME`（配置/凭据/会话/技能）**全部放 `/brand_data/`**，
   overlay 侧只留"可重装薄壳"。
2. **板子是精简系统**：无 `curl` / `git`（官方安装脚本用不了）、无编译器；有 `python3`（本平台 3.12）与 `venv`。
   ⇒ 安装走"板端自举"或"离线轮子"，两条路本文都记（见 `install-paths-*.md`）。
3. **板内无法自启自愈**：启动链全在只读根上，插不进任何 hook。⇒ 自启必须由**板外常电主机经串口**触发，
   恢复动作挂在那条既有链路上（两种挂法见 `persistence-autostart-*.md`）。
4. **两个会让人误判"装坏了"的硬约束**：Agent 要求模型上下文窗口 **≥64K**；本地 `llama-server` 必须带
   **`--cache-ram 512`**（默认 8 GiB 的主机侧 prompt cache 会把常规内存吃光）。详见 `hard-constraints-*.md`。

## 二、本文档集

| 文件 | 内容 |
|---|---|
| `README.md` | 本页：结论、共同前提、目录布局 |
| `install-paths-2026-09-16.md` | 两条安装路径对照（板端镜像自举 / 离线轮子），含判定条件与各自踩坑 |
| `persistence-autostart-2026-09-16.md` | 脏断电存活：资产放哪、overlay 薄壳、两种自启挂点及代价 |
| `verification-2026-09-16.md` | 不等真断电的等价验收法（模拟 overlay 被吞 → 恢复 → 复验） |
| `hard-constraints-2026-09-16.md` | 硬约束与反例（64K 门槛 / `--cache-ram` / 只读根放不了 shim / hosts 补丁） |
| `comparison-2026-09-16.md` | 两台同型板的实现对照 + 择优结论（board4 列已经 board4 本人实测校验） |
| `scripts/hermes-agent/` | 脱敏后的恢复脚本、systemd 单元母本、可达性自检脚本 |

## 三、共同前提（两台板一致，可当部署前检查单）

- 板端有独立可写持久分区（本文统一记作 `/brand_data/`），分配 ~78 GB 量级；
  **该分区根目录属主不是登录用户**，建目录需 `sudo` 后再 `chown` 到 `user`。
- 登录用户有 **免密 sudo**（串口恢复时本就是 root）。
- 根文件系统只读：`touch /usr/local/bin/x` 会直接返回 `Read-only file system`（加 `sudo` 同样失败）。
- 出厂 `/etc/resolv.conf` 指向**平台预置的遗留解析器地址**，该地址落在车载网口的路由上；
  只要默认路由不指向车载内网，它就永远不可用 ⇒ **一切域名解析挂死**（详见 `hard-constraints-*.md` 第 4 条）。
- 无 `nvidia-smi`：看内存用 `/proc/meminfo`（`MemAvailable` / `HugePages_*`），温度看
  `/sys/class/thermal/thermal_zone0/temp`。

## 四、推荐目录布局（板端，全部在持久分区）

```
/brand_data/hermes/            # 或 /brand_data/ai_workspace/hermes/
├── venv/                      # Python venv（用板载 python3，不下载解释器）
├── home/                      # ★ HERMES_HOME：config.yaml / .env / skills / state.db / sessions / logs
├── bin/                       # 启动 wrapper、自检脚本、包管理器（如需）
├── systemd/                   # 单元母本（overlay 侧由恢复脚本重装）
├── <pkg-cache>/               # 离线轮子（仅离线路径需要）
├── env.sh                     # 导出 HERMES_HOME / PATH
└── restore-*.sh               # ★ 幂等恢复脚本（重建 overlay 薄壳 + 起服务 + 复验）
```

overlay 侧（会被格式化，因此**必须可由恢复脚本重建**）：
`/etc/systemd/system/*.service`、`/etc/profile.d/99-*.sh`、`/etc/resolv.conf`。

## 五、最小验收标准

装完必须能同时满足（缺一条都算没装好）：
1. `hermes --version` 与一次真实问答跑通（`hermes chat -q "..."`，模型端与实际延迟都要记）；
2. overlay 薄壳全部由恢复脚本重建（含 DNS、大页池、单元、profile）；
3. 走一次"模拟 overlay 被吞"的等价验证（见 `verification-*.md`），复验服务、DNS、模型端可达、端到端；
4. 板端持久分区里留有**可离线重装**所需的资产（轮子或自举脚本）与文档。

[整理者注] 本文档由两台板各自的实测汇合而成：代号已中性化（板序号一律写作 `board3`/`board4`，
主机代号按仓库既有映射处理），数据分区统一写作 `/brand_data/`，登录账号一律写作 `user`；
删除项仅为凭据与内网访问方式，技术路径、命令、报错原文、数字 100% 保留。
