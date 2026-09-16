# Two install paths for Hermes Agent on a Thor board: comparison and decision rule

**中文标题：Thor 板上安装 Hermes Agent 的两条路径（板端镜像自举 / 离线轮子）对照与选择规则**

> 日期：2026-09-16 · 平台见 `README.md` · 两台板各自独立安装后对照

## 一、为什么需要两条路

板子是精简系统，**官方 shell 安装脚本在本平台直接跑不了**：它先 `curl` 引导包管理器、再 `git clone` 仓库，
而板上**既没有 `curl` 也没有 `git`**（也没有编译器）。所以安装必须换路径，而"能不能直连包索引"决定了走哪条。

## 二、判定条件（先探一次，再决定路径）

```bash
# 板端一次性探测（先修 DNS，否则必然全失败——见 hard-constraints-*.md 第 4 条）
python3 - <<'PY'
import urllib.request, time
for name, url in [("mirror-A", "https://pypi.tuna.tsinghua.edu.cn/simple/"),
                  ("mirror-B", "https://pypi.mirrors.ustc.edu.cn/simple/"),
                  ("pypi", "https://pypi.org/simple/")]:
    t = time.time()
    try:
        r = urllib.request.urlopen(url, timeout=10)
        print(f"{name}: OK {r.status} {time.time()-t:.2f}s")
    except Exception as e:
        print(f"{name}: FAIL {type(e).__name__} {time.time()-t:.1f}s")
PY
```
- 任一镜像可通 ⇒ 走 **路径 A（板端自举）**，最省事。
- 全部不通（隔离网/无出网）⇒ 走 **路径 B（离线轮子）**。

> 实测记录：本平台修好 DNS 后，两台板的镜像与官方源均可达（示例：0.19–0.24 s 量级返回 200）。
> 也就是说路径 B 在这两台板上其实是**绕远路**；它的价值只在"完全无出网的板子"这一场景。

## 三、路径 A：板端镜像自举（board4 采用）

要点：在持久分区建 venv，用板载解释器 + 包索引直接在板上装，**不需要第二台机器、不需要编译器、不需要传轮子**。

```bash
P=/brand_data/hermes
python3 -m venv "$P/venv"                      # 板载 python3（本平台 3.12）需带 venv 模块
"$P/venv/bin/pip" install -U pip wheel         # 自举 pip（若板端 pip 缺失，取 pip wheel 自举）
"$P/venv/bin/pip" install -i <镜像>/simple hermes-agent
```
- 前提：板载 `python3` 自带 `venv` 与 `ensurepip`（或能取到 pip 的 wheel 自举）。
- ⚠️ **板上没有 `apt`**（实测 `apt-get: command not found`）⇒ "用 apt 装 `python3-venv`/`pip`"这条路**不存在**，别写进任何指南。
- ⚠️ **pip 自举的坑（board4 实测）**：镜像索引里的 wheel 列表若按**字符串**排序，会选到远古版本
  （实测选到 `pip 9.0.3`——连自己都装不上）；必须按**版本号数值**排序，改用 `26.2.1` 后一次通过。
- 优点：一条路走完，无跨机依赖；后续升级/加包同样是板上直接做。
- 代价：依赖板端出网能力；镜像不可达时无路可走（需回退路径 B）。

## 四、路径 B：离线轮子（board3 采用）

思路：在**有出网的 x86 主机**上按目标平台解析并下载全部依赖，再整目录传到板上离线安装。**完全不需要板上出网**。

```bash
# ① x86 主机（有网）：按目标平台拉 aarch64 轮子
python3 -m venv /tmp/dlvenv && /tmp/dlvenv/bin/pip install -U pip
/tmp/dlvenv/bin/pip download --only-binary :all: \
  --platform manylinux2014_aarch64 --python-version 3.12 --implementation cp \
  --dest ./wheels hermes-agent uv
# ② 传板（示例路径）
scp -r ./wheels user@<board>:/brand_data/hermes/<pkg-cache>/
# ③ 板端：用静态包管理器（uv）建 venv 并离线装
uv venv --python /usr/bin/python3.12 --no-managed-python /brand_data/hermes/venv
uv pip install --no-index --find-links /brand_data/hermes/<pkg-cache> hermes-agent
```
实测规模：**61 个轮子 / 约 57 MB**，全部命中预编译轮子 ⇒ 板上不需要编译器。
- 优点：与板端网络完全解耦；轮子目录留在持久分区即成"离线重装源"。
- 代价：多一台机器、多一跳传输；`--only-binary :all:` 若有依赖只有源码包，会在此步直接暴露（需提前发现）。
- 包管理器小技巧：若取 `uv` 的 GitHub release 太慢，可**直接从包索引的 `uv` wheel 里解出二进制**
  （wheel 内含静态 aarch64 二进制），无需额外工具链。

## 五、共同收尾（两条路一致）

1. 在持久分区写 `env.sh`（导出 `HERMES_HOME`、`PATH`）；**不要试图往 `/usr/local/bin` 放 shim**
   （只读根，实测加 `sudo` 也写不进去，见 `hard-constraints-*.md` 第 3 条）。
2. 用 `/etc/profile.d/99-hermes.sh` 让登录 shell 拿到环境（该文件在 overlay 上，由恢复脚本重建）。
3. 配置模型 provider 与默认模型（**key 只落 `HERMES_HOME/.env`，权限 600**），随后跑一次真实问答并记录延迟。
4. 把安装步骤写成**幂等脚本**留在持久分区（换板/重装/离线恢复都要用）。

## 六、结论（本轮的择优口径）

- **默认走路径 A**（更短、更少移动件）；**路径 B 作为无出网兜底**，并在文档里保留完整命令。
- 两条路都必须在**修好 DNS 之后**再执行——否则现象是"所有源都超时"，很容易被误判成"镜像被封"。

## 七、board4 补充（已经 board4 本人实测校验）

- 板卡与系统：DRIVE Thor（p3960 / Tegra264），DriveOS 7.0.3 / Ubuntu 24.04.1 / 内核 6.1.119-rt45，12 核 58 G。
- 安装：板端 `python3` 与 `venv` 模块在位、无 `apt`，镜像可达 ⇒ 采用路径 A（板端自建 venv + 取 pip wheel 自举，
  见上方 pip 版本排序坑）。
- 位置：`/brand_data/ai_workspace/hermes/`（venv 与 `HERMES_HOME` 都在这一个目录里）。
- 自启：宿主侧守护（挂点 B）——不碰板端共用文件，多一跳。
- 模型：**默认 = 外网 DeepSeek**（管理员指定），辅助模型 = 内网 `New API` 的 code 别名，
  回落顺序 = `New API/code` → `交接主机` 的 `qwen3.8-27b`。
- **端到端已跑通（2026-09-16）**：`hermes chat` 非交互两轮成功（首轮 28.4 s 含初始化、次轮 3.9 s），
  `hermes status` 正确识别默认模型走外网 DeepSeek，`errors.log` 为空。

[整理者注] 路径 A 一节依据 board4 书面自述整理，已经 board4 本人实测校验；路径 B 一节的命令与规模数字来自 board3 本机实测记录。
凭据、账号、内网访问方式已删除；数据分区与主机代号按仓库既有映射中性化。
