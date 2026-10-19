# 串口控制台被 MCU 报错淹没：从错误归因到取证闭环

> **一句话**：串口控制台每秒涌入几十条 `Invalid TxFramePtr -1`，
> 登录提示符被冲掉、自动化脚本全部失灵。本文记录如何**用数据推倒一连串错误假设**，
> 最终定位到真实成因，并给出不损伤救援通道的工程解法。
>
> 实测环境：DRIVE Thor 域控板（Tegra264），2026-10 实测。

---

## 1. 现象

串口控制台（115200 8N1）持续输出：

```
[timestamp:1940578959781][error][Ccplex]Invalid TxFramePtr -1
[timestamp:1941578381785][error][Ccplex]Invalid TxFramePtr -1
[timestamp:1942578391191][error][Ccplex]Invalid TxFramePtr -1
```

后果：

- `login:` 提示符被淹没，人工登录困难
- 自动化脚本的 `expect("login:")` / `expect("Password:")` 频繁误判
- 读到的输出被切成碎片，无法解析命令结果

---

## 2. 第一步错误：把 burst 当成持续刷屏

**错误认知**："每秒一条，稳定刷屏。"

**怎么被推翻的**：把 3.5 小时的历史日志按分钟统计，发现真实分布是：

```
16:39:05  3次        ← 首次出现
16:53:57  2次        ← 间隔 14 分钟
17:11:38  1次        ← 间隔 18 分钟
17:34:46  1次        ← 间隔 23 分钟
17:39:18  3次
...
18:49 之后才逐渐变密
```

**3.5 小时总共只有 194 条。** 根本不是"每秒一条"。

### 更关键的发现：时间戳是假的

相邻两条的 timestamp 差：

```
1941578381785 - 1940578959781 = 999,422,004
1942578391191 - 1941578381785 = 1,000,009,406
```

**几乎精确相差 10 亿。** 取模 10^9 后：

```
578959781
578381785
578391191   ← 差 9,406 ns
578408303   ← 差 17,112 ns
578419264   ← 差 10,961 ns
```

**余数几乎连续，每次只差几微秒。**

⇒ 这是**纳秒级单调时钟被错当成 epoch 毫秒**打印出来的，
`timestamp / 10^9` 的整数部分是假的，后 9 位才是真值。

**真实结论：这些报错在几十微秒内连发几十条 —— 是一次事件爆发，不是持续刷屏。**

> **教训**：看到 timestamp 先算相邻差值。差值异常规整（恒等于某个大数）= 时间基准错了，
> 基于它的所有"频率"判断全部作废。

---

## 3. 第二步错误：在错误的时间做实验

基于"持续刷屏"的错误认知，设计了实验：
> 启动 FSI 接收端服务 → 观察刷屏是否停止。

**结果：干预前后都是 0 次命中。实验完全空转。**

因为那时候 MCU 处于稳态、根本没在报错，测的是空气。

> **教训**：做 A/B 对照前，**先确认现象此刻确实存在**。
> 基线为 0 的实验证明不了任何因果关系。

---

## 4. 真正的定位方法：找爆发前的前兆

不再盯报错本身，改为**看每一波爆发之前的几十行是什么**：

```bash
# 找出 Ccplex 首次出现的位置，打印它前面 5 行
grep -n 'Ccplex' log | head -1
sed -n '<行号-5>,<行号>p' log
```

结果 —— **每一波的前兆完全相同**：

```
[MCU] MCU Reset Reason: Multiple Reset Reasons found
[MCU] lPltfPwrMgr_EnablePREREG EN_PWR_VDD_POC_12V
[MCU] SafeStateMachine_Srv:2
[MCU] CarConfig_Srv:2
[SoC] [FSI] Compilation date: Feb 26 2026      ← ★ 固件重新启动
[SoC] [FSI] BswM Enter Wakeup
[SoC] [FSI] ***Exited EcuM_StartupTwo()...
[SoC] [FSI] Running -> Ccplex [id:2]
[SoC] [FSI] Com Init done                       ← 通信初始化完成
       ↓ 立刻开始报错
[Ccplex]Invalid TxFramePtr -1
```

**判据**：`Compilation date: ...` 这行在**每一波**都出现。这是 MCU 固件打印自己的
编译日期 —— **只有启动时才会打**。

⇒ **MCU 在反复重启。** 每次重启走完 AUTOSAR 启动流程
（`EcuM_StartupTwo` → `BswM Wakeup` → `Com Init done`），
然后尝试向 SoC 发帧，失败，吐出这一波错误。

### 触发 MCU 重启的上游原因

同一批日志里还看到：

```
[MCU] [122][warn][hwpm]:swc_CCP: Recv TimeOut
```

MCU 侧某通信服务接收超时 → 触发 MCU 复位。

---

## 5. 根因链条

```
MCU 侧 swc_CCP 收不到预期报文（Recv TimeOut）
        ↓
MCU 复位重启（"Multiple Reset Reasons found"）
        ↓
AUTOSAR 启动 → Com Init done
        ↓
尝试向 SoC 发送 FSI 帧
        ↓
SoC 侧 FSI 接收端未运行            ← 见 §6
        ↓
Invalid TxFramePtr -1（几十微秒内连发几十条后放弃）
        ↓
MCU 进入稳态静默 → 串口恢复干净
```

**重要推论**：这解释了为什么"断电重启后串口是干净的" ——
断电重启让 MCU 走完一次完整启动，报一波错后进入稳态。
**它不是开机就刷，而是在 MCU 因故重启时才刷。**

---

## 6. SoC 侧为什么没人接收

查 FSI 通信栈：

```bash
lsmod | grep -E 'fsicom|nvsciipc'
# tegra_fsicom      24576  0      ← ★ 引用计数 0：驱动在，没人用
# nvsciipc          36864  1 nvmap
# ivc_cdev          20480  1 nvsciipc

ls -l /dev/fsicom_client
# cr--------. 1 root nvfsicom_kmd 496, 0    ← 设备节点存在

systemctl is-active   nv_fsicom_daemon      # inactive
systemctl is-enabled  nv_fsicom_daemon      # disabled     ← ★ 出厂就没启用
```

服务依赖关系：

```
nv_fsicom_daemon.service            ← 用户态接收端（disabled）
        ↑ After=
fsi-adapter.service                 （业务层）
        ↑ After= + Requires=
nv_epl_daemon.service               （另一业务守护）
```

**`tegra_fsicom` 驱动已加载但引用计数为 0** —— 这是"用户在、业务不在"的直接证据。

> `disabled` 是 **systemd 的持久配置状态**，不是崩溃。
> 说明出厂设定就不启用它 —— 该平台的业务可能不走这条通道。

---

## 7. 三条处置路线与取舍

| 方案 | 做法 | 风险 | 结论 |
|---|---|---|---|
| **A. 硬件屏蔽 RX 中断** | 写 UART 控制器寄存器屏蔽接收中断 | **切断串口输入 = 毁掉唯一物理救援通道** | ❌ 绝不采用 |
| **B. 读取侧过滤** | 脚本内剥离噪声行再匹配 | 零风险，物理串口仍可用 | ✅ **采用** |
| **C. 关掉噪声源** | 启用 FSI 接收端 / 关 MCU 日志 | 需验证因果；可能连带拉起业务服务 | ⏸ 未验证 |

> **A 方案为什么不能做**：车规板 overlay 丢失时，SSH 不通（网络配置丢了），
> **串口是唯一入口**。为了一条噪声而切断救援通道，是本末倒置。

---

## 8. 方案 B 实现：读取侧过滤

### 四条过滤规则（缺一不可）

```python
NOISE_PATTERNS = [
    # ① 整行噪声：timestamp + error + 内容
    r"\[?timestamp:\d+\]?\[error\]\[Ccplex\]Invalid TxFramePtr -1\r?\n?",
]
# ② 孤立 NUL 字节（帧尾带的）
# ③ UTF-8 替换符 U+FFFD（多字节被截断时的残留）
# ④ 其它周期性噪声（如 ether_get_ethtool_stats: iface not up）
```

```python
def strip_noise(txt: str) -> str:
    for pat in NOISE_PATTERNS:
        txt = re.sub(pat, "", txt)
    txt = txt.replace("\x00", "")      # ② 孤立 NUL
    txt = txt.replace("\ufffd", "")    # ③ 残字节，必须剥，见下
    txt = NOISE_RE_ETHTOOL.sub("", txt)
    return txt
```

### 两个必须注意的实现细节

**① 匹配必须用过滤后的文本**

噪声每秒几十条时，如果 `expect` 用原始缓冲：
- 提示符永远淹没在噪声里，永远匹配不到
- 重试计数被噪声刷爆

**所有 prompt / login / Password 的判定，必须先 `strip_noise()` 再匹配。**

**② 必须剥 `\ufffd`（UTF-8 替换符）**

这是我们实际踩了两次的坑：

```
原始：  user@host:~$ <噪声中间截断>\ufffd
剥噪声后：user@host:~$\ufffd          ← 尾部残留替换符
```

`PROMPT_RE = r"...\$\s*$"` 的 `\s*$` 匹配不到 `\ufffd`，
**提示符明明在却判定失败**。

> 这类残字节来自：噪声行被读到一半时缓冲区截断，多字节字符被切开，
> 解码时产生 U+FFFD。只要噪声在异步写入，就一定会出现。

### ③ 活跃时间戳只在"有实义内容"时刷新

判定串口"是否活着"通常看最近一次收到数据的时间。
但噪声持续涌入会让这个时间戳永远新鲜 —— **串口卡死也判成活的**。

```python
clean = strip_noise(raw)
if clean.strip():                      # 剥完还有内容，才算真的收到东西
    self.last_active = time.time()
```

---

## 9. 完整证据链复盘

| 序号 | 实验/查询 | 结果 | 推论 |
|---|---|---|---|
| 1 | 按分钟统计历史日志 | 3.5 小时仅 194 条，分布稀疏 | 否证"持续刷屏" |
| 2 | 相邻 timestamp 差值 | 恒等于 ~10^9 | 否证"纳秒时间戳"；频率判断作废 |
| 3 | 取模 10^9 | 余数连续，间隔几十微秒 | **确认是 burst** |
| 4 | 干预前后对比（基线=0） | 前后均 0 条 | 实验空转，假设未被检验 |
| 5 | 查每波前兆 | 每波均有 `Compilation date` | **MCU 在重启** |
| 6 | 查 SoC 侧接收端 | 服务 disabled，驱动引用计数 0 | 帧无人接收 |
| 7 | 查 MCU 上游 | `swc_CCP: Recv TimeOut` | 触发 MCU 复位 |

---

## 10. 可复用纪律

1. **timestamp 先算差值** —— 差值恒等于某个大数 = 时间基准错，基于它的频率结论全部作废。
2. **A/B 实验前确认现象存在** —— 基线为 0 的实验不能证伪任何假设。
3. **别盯现象，找前兆** —— 爆发前的几十行往往直接给出触发事件。
4. **`disabled` ≠ 故障** —— 出厂状态和运行时崩溃是两回事，先分清再动手。
5. **驱动引用计数 = 用户态是否在用** —— `lsmod` 第三列是 0 就是"驱动在、业务不在"。
6. **不要为了消噪声切断救援通道** —— 读取侧过滤永远优先于物理屏蔽。
7. **归因要能解释全部观察** —— "MCU 重启导致"能解释"断电后干净""间隔不规则""burst 形态"，
   这比"某服务没开"更贴合数据。

---

## 附：整理者注

- 路径 `/dev/fsicom_client`、服务名 `nv_fsicom_daemon` / `tegra_fsicom` 为平台组件名，保留。
- 所有 IP、SN、主机名、账号已移除。
- 未解部分如实标注：**"启用 FSI 接收端能否消除报错"尚未验证**（因现象非持续存在，
  无法构造有效对照）。已排除的是"启动它会让刷屏消失"这一方向的证据不足，
  而非证明其无关。
