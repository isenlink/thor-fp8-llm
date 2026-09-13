# 板1 GPU 通道死锁事故全记录与预防规则

作者：AI助手 · 2026-09-13 · 状态：**已定论，作为后续一切 GPU 实验的强制约束**

## 一、三次事故对照表

| | #1 9/12 13:07 | #2 9/13 ~11:28 | #3 9/13 14:56 |
|---|---|---|---|
| 触发进程 | b3-microbench4（+b3-correctness 同道崩） | B4 wide-M kernel v1（b3 系列） | b3-correctness2-i64o2 |
| 改动类型 | **新 kernel**（手写 NVFP4 vec_dot 微基准） | **新 kernel**（B4 wide dp4a） | **仅 config 常量**（mmq-config-ampere.cuh：I=64 + occupancy=2） |
| 根因 | 自定义 block_q8_1 结构体 d 字段按 2B 算（实为 4B ggml_half2）→ qs 落到 offset 2 非 4B 对齐 → `(const int*)qs` 非对齐强转 → GPU fault | block_q8_1 的 qs 字段用 16B 向量加载（int4），但 qs 仅 4B 对齐 → GR fault（值守方从串口日志定位，协作邮局 id=40） | kernel hang（D 态卡 nvgpu 通道等待）+ error notifier 13；同一构建族 k512/k1024/ao2/512t/o2 全部正常，唯独 (I=64, occ=2) 组合挂 |
| 内核签名 | error notifier 13 → irq/296-s-nvgpu D 态 → hung task 120→604s → RCU stall | 同左 | 同左（14:56:59 监控捕获 notifier） |
| 恶化窗口 | ~25 分钟 | ~25 分钟（11:28 发作，11:55 失联） | ~20 分钟（14:56 发作，~15:15 SSH 死） |
| 恢复 | 物理断电 ×2 次（~2 小时） | 物理断电 | 物理断电 |
| 温度 | 正常 52-60°C | 正常 | 正常 |

证据：值守方《板1_GPU死锁_事实单_2026-09-13.md》（串口原始日志）；handoff-2026-09-12-b3.md §教训#2、§结尾（microbench4 首跑崩溃）；ANALYSIS-2026-09-13-b4-wide.md「傍晚 2」节。

## 二、共同模式（比单个根因更重要）

1. **三次全部由 kernel 级代码改动触发，与负载大小/时长/温度无关。**
   微基准跑几秒照样把板子打死。"负载小所以安全"是错的。
2. **触发分两类**：
   - A 类（#1 #2）：**量化结构体字段的非对齐/越界访存**。特征：手写 kernel 里对 block_* 结构体做强转/向量加载，对齐假设错误。
   - B 类（#3）：**MMQ config 常量组合**（I/occupancy/K_vram/线程数）也能打挂通道。"只改常量、用上游代码"不等于安全——occupancy 改变共享内存/屏障压力组合，后果不可预测。
3. **致命放大器是驱动，不是 kernel**：桌面 NVIDIA 上 kernel fault 杀死进程就完事；Thor 的 nvgpu 对通道错误无自愈——fault/hang → 中断线程 D 态 → RCU stall → 整板死锁，**只能物理断电 ≥10 秒**（软复位 tegrareset 无效、板内 reboot 卡死）。
4. **error notifier 13 = 死亡判决书**：一旦出现即进入 ~25 分钟倒计时，任何远端操作都救不回。正确动作是立即停止一切 GPU 操作、通知现场安排断电，不要浪费时间杀进程/软复位。

## 三、预防规则（强制）

### 实验风险分级
| 级别 | 类型 | 例子 | 处置 |
|---|---|---|---|
| L0 | 纯 host 侧 / 参数扫描 / 模型文件改动（走生产已验证的 kernel 路径） | K/p-min 扫描、GGUF 格式改动（F8 转换走既有 cuBLASLt 路径）、env 开关默认值不变的行为 | 正常流程 |
| L1 | 复用**已验证算子**的新图组合 | vocab crop（weight view + concat + fill，全是生产算子） | 2K 短测先行，再 128K |
| L2 | **任何新 kernel、kernel 源码改动、MMQ config 常量** | B4 wide、i64o2 | **条件放行**（见下） |

### L2 条件放行（2026-09-13 晚用户特批框架：不拒绝 kernel 级工作，但必须考虑周全）

全部满足才允许上板，缺一不可：

1. **串口监测确认在线**（值守方侧常驻监听明确回复"监测在线"）；监测不在线不跑 L2
2. **用户逐次特批**：每次 L2 实验前向用户说明改动内容/风险/回退方案，获明确同意
3. **静态审查**：量化结构体字段访问与 ggml-common.h 逐字段核对；禁止非对齐
   `(const int*)`/`(const int4*)` 强转（用 get_int_b4 逐字节模式）；shared 内存/
   occupancy 改动先算 footprint 再上板
4. **一次只改一个变量**；每个 L2 构建先在 x86 侧 ptxas/编译警告审查
5. **上板顺序**：known-good 冒烟 → L2 微基准（<30s，timeout 强约束）→ 门禁正确性
   → 才能进 server/性能测试；任一环节异常立即停止并标记 *.BADHANG
6. **时段选择**：只在用户可现场断电的时段跑 L2（事故恢复需要人到现场）

### 操作规程
1. 跑任何 GPU 负载前在 协作邮局 值守方线程发预告（内容：binary、改动类型、预计时长）。
2. 新 binary/新模型：先 2K（~25s）验证输出+acceptance，再上 128K（~13min）。一次只改一个变量。
3. 不再写任何访问量化块内部字段的新 kernel 代码（B4/B5 已收尾，verify 侧 kernel 优化关闭）。
4. 实验产物分级管理：打挂过的 binary/object 立即改名 *.BADHANG 并记录配置。
5. 止损信号：dmesg 出现 `nvgpu_set_err_notifier` / 进程 D 态卡 nvgpu → 停止一切、通知断电。

## 四、已拉黑的实验产物

- `b3-microbench4`（9/12，A 类对齐 fault）
- B4 wide-M dp4a kernel v1（9/13，A 类对齐 fault；v2 已修但路线已因 dp4a 算力上限放弃）
- `b3-correctness2-i64o2` / `b3-microbench7-i64o2` / `mmq-instance-nvfp4-i64o2.cu.o`（9/13，B 类 config hang，*.BADHANG）
- MMQ 配置空间全部：nthreads 512、occupancy 2、I=64、K_vram 512/1024（后两者性能上也证伪：84.7 < 90.2 GB/s）

## 五、给值守方（值守方）的监测特征（已在用）

- ping 通 + SSH 22 refused = 内核死锁（区别于断电/overlay 回滚）
- 早期信号：`error notifier set to` / `blocked for more than` / `rcu_preempt.*stall`
- 我方义务：跑前预告；出现信号后不再做任何远端操作，直接等断电窗口
