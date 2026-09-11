# Scripts / 实用脚本

> 板端 / 主机上实际用到的辅助脚本。板上没有 nvidia-smi、没有常规诊断工具，
> 这些脚本补上了"跑大模型前确认环境"的缺口。

## 脚本清单

| 脚本 | 用途 | 运行位置 | 依赖 |
|---|---|---|---|
| `uart_probe.py` | 串口探测：自动发现 ACM 设备 → 115200 8N1 → 断言 DTR/RTS → 发回车 → 监听 | 主机（接板串口的机器） | `pyserial` |
| `gpu_pool_check.py` | GPU 大页池 / 统一内存余量检查（板上无 nvidia-smi 的替代） | 板端 | 无（纯标准库） |

## uart_probe.py

**什么时候用**：板子串口静默、不确定接线/口序/是否活的时候。

```bash
pip install pyserial
sudo python3 uart_probe.py [监听秒数, 默认15]
```

**它做什么**：
1. 自动发现 `/dev/ttyACM*` 设备
2. 逐个以 115200 8N1 打开，断言 DTR/RTS
3. 发回车唤醒，监听 N 秒
4. 打印收到的字节数、CD/DSR/RI/CTS 状态、前 500 字节 + hexdump

**关键经验**（详见 [TROUBLESHOOTING.md](../TROUBLESHOOTING.md) A 节）：
- **串口静默 ≠ 接线/硬件问题**——默认量产固件下控制台可能被禁用
- 板载 CH34x 双串口：口1=主 SoC Linux 控制台，口2=MCU/AURIX nvshell
- 断言 DTR/RTS 是唤醒某些控制台的关键（不 assert 可能一直静默）

## gpu_pool_check.py

**什么时候用**：跑大模型**之前**，确认 GPU 池余量够不够。

```bash
python3 gpu_pool_check.py [期望占用 GiB, 默认 20]
# 例：要跑 27B（权重 ~20G + KV），期望 24G
python3 gpu_pool_check.py 24
```

**它做什么**：
1. 读 `/proc/meminfo` 的 HugePages_Total/Free/Hugepagesize
2. 算出 GPU 池总量 / 空闲 / 已用
3. 读 `MemAvailable`（真实余量，比 `free` 可靠）
4. 判定给定期望占用是否放得下

**为什么需要它**：
- 板上**没有 nvidia-smi / nvtop**
- 统一内存架构下，GPU 池 = hugepages，`free` 恒低（被池刚性划走），
  必须看 `MemAvailable` 和池内空闲
- 详见 [docs/05-system-tuning/hugepage-pool.md](../docs/05-system-tuning/hugepage-pool.md)

## 使用注意

- 脚本均为**只读探测**，不修改系统状态，可放心运行
- `uart_probe.py` 需要 root（访问串口设备），`gpu_pool_check.py` 普通用户即可
- 这两个脚本是"环境确认"工具，**不是**部署/推理脚本
  （llama-server 启动命令见 [docs/04-nvfp4-optimization/nvfp4-optimization-log.md](../docs/04-nvfp4-optimization/nvfp4-optimization-log.md)）
