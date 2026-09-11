# 板端存储与内存占用调查

> One-liner: Where things can actually live on a DRIVE Thor board — the
> read-only root + overlay layout, the 974M /home overlay, and why 21GiB of
> "used" memory is kernel carveout you cannot reclaim.
>
> 日期：2026-09-08（本仓库版：脱敏整理）

## /home 情况

/home 是 974M overlay（与 /etc、/var 共享 rw_overlay 974M，当时 777M 可用）。该 overlay 与根分区是分开的独立 vblkdev30。官方资料说的"/home 可用"指这个 overlay——但注意两个限制：

1. 只有 ~777M 可用，放编译产物（~10MB bin）绰绰有余，放大模型不可能
2. 重要：早期探测记录说"/home 是 overlay 重启可能丢"——但实测它与 /mnt/rw_overlay 是同一设备，重启保留性需验证（可放标记文件重启后查看）

## 内存占用调查结果

- 总内存 58Gi，当时 used 21Gi / free 37Gi
- 用户态进程占用极小：最大单进程 23MB，前 20 名进程加起来 <200MB
- 所以 21GB 的 "used" 大头是**内核保留区（carveout/reserved memory）**：/proc/iomem 显示大量 reserved 区域——这是 DriveOS 启动时固件/设备树划走的（GPU/SATA/CAM/安全岛等 carveout）
- 用户服务可优化空间：车厂组件服务（更新/安全/日志类，单个 6-23MB）加起来几十 MB，停了意义不大
- **关键点**：统一内存中 GPU 可用的只有 20GiB（vs 物理 58GiB），那 38GiB 差额是内核级 carveout，停用户态服务省不出这块——要动 carveout 需要改设备树，风险高，需慎重评估
- 后续我们通过 hugepages 扩容把 GPU 池做到 42GiB（见 [系统调优](../05-system-tuning/)），这是不动设备树的安全路线

## 经验总结

> 在 DriveOS 车规板上做 LLM 部署，第一课就是搞清"哪些内存/存储你真的能用"：
> 物理 58G ≠ GPU 可见 20G；根分区只读 ≠ 全盘只读；overlay 可写 ≠ 断电不丢。
> 全部摸清后再规划模型放哪、编译产物放哪、swap/临时文件放哪。
