# vLLM 0.32 / DRIVE Thor SM101 优化验证

本文与仓库原有 llama.cpp 历史结果分开维护。本轮仅在测试节点完成验证；不代表生产节点已升级。

## 范围与方法

vLLM 0.32.0 源码 5df63a7203cdb9328d824a27b682e77c7b9928be；PyTorch2.13、Python3.12、CUDA12.8、SM101；同一份27B混合NVFP4/FP8检查点、BF16 KV、上下文262144、chunk2048、MTP4起步、默认temperature0.3/medium。

性能输入为8192或102400 tokens、固定512输出；每次唯一cache_salt保证冷前缀，保留流式usage。prefill=输入token/TTFT，是端到端有效速度；decode=(输出token-首chunk token)/(末chunk时间-首chunk时间)。包含prefill的批次输出吞吐单列，不能用作纯decode。

## 已验证的组成

- 本机SM101原生FA2扩展，作为prefill和不匹配场景后备。
- 已有Triton unified_attention用于单序列短验证/草稿。严格限制24/4头、head256、BF16、原H6页长432/816、混合模型H8页长432/448；原H6/H8多序列回退FA2；另有严格限定页长432、批量1..4、max_q1..5的批量候选，FP32参考13/13、CPU边界28/28通过（包含4×64K和2×100K）。
- 受限CUDA Graph，捕获尺寸按MTP与并发容量计算；关闭通用JIT warmup，但保留模型专用warmup和实际capture。
- 原混合检查点的dense Qwen3Next兼容：num_experts=0、非mRoPE、丢弃残留视觉权重键。只在独立插件生效，不覆盖框架模型源文件。

8-token短路由CPU边界30/30、GPU数值26/26（432/448各13项，含100K及7/8-token）通过，MTP4/5/6/7已完成固定负载验证，单请求与MTP表见[测量记录](MEASUREMENTS.md)，最终默认并发2与并发4容量档已验证。不能把SM120的5090内核直接放到SM101上。

## 二进制下载

引擎二进制与本轮已验证扩展发布在 ModelScope：<https://www.modelscope.cn/models/navyyang/thor05-vllm032-sm101-deploy>（commit `955d66240aa0a6b8757d8b8665d8b0c556f631ba`）。包含 vLLM 0.32.0 aarch64 wheel、SM101 原生 FA2 扩展（SHA256 见仓内 TESTED-OVERLAY-MANIFEST.json）、tested-site overlay、合格代码与文档包、许可证及全件 SHA256SUMS。下载后先 `sha256sum -c SHA256SUMS` 再使用；wheel 名称中的 cu120 是构建元数据遗留，本轮工具链为 CUDA 12.8，不得据此宣称 CUDA 12.0 运行验证。

## 持久化与边界

板端长期文件、环境、模型、缓存、日志只能放在 /zeekr_data、/zeekr_map、/zeekr_hdmap；/home和/tmp在断电重启后会丢失。实验不自动重启、断电或修改驱动；GPU错误、持续D态或大页未归还时停线取证。

脚本来自本轮验证目录，复现布局为 /zeekr_data/ai_workspace/vllm032-clean/qualification-20261021，运行环境env/runtime-lib/toolchain放在其上一级。模型另行提供并校验；不随仓库分发。模型目录缺少VERIFIED-MANIFEST.json或有INCOMPLETE-DO-NOT-LOAD时启动器拒绝加载。测试端口仅loopback8081。

生产部署应由负责部署的维护人员先读取并备份当前已修复的服务、恢复钩子和配置，保留原8080端口，再整合参数；部署处需保存启动方式、依赖、数据路径、质量验收与回退步骤。不能用旧事故前的快照覆盖已修复现状。

## 质量

短题是既有150题中的平衡16题子集，并非完整150题准确率。另有102400输入、5字段中间/首尾事实及规则判分。微内核数值通过不能替代实际HTTP服务质量。部署后仍需实际端口health和目标场景请求同时通过。

源码及复现工具见 ../../scripts/vllm032-thor；第三方许可证见 third-party。单请求、MTP和并发阶段结果见[测量记录](MEASUREMENTS.md)；最终默认并发2与并发4容量档已验证，未作生产升级。


## 本机无GPU夹具

在scripts/vllm032-thor目录运行python3 selftest-launch-fullgraph.py、selftest-routing-batch.py、selftest-quality-judge.py、quality-tools.py --selftest、accept-long.py --selftest、accept-vehicle.py --selftest与bench-compare.py --selftest。GPU微核验证脚本需实际SM101及完整私有运行环境；不能在其他节点上默认运行。

默认seq2、可配置上限4，4档并非动态扩容。生产端口/模型ID/自启恢复机制由部署维护人员保持既有契约，并在实际服务端口完成质量验收。持久化目录约束同样适用于部署文档和JIT缓存。

最终默认MTP4/seq2：8K prefill1441.73、decode28.04 tok/s；100K prefill735.75、decode15.03 tok/s。实际服务health/车辆/工具/短题16/16/100K五字段5/5通过；并发4短题16/16，4×64K完成且无抢占。精确TTFT、批次吞吐、延迟限制和MTP对照见测量记录。
