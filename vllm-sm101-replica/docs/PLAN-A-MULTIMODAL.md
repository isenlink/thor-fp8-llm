# 方案A 完整复现文档：Qwen3.8-27B 多模态（Qwen3-Next 混合塔 + Qwen3VL 视觉壳）移植到 vLLM 0.11.2

> 2026-09-26 定稿。target-board（p3960, aarch64 sm_101, CUDA 12.8, Python 3.12）实测跑通。
> 本文是**唯一权威复现文档**：将来升级 vLLM 版本、或在新机器/新模型上重做此移植，按本文从零走一遍即可。
> 背景与动机见 `10-ENGINE-VERIFY-STATUS.md`（27B 原始 NVFP4-HF 权重含完整视觉塔 333 张量，
> 但 0.11.2 无 Qwen3-Next-VL 模型类，纯文本手术时被主动剥离 → 本方案补回）。

---

## 一、原理：为什么要"嫁接"，嫁接的是什么

| 组件 | 官方归宿 | 我们的归宿 |
|---|---|---|
| 视觉塔 (ViT + merger, BF16, 333 张量) | Qwen3VL 家族 | **0.11.2 自带 `qwen3_vl.py`，零移植** |
| 文本塔 (64 层：48 GDN 线性注意力 + 16 全注意力，NVFP4) | Qwen3Next / Qwen3.5 (0.30+) | **0.11.2 移植版 `qwen3_next.py`（纯文本手术产物，已验证）** |
| 模型壳 (processor / deepstack / embeds 合并 / KV 管理) | 0.30 `qwen3_5.py` 的 `Qwen3_5ForConditionalGeneration` | **自写 `qwen38_vl.py`：继承 0.11.2 `Qwen3VLForConditionalGeneration`，覆写点见下** |

核心洞察（来自 0.30 源码侦察）：
- 0.30 的 `Qwen3_5ForConditionalGeneration` 就是 `class ...(Qwen3VLForConditionalGeneration, IsHybrid)` ——
  官方自己也是"VL 壳 + hybrid mixin"的组合。我们复刻的是同样的结构，只是壳和塔都用 0.11.2 版本。
- 27B 的 `vision_config.out_hidden_size = 5120 == text hidden` → 视觉 embeds 与文本 embeds **同维度直接合并**，
  无需投影层。deepstack levels=0（无多尺度深栈），进一步简化。

## 二、前置资产清单

### Cross-compile host
- `<host-models>/Qwen3.8-27B-NVFP4-HF/`：原始权重（文本塔 NVFP4 + `model.visual` 333 张量 BF16）
- `qwen38-surgery/weights/model-v4.safetensors`：纯文本手术 v4 权重（唯一合法传输源，23,839,051,704 B）
- `qwen38-surgery/extract_visual.py` → 产物 `qwen38-surgery/weights/visual-tower.safetensors`（921,500,008 B，879MiB）
  - 键名保持 HF 原前缀 `model.visual.`，权重加载时靠 AutoWeightsLoader 的 skip/映射规则对位

### 板上（target-board）
- `<board-model-dir>` = v4 文本权重；`<board-model-dir>` = 视觉塔
- `<board-model-dir>`：**新模型目录**（config.json + tokenizer/processor 文件；不动 p3-models）
- 0.11.2 定制环境：`<board-workspace>`（含全部板上补丁，见 §五）
- gcc 离线包 `<board-workspace>`；`VLLM_CACHE_ROOT=<board-workspace>`

## 三、`vl27b-config.json`（结构改造，成败关键之一）

顶层结构 = Qwen3VL 嵌套式（`model_type: "qwen3_vl"`），内部：
- `text_config` = 纯文本手术版 config 的**扁平字段**（qwen3_next 族：linear_* 头参数、
  full_attention_interval=4、num_experts=0、max_position_embeddings=262144 等）
- `vision_config` = 原始 27B 的 vision_config **逐字段拷贝，但必须删除空键**（见 §五坑3）
- `quantization_config`：
  - 原有 NVFP4 ignore 列表**全部加上 `language_model.` 前缀变体**（权重加载后前缀是 `language_model.model.layers...`）
  - 追加 `"re:model\\.visual\\..*"`（视觉塔 BF16 不进 NVFP4 量化目标）

## 四、`qwen38_vl.py`（模型类，板上 venv `vllm/model_executor/models/` 内）

骨架（约 100 行）：
```python
class Qwen38VLForConditionalGeneration(Qwen3VLForConditionalGeneration, IsHybrid):
    hf_to_vllm_mapper = Qwen3VLForConditionalGeneration.hf_to_vllm_mapper
```

必须覆写/补齐的点（每一个都是实测踩坑后确定，缺一不可）：

1. **`IsHybrid` 继承**（from `vllm.model_executor.models.interfaces`）
   → 否则 `model_config.is_hybrid=False`，vLLM 跳过 mamba/attention 页对齐，
   MambaSpec block_size=None → `check_enough_kv_cache_memory` 直接崩。

2. **`get_mamba_state_shape_from_config` / `get_mamba_state_dtype_from_config` 类方法**
   照 0.30 `qwen3_5.py` 抄，但用 **0.11.2 的 API**：
   - shape: `MambaStateShapeCalculator.gated_delta_net_state_shape(tp, linear_num_key_heads,
     linear_num_value_heads, linear_key_head_dim, linear_value_head_dim, linear_conv_kernel_dim, num_spec)`
     —— 参数从 `vllm_config.model_config.hf_text_config`（**不是 hf_config**，顶层是 VL config 没有这些字段）
   - dtype: 0.11.2 的 `gated_delta_net_state_dtype` **只收 2 参**（model_dtype, mamba_cache_dtype），
     与 0.30 的 3 参不同

3. **构造时序**：`super().__init__` 会按 hf_config.model_type 建标准文本塔（触发 qwen2.py 断言）。
   必须在 super 前 `hf_config.model_type = "qwen3_next"`（或等价 monkey-patch 换塔），
   让文本塔落到移植版 `Qwen3NextForCausalLM`。

4. **文本塔 vllm_config 隔离**：塔内读 `vllm_config.model_config.hf_config` 的地方
   需要拿到 text_config 视角（vocab_size、linear_* 等字段）。手段：构造前把
   `model_config.hf_text_config`/`hf_config` 指向 text_config（注意 ModelConfig 可直接属性赋值）。

5. **forward 适配**：基类无条件给 `language_model` 传 `deepstack_input_embeds` kwarg，
   但 `Qwen3NextModel.forward` 不收 → 包一层 adapter 剥掉该 kwarg。

6. **权重加载**：`AutoWeightsLoader`（0.11.2 无 `skip_submodules` 参数，用
   `skip_prefixes` + `skip_substrs` + `ignore_*`）；加载两份权重（目录内 model.safetensors 的
   文本塔 + visual-tower.safetensors 的 `model.visual.*`）。

## 五、坑清单（按踩中顺序，每条都是真崩点）

1. **`is_list_of` 不在 `vllm.utils`**（0.11.2 已移走）→ ImportError + pydantic inspect 连锁失败。删无用 import。
2. **`visual.merger.linear_fc1` 不在量化 targets** → quant ignore 加 `re:model\.visual\..*`。
3. **`deepstack_visual_indexes: []` 空数组** → 基类 `is not None` 判定启用 deepstack → 空 buffer `IndexError: list index out of range`（qwen3_vl.py:1289）。
   **修法：config 里直接删掉该空键**（27B 本来就没有 deepstack 功能）。这是 0.11.2 上游边界 bug。
4. **`qwen2.py max_window_layers` 断言** → super().__init__ 建错塔。见 §四.3。
5. **prefix caching 断言**（qwen3_next 族禁用）→ 启动参数不加 prefix-caching 相关开关。
6. **量化 ignore 正则前缀失配**：写 `re:model\.layers\.\d+...` 时权重实际前缀已是
   `language_model.model.layers...` → ignore 列表必须两套前缀都留。
7. **`MODELS_CONFIG_MAP` 不含自定义架构**：`model_config.is_hybrid` 的判定链是
   `registry.inspect_model_cls → _model_info`（继承 mixin 决定），不是 config map。所以修法是 §四.1 的 mixin，不是改 map。
8. **KV 页对齐成功标志**：启动日志必须出现
   `Setting attention block size to 400 tokens to ensure that attention page size is >= mamba page size`
   + `Padding mamba page size by 0.25%`。出现即 hybrid 路径正确；没出现=IsHybrid 没生效。

### 5.9–5.12：MTP 接入追加坑（VL 嫁接版特有）

9. **draft 路由不会自动发生**：文本版 `{"method":"mtp","num_speculative_tokens":1}` 不传 draft
   也能跑（draft=target，`hf_config_override` 把 qwen3_next→qwen3_next_mtp）。
   VL 版 target model_type=qwen3_vl 不在 override 表 → 必须**显式传 draft 目录**：
   `{"method":"mtp","num_speculative_tokens":1,"model":"<board-workspace>"}`。
10. **Qwen3NextMTP 读 vocab_size 崩**：draft 构造时 `model_config.hf_config` 是 Qwen3VL 顶层
    （无 vocab_size/linear_*）→ 在 `qwen3_next_mtp.py` 的 `Qwen3NextMTP.__init__` 顶部
    **构造隔离副本**（`dataclasses.replace(vllm_config, model_config=浅拷贝)` + hf_config 指向
    text_config 并设 model_type=qwen3_next）。
11. **绝不能改共享 model_config 的 hf_config**：直接 `model_config.hf_config = tc` 会污染
    target 的 `architectures` property（=hf_config.architectures）→ 后续 multimodal 注册表
    `KeyError: Qwen3NextForCausalLM`（profile_run→_get_mm_dummy_batch）。
    这是最隐蔽的一坑：报错点离肇事点隔了一个阶段。
12. **Mamba+spec 白名单**：`mamba/abstract.py get_kv_cache_spec` 硬编码
    `model_type not in ["qwen3_next"]` 则 raise——target 顶层是 qwen3_vl →
    `NotImplementedError: Mamba with speculative decoding is not supported yet`。
    板上补丁：白名单加 `"qwen3_vl"`（我们实质就是 qwen3_next 塔，放行合理）。
13. （附）**eagle.py:992 image_token_index**：投机壳读 `target_model.config.image_token_index`，
    Qwen3VL 命名是 `image_token_id` → 模型类构造尾部补 `config.image_token_index = config.image_token_id`。

## 六、板上部署与启动（验证过的命令口径）

```bash
# 目录
<board-model-dir>{config.json, tokenizer*, processor*, preprocessor*, chat_template*}
# 权重（物理同源，目录里用相对引用/加载器双份读取）
<board-model-dir>{model.safetensors, visual-tower.safetensors}

# 启动（start-vl27-8996.sh 要点）
venv/bin/vllm serve <board-model-dir> \
  --served-model-name vl27 --port 8996 \
  --max-model-len 32768 \        # 首验口径；生产口径待测
  --max-num-seqs 3 --gpu-memory-utilization 0.94 \
  --no-enable-prefix-caching
# 环境：LD_LIBRARY_PATH/CPATH→gcc-root，VLLM_CACHE_ROOT，PATH 加 binutils
```

启动判据（检查点制，逐项确认才继续）：
1. 无 `ImportError`/量化 target 报错
2. `attention block size ... tokens` + `Padding mamba page size` 两行出现
3. `Available KV cache memory` + `GPU KV cache size` 正常（首验 77,600 tokens @32K 口径）
4. `startup complete` → 烟测

## 七、验收记录（2026-09-26 实测）

| 项 | 结果 |
|---|---|
| 纯文本问答 | ✅ 1+1=2 正常 |
| 图文描述 | ✅ 准确描述人物/场景/物品，中文回复 |
| OCR | ✅ 读出 "SEA WATER" 瓶标文字 |
| 图像编码 | ✅ prompt_tokens≈1070（图像 token 正常计入） |
| finish_reason | ✅ stop（完整收敛，非乱码非恒 token 0） |

### 7.1 生产口径实测（200K + MTP，09-26 晚补充）

启动脚本：`start-vl27-8996-v2.sh`（200K / seqs 3 / util 0.94 / MTP spec=1 / no-prefix-caching）。
**注意：MTP 接入多踩 4 坑**（见 §5.9–5.12）。

| 项 | 结果 |
|---|---|
| 短文本生成 | ✅ 16.5–18.3 tok/s（文本版基线 18.1，量级一致） |
| 短 JSON/代码（temp 0） | ✅ 19.4 / 18.7 tok/s 稳定输出（fr=stop）——低于 llama.cpp 同口径 27.5/30.3（其短结构化输出快 ~50%） |
| MTP 采纳率 | ✅ Mean acceptance length 1.71–2.00（生效，非 0%） |
| 100K 长上下文 | ✅ 94,879 tok prefill @ ~9,500 tok/s（~11s）+ 150 tok decode，总 120.8s |
| 100K 内容正确性 | ✅ 精准答出 0.0001 微米 / RO 膜寿命最长（94K 汉字中检索） |
| 长上下文 decode | ⚠️ ~2 tok/s 级（与文本版"长 ctx MTP 无优势"结论一致） |
| KV 余量 | KV 78,208 tokens → 200K 单发 0.39x；**200K 双发不可行**（需 ~1.28x>0.39x），VL 多模态版与文本版显存预算不同（视觉塔占用），不能照搬文本版 2.14x 结论 |
| 图文+MTP 同开 | ✅ 图文请求在 MTP serve 下正常（1070 tok 图像编码 + 描述正确） |

思考标签外泄仍在（`` 随正文输出），收尾项不变。

**对比参照**：同考卷下 VL-8B-FP8 路径（vllm 0.11.2 原生 Qwen3VL）输出恒为 token 0 乱码，未根治；
本方案 27B 一次跑通。

## 八、遗留与已知事项

1. **~~思考标签外泄~~（09-27 已修复）**：`--reasoning-parser deepseek_r1` 解决。
   注意**不能用 `qwen3` parser**——Qwen3 版要求输出同时含 `<think>` 开+闭标签，
   但 chat template 已把 `<think>\n` 注入 prompt，模型输出只有 `</think>`，
   strict 校验失败 → 整段推理连 `</think>` 留在 content。
   `deepseek_r1` parser 只需闭标签即可分割（无开标签时假定推理从头开始）→
   `content='2'` / `reasoning_content='推理过程'` 完美分离，速度无回退（16-19 tok/s）。
2. **生产口径未定**：8996 首验用 32K；200K 长上下文 + MTP + CG 并发口径需按 `RESTART-PLAN-54G.md`
   重新核算（KV 单价 64KB/token 口径沿用；mamba 状态 ~76MB/请求不变）。
3. **与文本 serve 共存**：显存独占（util 0.94 已满），27B 文本（8998）与 VL27（8996）不可同时起。
   切换顺序：杀旧（用 pid 文件 + `pkill -9 -f "VLLM::EngineCor[e]"`）→ 起新 → 盯日志到 startup complete。
4. VL27 未做速度基准；文本主线基准（单流 18.1 / 并发3 41.8 / 200K 双发 2.14x）不受影响。
5. 若未来升级 vLLM ≥0.30：`Qwen3_5ForConditionalGeneration` 原生支持本架构，
   本整套手术作废，直接用官方类 + 原始 NVFP4-HF 权重即可（需重做 sm_101 交叉编译，另一条线）。

## 九、文件索引

| 文件 | 位置 | 作用 |
|---|---|---|
| 本文 | `<host-workspace>/qwen38-surgery/PLAN-A-FINAL.md` | 权威复现文档 |
| 模型类 | `qwen38-surgery/qwen38_vl.py` ↔ 板上 venv `models/qwen38_vl.py` | 嫁接壳 |
| config | `qwen38-surgery/vl27b-config.json` ↔ 板上 `<board-model-dir>` | 嵌套 config（deepstack 空键已删） |
| 视觉塔抽取 | `qwen38-surgery/extract_visual.py` → `weights/visual-tower.safetensors` | 333 BF16 张量 |
| 启动脚本 | 板上 `<board-workspace>` | 8996 serve |
| 烟测 | 本地+板上 `vl27_smoke.py` | 文本/看图/OCR 三连 |
| 0.30 对照 | `qwen38-surgery/ref/vllm-0.30.0-*.whl`（已从 /tmp 抢救归档）的 `qwen3_5.py` | 官方实现参照 |
| 引擎主线文档 | `<host-workspace>/10-ENGINE-VERIFY-STATUS.md`（打包方内部，未随包） | 文本主线 + 本方案的工程上下文 |
