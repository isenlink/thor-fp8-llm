# Wiring up the model: default, auxiliary roles, and the fallback chain

**中文标题：把模型接上 —— 默认模型、辅助角色、回落链的配置结构**

> 日期：2026-09-16 · 来源：board4 实测（已按本人校验）

## 一、为什么要单独讲这个

板上装完只是第一步。**Agent 的主模型、辅助模型、回落链是三处独立配置**，
只配一处会导致：能对话但辅助功能（标题生成、压缩、视觉、检索）走向默认供应商而失败，
或者主供应商挂掉时没有回落而整体不可用。三处都要写。

## 二、配置结构（三处，缺一不可）

```yaml
model:                                   # ① 主模型
  default: <model-id>
  provider: <provider-name>

custom_providers:                        # ② 供应商定义（凭据走 key_env，不写明文）
  - name: <provider-name>
    base_url: https://<api-host>/v1
    key_env: <PROVIDER>_API_KEY
    model: <default-model>
    models: [{id: <a>}, {id: <b>}]

auxiliary:                               # ③ 各辅助角色（逐角色指定）
  <role>:
    provider: <provider-name>
    model: <aux-model>
    base_url: http://<lan-host>:<port>/v1
    key_env: <KEY_ENV>
    timeout: 120

fallback_providers:                      # ④ 回落链，按顺序尝试
  - {provider: <p1>, model: <m1>, key_env: <K1>}
  - {provider: <p2>, model: <m2>, key_env: <K2>}
```

辅助角色不是只有一个——实际有十余个（视觉、网页提取、压缩、标题生成、技能检索、
审批、MCP、分类、看板分解、画像描述、策展、会话检索、记忆刷写等）。
**每个角色都能单独指定供应商与超时**；不指定则回落到主模型供应商。

## 三、凭据纪律（硬要求）

- **一律走 `key_env`**：配置文件里只出现**环境变量名**，真实值放 `HERMES_HOME/.env`，权限 `600`。
- 理由：`config.yaml` 是要被读、被传、被贴进 issue 的文件；凭据写进去等于把泄密面扩大。
- 验证：`hermes status` 会按供应商逐项显示"已设置/未设置"，**但不回显值**——用这个确认配置生效。

## 四、一个容易写反的取舍

内网端点作回落，价值是"**外网断的时候仍然可用**"，**不等于"更稳"**——
它自己的上游挂了同样不可用。文档里别把内网回落写成"更可靠"。

同理，**板本机 `llama-server` 作后端时才有那两道硬约束**（窗口 ≥64K、`--cache-ram 512`，
见 `hard-constraints-*.md`）；后端是云端 API 时不会遇到，别当成通用前提。
该文档已把适用范围写在开头。

## 五、实测记录（board4）

三端点板上直连全部通过（裸 API 延迟）：

| 角色 | 端点类型 | 实测结果 |
|---|---|---|
| 主模型 | 外网 API | ✅ 0.85 s，模型 `deepseek-flash` |
| 辅助 | 内网网关别名 | ✅ 0.39 s（16/16 稳定路由） |
| 回落 | 内网另一端点 | ✅ 0.40 s（思维链模型，`content` 可能为空、有 `reasoning_content`） |

端到端：`hermes chat` 非交互两轮成功（首轮 28.4 s 含初始化、次轮 3.9 s），
`hermes status` 正确识别默认模型，`errors.log` 为空。

⚠️ 一个必须注意的坑：**思维链模型的 `content` 可能为空**，正常回复落在 `reasoning_content`。
把它当主模型时，如果客户端只读 `content`，会表现为"模型没回答"。

[整理者注] 本节依据 board4 实测记录整理：主机与端点地址用占位（`<api-host>` / `<lan-host>`），
凭据只留环境变量名，实际密钥值未出现。延迟数字与模型标识 100% 保留。
