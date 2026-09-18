# Verifying persistence without actually cutting power

**中文标题：不等真断电，怎么验证"脏断电不丢"**

> 日期：2026-09-16 · 平台见 `README.md`

## 一、为什么不直接清空 overlay 试

把 overlay 上层清空 ≈ 把板子做一次出厂化：静态 IP、账号、sudoers、设备名全丢，
板子会**同时失联**（只能靠串口现场恢复），验证成本远高于收益，而且失败后的恢复动作本身
会把"验证"变成"事故处理"。所以用**等价状态**验证。

## 二、等价验证的分层

### L1 · 单件验证（每次改恢复脚本后都做）
逐个制造"该件被格式化"的状态，跑恢复，看是否回来：
1. 删单元 → `daemon-reload` → 跑恢复 → 单元应被重装且服务 `active`
2. 删 `/etc/profile.d/99-*.sh` → 跑恢复 → 文件应重建、登录 shell 能拿到环境
3. 把 `/etc/resolv.conf` 改回出厂坏值 → 跑恢复 → 解析应恢复、模型端可达
4. 大页池调低 → 跑恢复 → 应回到目标值

### L2 · 合并验证（本轮实做）
一次性把上述薄壳全部破坏 + 停服务，再跑恢复，然后逐项复验。

```bash
# 破坏（只动自己部署的东西，不碰系统其它部分）
sudo systemctl stop hermes.service llama-server.service
sudo rm -f /etc/systemd/system/hermes.service /etc/systemd/system/llama-server.service
sudo rm -f /etc/profile.d/99-hermes.sh
printf 'nameserver <出厂坏值>\n' | sudo tee /etc/resolv.conf
sudo systemctl daemon-reload
# 恢复
sudo bash /brand_data/hermes/restore-hermes.sh
# 复验
systemctl is-active hermes.service llama-server.service
head -1 /etc/resolv.conf
python3 /brand_data/hermes/bin/check-ds.py     # 模型端可达性自检
hermes chat -q "一句话自我介绍"                 # 端到端
```

## 三、本轮实测记录（board3，2026-09-16）

| 阶段 | 结果 |
|---|---|
| 破坏后 | 单元 0 个；profile.d 缺失；`resolv.conf` = 出厂坏值；服务 `failed/inactive`；域名解析失败 |
| 恢复后 | 2 个单元已重装；`profile.d/99-hermes.sh` 就位；`resolv.conf` = 可用解析器（含 `options timeout:2 attempts:2`）；两个服务 `active`；模型端可达 **0.59–0.81 s** |
| 端到端 | `hermes chat -q` 正常返回（2–3 s） |

## 四、L3 · 真断电彩排（本轮未做，建议单独排期）

真正把 overlay 清空 + 由板外常电主机经串口跑全套恢复。做之前必须满足：
- 串口看护主机在场且**其自身存活已自证**（见 `persistence-autostart-*.md` 第 6 条）；
- 板端静态 IP 等其它恢复项都在同一恢复脚本覆盖范围内（否则板子会失联）；
- 有人能到现场（最后一层保险）。

## 五、验收清单（可复制进部署记录）

- [ ] 服务 `active` 且 **开机自启已 enable**
- [ ] 登录 shell 直接可用（`hermes --version`）
- [ ] 模型端可达（自检脚本输出耗时）
- [ ] 端到端真实问答通过（记录模型名与实际延迟）
- [ ] L2 等价验证四项全过（单元 / profile / DNS / 大页池）
- [ ] 恢复脚本幂等（连跑两次，第二次应无实际变更）
- [ ] 持久区留有离线重装资产与文档
- [ ] **恢复日志可读**（每次恢复写时间戳、做了什么、复验结果——这份日志是"它确实能回来"的唯一证据）

[整理者注] 实测记录取自 board3 的恢复日志与自检输出；路径、账号、主机代号按仓库既有映射中性化，凭据类内容已删除。
