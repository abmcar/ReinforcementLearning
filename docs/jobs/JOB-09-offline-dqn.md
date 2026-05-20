# JOB-09: Offline DQN(CQL / Discrete BCQ)

**Phase**: B
**依赖**: JOB-08
**Status**: ⬜ Pending

---

## 背景

本任务只有历史日志,**不能在线交互**,属于典型的 offline / batch RL 场景。朴素 DQN 在 offline 上会对未见过的动作(OOD action)做高估,导致策略发散或偏。文献给出多种解法,CQL 和 Discrete BCQ 是最常用且效果验证充分的两种。本 job 在 JOB-08 的 DQN 基础上加入 offline RL 正则,作为最终主推方法。

## 目标

实现至少一种 offline RL 算法(CQL **或** Discrete BCQ),在 JOB-07 环境上完整训练,确认 Q 值不发散、策略合理(贴近 behavior policy 同时有改进)。

## 输入

- **依赖产出**:JOB-08 的 DQN/Double/Dueling 模型框架、JOB-07 的环境

## 工作内容

- 在 `docs/offline_dqn.md` 写明:为什么需要 offline RL、CQL 和 Discrete BCQ 各自的原理与权衡、最终选哪一个(或两个都做)、超参选择
- 实现 **CQL**(Conservative Q-Learning):在 DQN loss 上加 `α · E_{s~D}[ logsumexp(Q(s,·)) - Q(s, a_data) ]`(对 batch 求均值)
  - 或实现 **Discrete BCQ**:behavior policy estimator + threshold filter,约束动作只能从数据支持的动作集合选
- 训练监控:Q 值分布、loss 曲线、policy 与 behavior policy 的 KL / overlap
- 比较 offline 算法 vs 朴素 DQN(JOB-08)在 Q 值发散现象上的差异
- 输出训练好的模型权重(到 `outputs/checkpoints/`)和评估前的 sanity 报告

## 产出

**提交到 Git**:
- `src/rl/models/cql.py` 或 `src/rl/models/discrete_bcq.py`(至少一个)
- `experiments/configs/cql.yaml` / `discrete_bcq.yaml`
- `docs/offline_dqn.md`:算法说明、选型理由、训练监控结果
- `docs/figures/`:Q 值分布对比、loss 曲线

**不提交**:
- 模型权重、训练日志(放 `outputs/`)

## 验收标准

- [ ] 至少一种 offline RL 算法实现完成,训练**满足收敛标准**:Q 值 max 在最后 5k step 不再发散(标准差 / 均值的相对增长 < 5%),loss 滑动平均不再下降但稳定
- [ ] `docs/offline_dqn.md` 含算法原理、超参表、Q 值监控曲线
- [ ] 提供 vanilla DQN(JOB-08)vs offline DQN(本 job)的 Q 值分布对比图,定量说明 offline 正则把 Q 值 max 压低了多少

## 参考资料

- CQL:Kumar et al., NeurIPS 2020,[arXiv:2006.04779](https://arxiv.org/abs/2006.04779)
- Discrete BCQ:Fujimoto et al.,[GitHub](https://github.com/sfujim/BCQ)(`discrete_BCQ.py`)
- Offline RL 综述/经验:[Berkeley BAIR Blog](https://bair.berkeley.edu/blog/2020/12/07/offline/)

## 备注

- 如果时间紧,**Discrete BCQ 更简单**(无需调 α 之类的额外超参);CQL 更通用但更难调。
- 本 job 的输出是"训练好的离线 DQN agent"。完整的双目标实验(worker / requester)在 JOB-10 完成。
- 本 job 不要求多 seed(sanity 性质),正式多 seed(≥ 3)统计随 JOB-10 final 实验一并跑。
