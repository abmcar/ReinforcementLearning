# JOB-08: DQN / Double / Dueling 模型

**Phase**: B
**依赖**: JOB-07
**Status**: ✅ Completed

---

## 背景

作业明确要求"使用 DQN 系列模型"。本 job 实现 DQN 主线的基础模型族:vanilla DQN、Double DQN、Dueling DQN。后续 JOB-09 在此基础上加 offline RL 正则项,JOB-10 用 JOB-07 环境训练并评估。

## 目标

实现 vanilla DQN、Double DQN、Dueling DQN 三个模型的训练管线,在 JOB-07 环境上跑通一个小规模 sanity 训练(确认 loss 下降、Q 值不发散)。

## 输入

- **依赖产出**:JOB-07 的 `OfflineRecommendationEnv`、reward function、batch sampling 接口
- **配置**:超参用 YAML / dataclass 管理,放 `experiments/configs/`

## 工作内容

- 实现 Q 网络。输入约定:`(state, candidate_projects_features) -> Q_values over candidates`(候选集变长,不要 hardcode action 维度)。建议用 attention / scoring 头处理变长候选。
- 实现训练 loop:经验回放(可直接用全量 transition + 随机采样)、target network、ε-greedy 在 offline 场景下退化为软探索 / 不需要、loss(MSE / Huber)。
- 实现 Double DQN:用 online net 选 action,target net 估 Q。
- 实现 Dueling DQN:V(s) + A(s, a) - mean(A) 分解架构。
- (可选)Prioritized Experience Replay。
- **Sanity 训练**:在小数据集 / 小 batch / 短训练步数下,确认 loss 下降、Q 值不发散(典型上限可以加 clip)。
- 把超参、loss 曲线、Q 值统计输出到 `outputs/runs/JOB-08/`

## 产出

**提交到 Git**:
- `src/rl/models/dqn.py`(vanilla)、`src/rl/models/double_dqn.py`、`src/rl/models/dueling_dqn.py`
- `src/rl/trainer.py`(训练 loop)
- `experiments/configs/dqn_baseline.yaml` 等
- `docs/dqn_models.md`:三个模型架构说明、超参选择
- 训练曲线截图放 `docs/figures/`

**不提交**:
- 模型权重(放 `outputs/`、`checkpoints/`、`runs/`,都已加入 `.gitignore`)

## 验收标准

- [x] 三个模型(vanilla / Double / Dueling)训练 loop 跑通
- [x] Sanity 训练满足:**最后 1k step 的 loss 滑动平均比前 1k step 下降 ≥ 10%**,且 Q 值 max < 100(或在 `docs/dqn_models.md` 给出合理阈值与曲线)
- [x] 配置文件化,无 hardcoded 超参

## 参考资料

- Double DQN:Hasselt et al., [arXiv:1509.06461](https://arxiv.org/abs/1509.06461)
- Dueling DQN:Wang et al., [arXiv:1511.06581](https://arxiv.org/abs/1511.06581)
- 推荐场景组合参考:DRN, Zheng et al. WWW 2018, [DOI:10.1145/3178876.3185994](https://doi.org/10.1145/3178876.3185994)

## 备注

- 这里**只做模型 + sanity training**,不出最终指标。最终指标由 JOB-10 出。
- Q 值发散是 offline RL 经典坑(OOD 动作高估)。如果 sanity training 阶段就看到 Q 值飙升或 loss 震荡,**不要硬调超参**,这是 JOB-09 引入 offline RL 正则项的信号。
