# JOB-10: DQN 双目标训练与评估

**Phase**: B
**依赖**: JOB-09, JOB-04, JOB-05
**Status**: ✅ Completed

---

## 背景

到此为止,JOB-08/09 已经实现 DQN 模型和 offline 正则,JOB-07 提供 env 和两套 reward,JOB-04/05 提供评估框架和 baseline。本 job 是 DQN 主线的**最终交付**:用 worker reward 训一组 agent、用 requester reward 训一组 agent、跑完整评估,产出实验报告核心数据。

## 目标

完成 DQN 主线在**参与者**和**请求者**两个目标下的完整训练与评估,产出可被 JOB-16 直接引用的指标表和分析。

## 输入

- **依赖产出**:JOB-07 env + reward function、JOB-08/JOB-09 模型、JOB-04 evaluator、JOB-05 baseline 结果

## 工作内容

- **接入评估框架**:实现 DQN 模型到 JOB-04 `recommend(worker_id, timestamp, candidates) -> list[int]` 接口的适配层(签名见 `docs/roadmap.md` § 6.5;对候选集计算 Q 值后 argsort)。让 DQN/Offline DQN 都能直接喂给 JOB-04 evaluator。
- **参与者目标实验**:
  - 用 `worker_reward_fn` 训练若干模型(至少:Double+Dueling DQN、Offline DQN)
  - 多 seed(≥ 3)训练
  - 在 test split 上跑 JOB-04 全部指标
  - 与 JOB-05 baseline 对比
- **请求者目标实验**:同上,用 `requester_reward_fn`
- **超参敏感性**:至少跑一个超参的 ablation(如 reward 权重、候选集 K、γ)
- **分析**:
  - 模型在两个目标下哪些指标显著优于 baseline?哪些没有?
  - 同一模型在 worker reward 和 requester reward 下的策略差异(打印若干推荐示例)
  - Offline RL 是否比朴素 DQN 显著更好?
- 所有结果写入 `docs/dqn_results.md`(主结果表 + 分析)

## 产出

**提交到 Git**:
- `experiments/scripts/run_dqn_worker.sh`、`run_dqn_requester.sh`(一键复现)
- `experiments/configs/`:所有 final 实验的配置
- `docs/dqn_results.md`:完整结果表 + 分析 + 与 baseline 对比
- `docs/figures/`:训练曲线、指标对比图

**不提交**:
- 模型权重、所有 runs 日志(放 `outputs/`)

## 验收标准

- [ ] 参与者目标和请求者目标分别有完整实验结果
- [ ] 多 seed(≥3),报 mean ± std
- [ ] 与 JOB-05 全部 baseline 对比
- [ ] `docs/dqn_results.md` 包含结果表、分析、关键发现
- [ ] 一键脚本可复现 final 实验
- [ ] `docs/dqn_results.md` 记录**所有跑过的配置**(不只是 best),包括失败/不显著的结果

## 参考资料

- 对照论文与算法选型见 `docs/roadmap.md` § 6「DQN 主线」。
- 直接对照(双 DQN 双目标):Shan et al. 2019, [arXiv:1911.01030](https://arxiv.org/abs/1911.01030)

## 备注

- 训练时间预算未定,若全量训练过长,允许在 train split 的子集上跑 final,在 `docs/dqn_results.md` 标注。
