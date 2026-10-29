# AutoData：Agentic Self-Instruct 复现

本仓库复现 *Autodata: An agentic data scientist to create high quality
synthetic data*（Kulikov et al.，arXiv:2606.25996）中的 CS 论文设定。
一个 agent 读论文，写出研究问题、context 和带权重的 rubric。弱 solver
和强 solver 各自作答。弱 solver 答不好、强 solver 答得好，这道题才留下。
留下的题用 GRPO 训练弱 solver。外层循环改写 agent 自己的 prompt。

[English](README.md)

## 现状

下面每个环节都能运行，有测试覆盖（245 个）。哪些真实跑过，哪些没有：

| 环节 | 真实运行 | 结果 |
|---|---|---|
| 内循环 + 最终过滤 | gate，5 篇 SciArena 论文，flash 模型 | 5/5 接受，5/5 保留；18 轮；35 分钟；$0.43 |
| 内循环，默认模型 | smoke，1 篇论文，2 轮 | 两轮都是 `failed_on_strong` |
| 测试集评测 | gate 的 2 道题 | base mean@3 0.279，best@3 0.309（只验证流程，不算结果） |
| Meta 优化 | 试验，1 轮迭代，每边 10 个验证 session | 按固定过滤器计：种子 2/10，子代 5/10（见下） |
| GRPO 训练 | 冒烟测试，真实 batch 形状（16 × 8）跑 2 步，24 道题，2 张卡 | 梯度正常（第 2 步 KL 5e-4）；每步 13–14.5 分钟，judge 约 $0.15 |

gate 中被接受的题：弱 0.20–0.49，强 0.60–0.85，差距 0.33–0.42。

meta 试验只验证流程。每边 10 个 session、一个随机种子，说明不了 prompt
的好坏。analyzer 没有拿到任何提示，自己找出了论文 Sec 4 报告的两条原因：
高权重的 rubric 条目依赖只在论文里出现的事实；负向条目惩罚另一种站得住的
答案。子代在 loop 自己的 verifier 下过了 8/10。固定的最终过滤器又拦下其中
3 道：一道 context 直接给出了答案，两道 rubric 只要求复述 context。现在
meta 分数只在题目通过这道过滤器时才把 session 记为 pass。

本 README 的旧版本说 4B solver 对这个任务"太强"。那个结论来自一个 bug：
solver 读到了整篇论文。那些数字已经删除。

## 工作原理

### 角色

| 角色 | 默认模型 | `models=flash` |
|---|---|---|
| challenger、quality verifier、judge、meta-optimizer | Kimi-K3 | DeepSeek-V4.1-Flash |
| 强 solver | Qwen3.8-max | Qwen3.8-Flash |
| 弱 solver | Qwen3.5-4B（vLLM） | Qwen3.5-4B（vLLM） |

论文用的是 Kimi-K2.6、Qwen3.5-397B-A17B 和 Qwen3.5-4B。所有远程角色
都走同一个 OpenAI 兼容网关（`PROXY_BASE_URL`）。

### 内循环（Sec 3.1，Fig 7）

每篇论文最多 15 轮：

1. challenger 读完整篇论文，写出题型、技能标签、context、问题、参考答案，
   以及 10–15 条带权重的 rubric。
2. quality verifier 读论文和这份题目包，检查四件事：context 是否泄露答案，
   问题是否只考记忆，rubric 条目数，题型与问题是否一致（Fig 9）。
3. 弱 solver 作答 3 次。它只看到 context 和问题，看不到论文。
4. judge 逐条给每个答案打分，看不到参考答案。
5. 弱 solver 的分数过关后，强 solver 作答 3 次，用同样方式打分。
6. 接受条件：weak_avg ≤ 0.65，weak_max ≤ 0.75，弱 solver 没有 0 分，
   0.60 ≤ strong_avg < 0.95，差距 ≥ 0.20（Fig 7）。
7. 不满足时，challenger 拿到这篇论文此前失败的题（按失败类型分组），
   换一个角度重出。

两个 solver 都开 thinking，共用 65,536 token 的上限，temperature 1.0。
judge 只评 `</think>` 之后的文字。网关故障时重跑同一道题，不记成模型答错。

被接受的题还要过最终过滤器：种子 verifier prompt 再读一遍，再加固定的
rubric 条目数规则（正向 ≥ 4，负向 ≥ 3，总数 10–20）。只有过了这一关的题
才写进 `final_qa.jsonl`。

### prompt 即数据

十个 prompt 以文本文件存放在 `src/autodata/llm/prompt_text/cs/`，加载成
不可变的 `PromptSet`。其中四个可以改写：challenger 的 system prompt、
首轮指令、重出指令，以及 verifier 的 system prompt。solver 和 judge 的
prompt 是量尺，固定不动；user 消息的框架也固定。用 `prompts_dir` 可以让
一次运行换用另一套 prompt。

### Meta 优化（Sec 4）

每轮迭代按论文的七步走：

1. 按 Boltzmann 采样选父代，T = 0.1。
2. 在一小批训练论文上运行父代。
3. analyzer 读完整轨迹，写出根因分析。
4. implementer 以分隔块的形式改写可变 prompt。它写的任何内容都不会被执行。
5. 在验证论文上分别运行父代和子代。
6. 子代分数严格高于父代历次评估的均值，才保留子代。
7. 记录结果，供下一轮的 analyzer 读取。

分数是 session 通过率。一个 session 只有在题目被接受、并通过固定的最终
过滤器时才算通过。因网关故障丢失的 session 不计入分母。打上分的 session
不到 80% 时，这次评估作废。

### 数据、评测、训练（Sec 3.1.2）

- `build_dataset.py` 只读 `final_qa.jsonl`，去重后留出 100 道测试题。
  按论文切分，测试论文不会出现在训练集里。
- `eval_solver.py` 用弱 solver 的设置把每道测试题答 3 次，报告 mean@3
  和 best@3。某次作答因网关故障丢失，这道题不计分，单独记数。
- `train_grpo.py` 通过 TRL 跑 GRPO。chat prompt、thinking 开关、token
  上限和温度与 loop 中的弱 solver 一致。rollout 直接从策略本身采样，
  不做 top-k/top-p 截断，不加 presence penalty：TRL 会校正 vLLM 的
  processed logprob 与训练端 logprob 之间的差，这些处理会把差拉大，
  直到每条回答都被屏蔽。reward 是 loop 的 judge 对答案的打分。judge
  出错时不给 reward，而不是给 0。每步 16 道题 × 8 条回答，lr 恒定 1e-6，
  200 步，KL 0.04，逐 token 截断的重要性采样。

## 运行

```bash
pip install -e ".[dev]"                 # 流水线和测试
pip install -e ".[serve,train]"         # GPU 机器上
cp .env.example .secrets/remote.env     # 填 PROXY_*、WEAK_SOLVER_BASE_URL
export AUTODATA_DATA_ROOT=/data/autodata   # 其下有 papers-sciarena/、outputs/、models/Qwen3.5-4B/
```

先起弱 solver，再跑各个环节。每个脚本都是 Hydra 应用，任何配置项都能在
命令行覆盖。

```bash
MODEL_DIR=$AUTODATA_DATA_ROOT/models/Qwen3.5-4B bash scripts/serve_weak_solver.sh

python scripts/run_inner_loop.py run_id=gate run.papers_limit=5 models=flash
python scripts/build_dataset.py run_id=data1 'data.runs=[<outputs_root>/gate]'
python scripts/eval_solver.py run_id=data1 eval.test_set=<outputs_root>/data1/data/test.jsonl
python scripts/run_meta.py run_id=meta1 models=flash meta.iterations=2
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --config_file conf/accelerate/zero3.yaml \
    scripts/train_grpo.py run_id=grpo1 train.data=<outputs_root>/data1/data/train.jsonl
```

每次运行写到 `<outputs_root>/<run_id>/`，里面有 Hydra 配置和运行日志。
调用模型的运行还会写 `usage.jsonl`，记录每次调用的 token 数。

在笔记本上对接 GPU 机器时，在 `.secrets/remote.env` 里填好 `REMOTE_*`。
`scripts/sync_to_remote.sh` 推送代码（`-n` 只演练），
`scripts/sync_from_remote.sh` 拉回结果。

测试：`pytest`。lint：`ruff check src tests scripts`。

## 花费与时间

用 `models=flash`，按网关标价：

- 内循环：5 篇论文的 gate 用了 35 分钟、$0.43，同时跑 4 篇。
- Meta 优化：试验（2 个训练 session，2 × 10 个验证 session）用了
  96 分钟、$0.54，`meta.workers=10`。默认配置（5 篇训练论文，25 篇验证
  论文 × 4 个样本，父代和子代各一遍）在 `meta.workers=4` 下估计每轮
  约 $5、7 小时，这是估算。大部分时间花在强 solver 上。
- GRPO：两张 RTX 6000D 上每步 13–14.5 分钟，按 2 步实测；200 步约 45
  小时。回答平均 5.5k–6k token，最长 12k，没有一条被截断。judge 每步
  约 $0.15。

## 与论文的差异

- **语料。** 2,543 篇 Markdown 格式的 SciArena 论文，不是论文用的 1 万篇
  S2ORC。其中 3 个文件有问题：两个装的是别的文档，一个是空文件（读取时跳过）。
- **接受条件。** 用 Fig 7 的阈值。Sec 3.1 正文给的是另一组（strong ≥ 0.65，
  weak < 0.5）。
- **轮数上限。** 15。论文报告每道被接受的题平均用 6.59 轮，没给上限。
- **Meta session。** 每个 session 只跑一轮（`meta.session_rounds=1`）。
  完整跑 15 轮几乎总能通过（gate：5/5），分不出 prompt 的好坏。论文的
  session 跑 6 小时。只跑一轮时，重出指令用不上。
- **Meta 规模。** 4 轮迭代，依次执行。论文跑了 233 轮，多轮并行。
- **最终过滤器。** 论文没有给出它的过滤 prompt。我们沿用种子 verifier 和
  Fig 9 的条目数规则。规则要求至少 3 条负向条目，因此论文 Sec 4 发现的
  "只用正向条目"的 rubric 在这里会被拦下（`final_filter.min_negative`）。
- **训练。** 全参微调，与 lr 1e-6 相配。权重和 Adam 状态用 fp32：在
  bf16 下，这个学习率的大部分更新会被舍入掉。加上 fp32 的参考模型，
  一张 85 GB 的卡放不下（单卡冒烟测试 OOM），所以用 ZeRO-3 分到两张卡，
  每张卡各带一个同卡 vLLM，反向传播时休眠；峰值每卡 72 GB。每题 8 条
  回答和恒定学习率是我们自己定的。rollout 去掉了 loop 的 top-k、top-p
  和 presence penalty（见上）。checkpoint 只存权重，完整的约 54 GB。
- **没有实现。** CoT Self-Instruct 基线及其测试集，以及法律推理设定。

## 目录

```
conf/                    Hydra 配置：config.yaml、models/、acceptance/；
                         GRPO 用的 accelerate/zero3.yaml
scripts/                 run_inner_loop、run_meta、build_dataset、eval_solver、
                         train_grpo、serve_weak_solver、sync_*
src/autodata/
  llm/                   client、schemas、PromptSet、prompt_text/cs/*.txt、用量计量
  pipeline/              inner_loop、quality_verifier、evaluate_rubric、
                         acceptance、final_filter、at_k
  meta/                  outer_loop、evaluation、digest、analyzer、implementer、
                         population、history
  data_module/           paper_loader、qa_dataset、arxiv_html
  training/              grpo（rollout 设置与 reward）
tests/                   fakes.py 在不联网的情况下驱动每个 agent
```
