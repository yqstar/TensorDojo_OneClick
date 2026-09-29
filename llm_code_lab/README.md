# TensorDojo · AI 算法训练场

一个用于**亲手实现、运行并检验推荐算法、大模型与强化学习模块**的本地 LeetCode 风格练习器。当前共 **90 道题：大模型 42 道、推荐算法 32 道、强化学习 16 道**。本轮在原有 50 题之后追加 24 道大模型题和 16 道强化学习题，原有题目 ID 与顺序保留。
不是模型能力排行榜，也不是通过 LLM 主观打分。题目代码由真正的 Python / PyTorch 执行。

## 两种使用方式

| 方式 | 可以做什么 | 需要什么 |
| --- | --- | --- |
| 双击 `index.html` | 离线看题、编辑代码、查看提示和题解、笔记、计时、导出备份 | 现代浏览器 |
| 运行 `python start.py` | 上述全部 + 真实前向、数值、梯度判题 | 可用的 Python + PyTorch 环境 |

**HTML 没有内置 Python 或 PyTorch 解释器。只下载 HTML 不会得到真实判题能力。**
前端无 CDN、无远程字体、无模型 API；安装依赖完成后，工具运行不依赖互联网。点击“原始资料”链接会由浏览器打开外部网站。

## 快速启动

完整项目优先使用根目录的 `Start.command`（macOS / Linux）或 `Start.bat`（Windows）。macOS 终端也可以在项目根目录执行：

```bash
bash Start.command
```

启动器会复用可用环境，必要时创建项目环境并安装依赖，执行题库自检后打开浏览器。首次准备依赖需要联网；具体安装、离线缓存和兼容分支见 [双击启动说明](../README_双击启动.md)。已经运行旧版本服务时，先按 `Ctrl+C` 停止，再重新启动并刷新页面。

已有可以 `import torch` 的环境，直接激活该环境，进入解压后的 `llm_code_lab` 目录：

```bash
python start.py
```

服务默认打开 `http://127.0.0.1:8765`。请使用服务页面操作，不要继续在 `file://` 页面里提交。

没有现成环境时，常规安装方法如下。系统、处理器和 Python 版本必须有相应的 PyTorch wheel；首次安装需要联网：

```bash
cd llm_code_lab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python selftest.py
python start.py
```

Windows 的激活命令是：

```bat
.venv\Scripts\activate
```

端口被占用时：

```bash
python start.py --port 8766
```

不自动打开浏览器：

```bash
python start.py --no-browser
```

停止服务：在终端按 `Ctrl+C`。

### 环境兼容性

所有算例仅使用 CPU。当前项目环境使用 Python 3.11 / PyTorch 2.2.2；自检成功后会把实际 Python、系统和 PyTorch 版本写入 `test_report.json`，跨环境使用以本机自检结果为准。
不要求 GPU、不加载模型权重、不进行大规模训练。

`requirements.txt` 没有锁定操作系统相关的完整依赖树，其他版本必须通过 `python selftest.py` 自测。
尤其是旧版 Intel macOS，不能假设最新 Python / 最新 PyTorch 都提供兼容 wheel；先复用现成能运行 PyTorch 的环境，或使用根目录启动器的兼容安装分支。

官方环境说明：
- https://pytorch.org/get-started/locally/
- https://pytorch.org/get-started/previous-versions/

遇到 `No matching distribution found for torch`，优先核对 Python 版本、操作系统和处理器架构，不要通过随意复制网络上的 wheel 来绕过兼容性问题。

## 题库：大模型 42 道、推荐 32 道、强化学习 16 道

### 原有大模型基础题

| # | 模块 | 重点 |
| --- | --- | --- |
| 01 | ReLU | 前向、零点导数约定、保留计算图 |
| 02 | ReLU 手写反向 | 上游梯度与链式法则 |
| 03 | 稳定 Sigmoid | 极端数值、前向与反向稳定性 |
| 04 | SiLU / Swish | 激活与逐元素乘法 |
| 05 | GELU tanh 近似 | 明确近似版本 |
| 06 | 稳定 Softmax | 归一化维度与数值稳定 |
| 07 | Linear | 最后一维投影、非方阵权重 |
| 08 | LayerNorm | 方差口径、eps、D=1 |
| 09 | RMSNorm | 不减均值、均方根缩放 |
| 10 | SwiGLU FFN | 两路上投影、门控、下投影 |
| 11 | Scaled Dot-Product Attention | L≠S、Dk≠Dv、缩放维度 |
| 12 | Masked Attention | causal / padding / 广播 / 全屏蔽 |
| 13 | Causal Multi-Head Attention | 拆头、拼头、输出投影 |
| 14 | RoPE | 相邻维旋转、位置偏移 |
| 15 | GQA | query 头与 KV 头的映射 |
| 16 | KV Cache | 前缀拼接、chunk 因果偏移 |
| 17 | Cross-Entropy | logits 到稳定平均损失 |
| 18 | Pre-Norm Decoder Block | RMSNorm + MHA + SwiGLU + 两次残差 |

第 18 题是一个 Decoder Block，不是完整 Transformer 语言模型；它不含 embedding、RoPE、dropout、LM head、采样器等。

### 推荐算法面试手写题

新增题目按规划顺序排在原 18 道之后。P0 / P1 / P2 是练习优先级，按知识依赖和练习价值安排，不是面经频次排行榜；三个层级都已经加入题库。

| 优先级 | 数量 | 题目 |
| --- | --- | --- |
| P0 · 首批必练 | 16 | MSE、BCE with Logits、逻辑回归前向、逻辑回归手写梯度、SGD + Momentum、Adam、AUC、GAUC、Recall@K / NDCG@K、Embedding Mean Pooling、双塔余弦评分、BPR、In-batch Softmax、FM 二阶交互、DCN Cross Layer、DIN 兴趣聚合 |
| P1 · 核心进阶 | 8 | 加权 BCE、Binary Focal Loss、进阶 CE、全局梯度范数裁剪、DeepFM、Wide & Deep、MMoE、ESMM 联合损失 |
| P2 · 专项进阶 | 8 | ECE 概率校准、LogQ 采样修正 Softmax、FFM、CIN 单层、AUGRU Cell、SASRec 风格 Block、PLE 提取层、IPS / SNIPS |

基础多分类 CE 复用原第 17 题 `cross_entropy`，不重复新增；进阶 CE 单独覆盖软标签、label smoothing 和 `ignore_index`。模型题采用题面固定的函数式版本，例如 DIN 为简化评分器加未归一化兴趣聚合、SASRec 为指定结构的 Block；请按各题契约实现，不把模块练习当作完整工业推荐系统。

每道题包含：中文描述、公式、输入输出口径、示例或维度流、分层提示、易错点、追问、参考实现和原始资料。
题目暂时使用函数接口，不要求提交 nn.Module 类；权重显式传入，更便于检查输入和参数的梯度。

### 新增大模型题 · 24 道

| 模块 | 数量 | 题目 |
| --- | --- | --- |
| 参数适配与手写反向 | 3 | LoRA 前向、LoRA 权重合并、RMSNorm 手写反向 |
| 位置与注意力 | 5 | 正弦位置编码、ALiBi 偏置、滑动窗口注意力、RoPE 线性位置插值、Cross-Attention |
| 稀疏模型与缓存 | 4 | MoE Top-k 路由、MoE 前向、负载均衡辅助损失、定长 KV Cache 更新 |
| 生成与采样 | 4 | 温度 Softmax、Top-k 分布、Top-p / Nucleus 分布、重复惩罚 |
| 训练与偏好对齐 | 6 | Causal LM Loss、序列 Log Probability、KL 蒸馏、DPO、奖励模型成对损失、Token Perplexity |
| 优化与累积 | 2 | AdamW 单步更新、按有效样本数加权的梯度累积 |

新增大模型题按 P0 11 道、P1 12 道、P2 1 道安排。建议先完成 LoRA、Cross-Attention、Causal LM Loss、温度与截断采样，再练 MoE、蒸馏与 DPO。

这里的 Top-k / Top-p 计算确定性概率分布，不做随机抽样；MoE 使用题面固定的无容量限制版本；RoPE 扩展特指线性位置插值。DPO 输入已计算好的序列 log-prob，不包含整套数据整理或模型微调流程。

### 新增强化学习题 · 16 道

| 优先级 | 数量 | 题目 |
| --- | --- | --- |
| P0 · 首批必练 | 10 | 折扣回报、TD(0) Target、GAE、优势标准化、REINFORCE、分类策略熵、重要性采样比率、PPO 裁剪策略损失、DQN 损失、Polyak 更新 |
| P1 · 核心进阶 | 4 | PPO 裁剪价值损失、Double DQN Target、价值迭代单步、GRPO 组内优势 |
| P2 · 专项进阶 | 2 | C51 分布投影、GRPO Token 策略损失 |

建议顺序：回报与优势 → 策略梯度与 PPO → 价值学习 → GRPO。每道题只实现一个可用小张量核验的模块，不需要安装 Gymnasium、运行游戏环境或下载语言模型。

题面明确区分真正终止与时间截断：GAE 在真正终止时关闭 bootstrap，在终止或截断处停止优势递推。策略损失只对新策略求导，旧策略、参考策略、优势和目标值按题面作为常量。PPO 价值损失采用明确指定的裁剪变体；GRPO 使用逐序列有效 token 平均再按有效序列平均的教学版本，不代表所有训练框架的默认实现。

## 一次练习怎么做

界面提供浅色与深色外观，沿用浏览器中已保存的主题；窄屏可以通过左上角打开题库。方向与优先级菜单支持方向键、Home/End、键入匹配、Enter 确认、Escape 取消，以及 Tab 移焦。页面与交互验收见 [页面优化报告](../TEST_REPORT_页面优化.md)。

左侧可以按“推荐算法 / 大模型 / 强化学习”和 P0 / P1 / P2 筛选，继续结合搜索、收藏与通过状态选题。顶部“学习路径”按方向折叠展示可直接跳题的路线；推荐方向从 MSE / BCE / 原有 CE 开始，接逻辑回归、优化器、评估指标，再进入召回、特征交互、兴趣与多任务。无需先做完全部大模型题。

先只看题目和张量维度，把实现写到右边编辑器。“运行示例”检查前两组测试，不记入通过进度。
“提交评测”运行完整本地测试。失败时展开对应结果，查看失败阶段、输入摘要、输出前八个元素以及异常堆栈。

参考代码默认折叠。展开后，本轮通过记为“参考后通过”。重置草稿可以开始新一轮未看题解的练习。
这只是帮助个人复盘的状态，不是身份验证、防作弊或掌握程度的严格证明。
进度中的“通过”是历史记录；通过后修改代码，不会自动证明新代码也正确。

### 快捷键

- `Ctrl / Cmd + Enter`：运行示例；加 `Shift`：提交完整测试。
- `Ctrl / Cmd + S`：保存；`Ctrl / Cmd + K`：搜索。
- `Tab`：插入四个空格；`Enter`：基础自动缩进；`Esc`：离开代码编辑器。
- `F8` 或双击左上角标志：切换到本地工作记录页；F8 返回。

编辑器是轻量 textarea + 语法着色，不是完整 IDE；暂不含类型检查、智能补全、调试断点和完整 Python 语法解析。

## 判题如何工作

```text
浏览器中的 Python 草稿
    ↓ POST /api/run（本地、同源、启动时随机 token）
本地 HTTP 服务
    ↓ 独立 Python 子进程（单次最多 20 秒）
构造固定种子的测试输入
    ↓
运行用户实现与独立参考实现
    ↓
输入不变性 → 返回契约 → 有限数值 → 前向对齐 → 一阶梯度对齐
    ↓
结构化 JSON 结果 → 页面诊断与练习进度
```

### 正确性标准

真实检查：Tensor / tuple 类型、shape、dtype、device、输入是否被修改、输出是否有限，以及容忍误差范围内的数值一致性。输入检查还包含容器结构、Tensor 身份、stride 和执行前记录的 `requires_grad`；原地 `detach_()`、删除权重项或替换输入不能绕过梯度检查。

float64 前向容忍为 `rtol=1e-7, atol=1e-8`；float32 为 `rtol=2e-4, atol=2e-5`。梯度容忍分别放宽五倍。

题目区分两种判题方式：

| 元数据 | 页面标签 | 检查内容 |
| --- | --- | --- |
| `check_mode='autograd'` | 前向 + 梯度 | 前向数值、返回与输入契约，以及题面指定输入和模型参数的两个随机 VJP |
| `check_mode='numeric'` | 数值评测 | 返回数值、返回与输入契约；用于手写梯度、优化器、评估指标等，不要求对结果继续自动求导 |

可微题只对题面指定的预测输入和模型参数求导；标签、样本权重、采样概率等固定数据不参与求导，具体通过算例的 `grad_paths` 选择。默认未指定 `grad_paths` 时，检查全部浮点输入。模型参数权重与固定样本权重的角色不同，不应混淆。

比较的是两个随机上游向量对应的 VJP（vector-Jacobian product），而不是只对 `output.sum()` 求导。
例如 Softmax 的输出和恒为 1，只验证这个和的梯度可能漏掉错误。
本版**没有**运行有限差分 `gradcheck`、完整 Jacobian 枚举或二阶梯度验证。ReLU 手写反向和逻辑回归手写梯度只核验返回的梯度数值；优化器题包含有限多步的状态更新检查，不以训练损失下降代替正确性检查。

输入覆盖按题目设计，包含 float32/float64、多维、小维度、非连续输入、极端 logits，以及 mask / KV cache 的特定边界。
种子固定为 1729，以便复现；并非每次提交都换一套未知随机题。测试是有限集合，通过并不等于形式化证明。

耗时仅记录一次 CPU 前向调用，不含严格的 warm-up / 多次测量设计，因此**不用于性能排名或复杂度证明**。

### 明确的题目约定

ReLU 零点导数取 0；GELU 使用 tanh 近似；LayerNorm 使用总体方差；RMSNorm 不减均值；RoPE 相邻维配对。
Attention 布尔 mask 的 True 表示可见；全屏蔽行返回零且梯度有限。
KV cache 题使用前缀位置偏移的 causal mask，不直接套用左上对齐的非方阵 causal mask。
推荐题同样固定口径：AUC 并列正负对记半分；GAUC 跳过单类组并按有效组样本数加权；NDCG 使用指数 gain、同分按原索引稳定排序；DIN 不做历史权重 softmax；LogQ 题明确规定采样修正与同 ID 假负例 mask。
这些约定写进题目，避免把另一种合法口径误判为代码错误。

### 不做什么

当前不会拦截 F.relu / F.softmax 等高层快捷调用；“手写”约束主要依靠题目建议与个人自律。
不把答案字符串、变量名、代码长度或 LLM 的主观判断作为数值正确性的依据。
测试源码和参考实现都在本地，可被查看；它不是竞赛级隐藏测试系统。

## 安全边界：务必先读

**这是可信个人代码的本地练习器，不是恶意代码沙箱。**

服务仅绑定 127.0.0.1，并进行 Host、Origin 和随机请求 token 检查；每次判题使用独立子进程，有超时、有限日志、一次一个任务。
但提交代码依然具有当前用户的文件系统和网络权限，未配置硬内存限制、文件隔离、网络隔离、seccomp 或虚拟机隔离。
进程隔离与超时不能阻止任意恶意代码，也不能防止所有资源耗尽情况。

不要运行来源不明的代码。不要把这个服务暴露到公网或不可信内网。
要做多人在线平台，必须换成专门隔离的判题 worker，并补齐资源限制、网络隔离、权限、任务队列及审计；不能直接复用当前 exec 方案对外服务。

## 数据存储

草稿、笔记、计时、收藏与进度存在浏览器 localStorage；工具没有服务器端账户或数据库。
本轮扩充保留原 50 题的 ID、顺序和原 localStorage 存储键。在相同浏览器、相同来源地址打开更新后的页面，既有题目的记录继续按原 ID 读取；新增题目使用独立 ID，不重新编号存储记录。
清理浏览器数据会丢失本地记录。file://、不同端口、127.0.0.1 与其他域名属于不同的存储上下文。
通过“导出进度 / 导入”在不同浏览器或打开方式间迁移。备份包含代码和笔记，分享前自行检查。

浏览器禁用 localStorage 时页面会提示导出备份；不能保证所有浏览器对 file:// 存储提供完全一致的行为。

## 目录与二次开发

```text
llm_code_lab/
├── index.html          # 可离线浏览的完整前端，嵌入题库
├── ui.template.html    # 前端编辑源文件
├── build_bank.py       # 原有题目、add 接口、分组排序与构建入口
├── rec_basics_bank.py  # 推荐损失、逻辑回归、优化器与去偏题面
├── rec_basics_cases.py # 对应独立参考、算例与错误反例
├── rec_retrieval_bank.py  # 推荐评估指标与召回题面
├── rec_retrieval_cases.py # 对应独立参考、算例与错误反例
├── rec_models_bank.py  # 特征交互、序列与多任务题面
├── rec_models_cases.py # 对应独立参考、算例与错误反例
├── llm_blocks_bank.py  # LoRA、位置、注意力、MoE 与缓存题面
├── llm_blocks_cases.py # 对应独立参考、算例与错误反例
├── llm_training_bank.py  # 解码、训练、对齐与优化题面
├── llm_training_cases.py # 对应独立参考、算例与错误反例
├── rl_bank.py          # 回报、策略梯度、价值学习与 GRPO 题面
├── rl_cases.py         # 对应独立参考、算例与错误反例
├── problems.json       # 生成的题库元数据
├── cases.py            # 原有参考与算例，以及六组扩展模块分发
├── judge.py            # 返回契约、数值与 VJP 判题器
├── start.py            # 本地服务与子进程执行
├── selftest.py         # 参考答案回归 + 错误实现反例
├── requirements.txt
├── test_report.json
├── browser_report.json
├── api_report.json
└── README.md
```

### 扩展题目

原有题目仍在 `build_bank.py` / `cases.py`；扩展题分为三组推荐、两组大模型和一组强化学习模块，接口一致：

- `register(add)`：注册中文题面、签名、独立可运行的学习用答案、来源和前置题；通过 `track='llm'/'recsys'/'rl'`、`priority='P0'/'P1'/'P2'`、`check_mode='autograd'/'numeric'` 声明分类与判题方式。
- `IDS`：该算例模块支持的题目 ID 集合。
- `reference(pid, *args)`：独立数值参考，尽量采用与学习用答案不同的计算方法。
- `make_cases(pid, seed=1729)`：返回算例列表，首两例用于“运行示例”，全部用于“提交评测”。每项包含 `name`、`args`（tuple）、`grad`（bool），可选下述求导约束。
- `mutations(bank)`：返回 `{反例名称: (题目ID, 错误代码)}`，每道新题至少覆盖一种与核心考点有关的错误；`selftest.py` 会检查覆盖并验证其无法通过。

算例可以用 `grad_paths` 指定可求导输入，例如 `('input[0]', 'input[2].weight')`。路径支持精确 Tensor 和容器子树；`input[2]` 会选中其下的浮点 Tensor。整数标签和布尔 mask 不会开启梯度。`grad=False` 的数值题不执行 VJP，题面 `check_mode` 应与算例配置保持一致。

可选的 `require_connected_paths` 使用相同路径匹配方式：在独立参考对该路径存在梯度连接时，要求提交也有梯度连接，自动求导结果不能为 `None`；梯度值为零允许通过。它比一般数值 VJP 检查更严格，适合明确要求“返回与 logits 相连的零”的全忽略 CE 算例，不要求所有边界都强制添加无意义的依赖。

新增题还需更新 `build_bank.py` 中对应的分组顺序和页面 `LEARNING_PATHS`；接入新的一组模块时，同步更新构建注册列表和 `cases.py` 的模块列表。构建器检查题目 ID、规划顺序、前置依赖存在性与循环；页面题数、分类计数和进度分母从题库动态读取。

在 `llm_code_lab` 目录运行：

```bash
python build_bank.py
python selftest.py
```

然后重启服务并刷新页面。新增题目应有 float32/float64 的有意义边界测试，同时验证公开示例与全量提交。

修改页面请改 `ui.template.html` 并重建，避免下次构建覆盖直接对 index.html 的修改。

筛选组件的行为回归可在项目根目录运行 `node tests/test_ui_controls.cjs`；展开菜单、窄屏与实际键盘操作仍需在浏览器中验证。

### 验证与报告

`python selftest.py` 会遍历当前全部题目的参考答案，再运行原有与六组扩展模块提供的错误实现反例。全部通过后写入 `test_report.json`，记录环境、题数、算例数和反例数；请以对应代码版本生成的报告为准。本轮验收见 [大模型与强化学习报告](../TEST_REPORT_大模型与强化学习.md)。

`browser_report.json` 与 `api_report.json` 保存界面和服务验证记录。旧报告只能说明对应版本的验证范围，不能代替扩充题库后的检查；升级后还需确认页面题数、方向与优先级筛选、学习路径跳转、真实运行/提交和已有草稿读取。

## 后续可扩展方向

可继续增加 debug 题、有限差分梯度题、同参数量模块替换题、消融实验题。
大模型方向可扩展量化、分布式并行与投机解码；强化学习方向可扩展 SAC、TD3 与离线策略评估。LoRA、AdamW、KL 蒸馏、Top-k / Top-p、MoE、PPO、DPO 与 GRPO 已纳入当前题库。
推荐方向可继续扩展 HSTU、其他 Target Attention 变体与 RankMixer token mixing；梯度裁剪、DIN、MMoE 和 PLE 已纳入当前题库。
优先把每个模块的输入输出口径和可靠测试做完整，再考虑云端账号、排行榜或 AI 提示生成。

## 数学与 API 原始资料

这些资料用于核对算法定义与接口，不代表作者为本项目背书。

- Attention Is All You Need：https://arxiv.org/abs/1706.03762
- GLU Variants Improve Transformer：https://arxiv.org/abs/2002.05202
- RoFormer：https://arxiv.org/abs/2104.09864
- Root Mean Square Layer Normalization：https://arxiv.org/abs/1910.07467
- PyTorch SDPA：https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
- PyTorch testing：https://docs.pytorch.org/docs/stable/testing.html
- PyTorch gradcheck：https://docs.pytorch.org/docs/stable/generated/torch.autograd.gradcheck.gradcheck.html
- 浏览器 Python 备选方案 Pyodide：https://pyodide.org/en/stable/usage/loading-packages.html

Pyodide 可以在浏览器加载 Python 和 NumPy 等兼容包，但必须打包相应运行时和依赖，不能把 CDN 页面说成“首次使用也完全离线”。
本原型选择本地 PyTorch，因此没有集成 Pyodide。
