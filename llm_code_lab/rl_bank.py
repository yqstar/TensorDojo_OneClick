"""Small, explicit reinforcement-learning components; no environments or training loops."""

SOURCES = {
    'gae': ('Generalized Advantage Estimation 原始论文', 'https://arxiv.org/abs/1506.02438'),
    'ppo': ('Proximal Policy Optimization 原始论文', 'https://arxiv.org/abs/1707.06347'),
    'ppo_impl': ('OpenAI Baselines PPO2：value clipping 实现', 'https://github.com/openai/baselines/blob/master/baselines/ppo2/model.py'),
    'dqn': ('PyTorch DQN 教程：Huber 与目标网络', 'https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html'),
    'ddqn': ('Deep Reinforcement Learning with Double Q-learning', 'https://arxiv.org/abs/1509.06461'),
    'c51': ('A Distributional Perspective on Reinforcement Learning', 'https://arxiv.org/abs/1707.06887'),
    'grpo': ('DeepSeekMath：GRPO 与 outcome supervision', 'https://arxiv.org/html/2402.03300v3#S4.SS1'),
    'terminal': ('Gymnasium：区分 termination 与 truncation', 'https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/'),
    'distribution': ('PyTorch Distributions：Categorical 与 score function', 'https://docs.pytorch.org/docs/stable/distributions.html'),
    'vi': ('UC Berkeley CS188：Value Iteration', 'https://inst.eecs.berkeley.edu/~cs188/textbook/mdp/value-iteration.html'),
}


def register(add):
    def put(pid, title, topic, level, signature, summary, formula, contract,
            example, hints, pitfalls, followups, solution, *, priority='P0',
            mode='numeric', sources=('ppo',), prerequisites=(), minutes=25):
        add(pid, title, topic, level, signature, summary, formula, contract,
            example, hints, pitfalls, followups, solution,
            sources=tuple({'title': SOURCES[s][0], 'url': SOURCES[s][1]} for s in sources),
            track='rl', priority=priority, check_mode=mode,
            prerequisites=prerequisites, minutes=minutes)

    put('discounted_returns', '折扣回报 · 终止与 Bootstrap', '强化学习 · 回报与优势', '基础',
        'discounted_returns(rewards, terminated, bootstrap, gamma=0.99)',
        '对并行轨迹反向计算折扣回报，遇到真正终止断开未来奖励；未终止的采样尾部使用给定 bootstrap。',
        'G_T=bootstrap；G_t=r_t+γ·(1−terminated_t)·G_{t+1}',
        ['rewards：[T,B] 浮点；terminated：[T,B] bool；bootstrap：[B]；T,B≥1，0≤gamma≤1。',
         'terminated[t,b] 指执行第 t 步后环境真正终止；终止步自身奖励仍计入。中间终止后可接新 episode，但不得跨终止累积。',
         '返回 [T,B]，保持 dtype/device；所有输入固定，仅数值评测，不原地修改。',
         '本题只接收 termination；若采样尾部因时间限制截断，应把对应终态价值传入 bootstrap；中间 truncation 的处理另见 GAE。'],
        '例 1：r=[[1],[2]], terminal全False, bootstrap=[4], γ=0.5 → [[3],[4]]。\n例 2：同输入 terminal=[[True],[False]] → [[1],[4]]。',
        ['从最后一行向前递推，当前累计量初始为 bootstrap。', 'mask 乘的是下一步累计回报，不是当前 reward。'],
        ['终止步的 reward 也被清零。', '忽略 bootstrap，或沿 batch 而非时间逆序。'],
        ['有限采样窗口与真实 episode 终止有什么不同？', 'γ 接近 1 会怎样改变有效回报跨度？'],
        '''
        import torch
        def discounted_returns(rewards, terminated, bootstrap, gamma=0.99):
            running = bootstrap
            result = []
            for t in range(rewards.shape[0] - 1, -1, -1):
                running = rewards[t] + gamma * (~terminated[t]).to(rewards.dtype) * running
                result.append(running)
            return torch.stack(result[::-1])
        ''', sources=('terminal', 'gae'), minutes=20)

    put('td0_target', 'TD(0) · 单步 Bootstrap Target', '强化学习 · 回报与优势', '基础',
        'td0_target(rewards, next_values, terminated, gamma=0.99)',
        '计算固定的单步 TD target，区分当前奖励与下一状态价值；target 作为数据，不在本题训练 value 网络。',
        'target = rewards + γ·(1−terminated)·next_values',
        ['rewards、next_values 为同 shape 非空浮点 Tensor，可为标量；terminated 是同 shape bool。',
         '0≤gamma≤1；所有输入有限；返回同 shape/dtype/device Tensor；只核验数值，不修改输入。',
         'terminated=True 时不 bootstrap；truncation 不等同终止，调用者必须传真实终态的 next_values，而不是 reset 后新状态价值。'],
        '例 1：r=[1,2], next_v=[10,20], terminal=[False,True], γ=0.5 → [6,2]。\n例 2：γ=0 时输出等于 rewards，与 next_values 无关。',
        ['使用 ~terminated 构造保留下一状态价值的 mask。', '广播前确认输入 shape 完全一致，不能按 batch 重复单个 target。'],
        ['给 terminal 状态加上下一状态价值。', '把奖励本身也乘 gamma，或误把 truncated 当 terminated。'],
        ['TD target 与 Monte Carlo return 的偏差/方差有何差别？', '为什么目标网络的输出通常不参与当前网络反向传播？'],
        '''
        def td0_target(rewards, next_values, terminated, gamma=0.99):
            return rewards + gamma * next_values * (~terminated).to(rewards.dtype)
        ''', sources=('terminal',), prerequisites=('discounted_returns',), minutes=15)

    put('gae_advantages', 'GAE · 终止与截断分离', '强化学习 · 回报与优势', '进阶',
        'gae_advantages(rewards, values, next_values, terminated, truncated, gamma=0.99, lam=0.95)',
        '实现带明确 episode 边界的 GAE。TD residual 可以在时间截断处 bootstrap，但 advantage trace 不能流进 reset 后的新 episode。',
        'δ_t=r_t+γ·(1−terminated_t)·next_values_t−values_t\nA_T=0；A_t=δ_t+γλ·(1−terminated_t)·(1−truncated_t)·A_{t+1}\nreturns_t=A_t+values_t',
        ['rewards/values/next_values：[T,B] 同 dtype/device 浮点；terminated/truncated：[T,B] bool；T,B≥1。',
         '0≤gamma,lam≤1；两个标志可同时 True，此时终止优先：既不 bootstrap，也不延续 trace。',
         'next_values[t] 是该 transition 的真实下一状态价值；截断处必须用 final observation，不能拿 reset 后的 values[t+1] 替代。',
         '采样窗口末尾 trace 初始为零，但最后一步 δ 仍可使用 next_values；返回 tuple (advantages, returns)，均 [T,B]。',
         '本题是 rollout 数据预处理，全部输入固定，只数值评测；不做优势归一化。'],
        '例 1：r=[[1],[2]], v=next_v=0, 无边界, γ=1, λ=0.5 → A=[[2],[2]], returns同A。\n例 2：单步 r=1,v=2,next_v=4,γ=0.5：truncated=True→A=1；terminated=True→A=-1。',
        ['用 terminated 单独控制 δ 的 bootstrap。', '用 terminated OR truncated 控制递推链。', '返回 value target 时把当前 values 加回来。'],
        ['把截断与终止都用 done 禁止 bootstrap。', '截断处不断 trace，混入下一 episode 的优势。', '把 returns 误写成只有 advantages。'],
        ['λ=0 和 λ=1 分别对应怎样的估计？', '自动 reset 的 vector environment 如何提供 final observation？'],
        '''
        import torch
        def gae_advantages(rewards, values, next_values, terminated, truncated, gamma=0.99, lam=0.95):
            delta = rewards + gamma * next_values * (~terminated).to(values.dtype) - values
            continuation = (~(terminated | truncated)).to(values.dtype)
            running = torch.zeros_like(values[0])
            result = []
            for t in range(rewards.shape[0] - 1, -1, -1):
                running = delta[t] + gamma * lam * continuation[t] * running
                result.append(running)
            advantages = torch.stack(result[::-1])
            return advantages, advantages + values
        ''', sources=('gae', 'terminal'), prerequisites=('td0_target',), minutes=35)

    put('normalize_advantages', '优势归一化 · 总体方差', '强化学习 · 回报与优势', '基础',
        'normalize_advantages(advantages, eps=1e-8)',
        '按本批全部元素标准化优势，固定总体方差与 epsilon 位置，避免单元素或常量优势出现 NaN。',
        'μ=mean(A)；σ=sqrt(mean((A−μ)²))；output=(A−μ)/(σ+eps)',
        ['advantages：任意 shape 非空有限浮点 Tensor；eps>0，位于标准差外；返回同 shape/dtype/device。',
         '对全部元素求均值和方差，不按行、列或单个 episode 单独归一化；方差分母 N，不是 N−1。',
         '常量或单元素输入输出零；仅数值评测，此题把 advantage 看作固定训练数据。'],
        '例 1：A=[1,3],eps=1 → [-0.5,0.5]。\n例 2：A=[[7,7],[7,7]] 或 A=7 → 对应 shape 的全零。',
        ['不传 dim 时对全部元素求均值。', '先中心化，再求 mean(square)，最后 sqrt 后加 eps。'],
        ['torch.std 默认使用 N−1 的口径。', '把 eps 加入方差内部，或只沿 batch 维标准化。'],
        ['优势归一化为何不等于给 reward 做标准化？', '按完整 rollout 与按 minibatch 归一化会有什么差别？'],
        '''
        def normalize_advantages(advantages, eps=1e-8):
            centered = advantages - advantages.mean()
            std = centered.square().mean().sqrt()
            return centered / (std + eps)
        ''', sources=('ppo_impl',), prerequisites=('gae_advantages',), minutes=15)

    put('reinforce_loss', 'REINFORCE · 已采样动作损失', '强化学习 · 策略优化', '进阶',
        'reinforce_loss(logits, actions, returns)',
        '从离散策略 logits 提取已采样动作的 log probability，用固定回报构建可最小化的 REINFORCE 损失。',
        'L = −mean_b[returns_b · log softmax(logits_b)[actions_b]]',
        ['logits：[B,A]；actions：[B] long，范围 [0,A)；returns：[B] 同 dtype/device 浮点，可为负；B,A≥1。',
         '返回标量 Tensor；只检查 logits 梯度；actions/returns 固定，不在本题采样动作。',
         '不对 returns 归一化，不加 baseline、熵或额外时间折扣；支持 ±1000 logits；均值分母是 B。'],
        '例 1：logits=[[0,0]], action=[1], return=[2] → loss=2ln2。\n例 2：同 logits/action，return=[-2] → loss=-2ln2，梯度方向反转。',
        ['先计算稳定 log-softmax，再 gather 动作列。', '负号把最大化回报对应的目标转换为最小化损失。'],
        ['对动作概率而非 log probability 加权。', '遗漏负号或把所有动作一起求平均。'],
        ['不依赖动作的 baseline 为什么能降低方差？', '为何传入的是采样动作，而不是 argmax 动作？'],
        '''
        import torch
        def reinforce_loss(logits, actions, returns):
            shifted = logits - logits.amax(-1, keepdim=True)
            log_probs = shifted - torch.logsumexp(shifted, dim=-1, keepdim=True)
            chosen = log_probs.gather(1, actions[:, None]).squeeze(-1)
            return -(chosen * returns).mean()
        ''', mode='autograd', sources=('distribution',), prerequisites=('softmax', 'discounted_returns'))

    put('categorical_entropy', 'Categorical Entropy · 离散策略熵', '强化学习 · 策略优化', '基础',
        'categorical_entropy(logits)',
        '计算每条离散策略分布的熵，为探索正则提供基础组件；输出每行结果，不在此题跨 batch 平均。',
        'H_b = −Σ_a p_ba log p_ba；p=softmax(logits)，log 为自然对数',
        ['logits：[B,A] 非空有限浮点，B,A≥1；返回 [B]，保持 dtype/device。',
         '只对 logits 求导；支持 ±1000；先算稳定 log-softmax，再用其 exp 得到 p，避免 0*log(0)。',
         '返回正熵 H，不是训练损失中的负熵；A=1 时熵及梯度为零。'],
        '例 1：logits=[[0,0],[0,0]] → [ln2,ln2]。\n例 2：只有一个动作 logits=[[1000]] → [0]。',
        ['logits 逐行减最大值，再用 logsumexp。', '归约动作维；保留 batch 维。'],
        ['符号反了，或输出 batch 平均标量。', '先算概率再 log，在极端分布出现 NaN。'],
        ['均匀分布为何具有最大熵？', '策略熵与到参考策略的 KL 有什么区别？'],
        '''
        import torch
        def categorical_entropy(logits):
            shifted = logits - logits.amax(-1, keepdim=True)
            log_probs = shifted - torch.logsumexp(shifted, dim=-1, keepdim=True)
            return -(log_probs.exp() * log_probs).sum(-1)
        ''', mode='autograd', sources=('distribution',), prerequisites=('softmax',), minutes=20)

    put('importance_sampling_ratio', 'Importance Ratio · Log 概率之差', '强化学习 · 策略优化', '基础',
        'importance_sampling_ratio(new_log_probs, old_log_probs)',
        '由同一批已采样动作的新旧 log probability 计算逐样本比率，避免分别 exp 后相除造成下溢。',
        'ratio = π_new(a|s)/π_old(a|s) = exp(new_log_probs−old_log_probs)',
        ['两个输入严格同 shape 的非空浮点 Tensor，可为标量；都为有限非正数；|new−old|≤30。',
         '返回同 shape/dtype/device；只检查 new_log_probs 梯度，old_log_probs 是固定行为策略数据。',
         '绝对 log probability 可低至 -1000；不做 clipping、mean 或额外归一化。'],
        '例 1：new=log(0.4),old=log(0.2) → 2。\n例 2：new=old=-1000 → 1；先各自 exp 会得到 0/0。',
        ['先相减，再只调用一次 exp。', 'old 是常量，不把这一步当作同时训练两套策略。'],
        ['把比率方向写成 old/new。', '在 ratio 阶段提前裁剪到 [0.8,1.2]。'],
        ['为什么极大的 importance weight 会带来高方差？', '逐动作 ratio 与整段轨迹 ratio 的关系是什么？'],
        '''
        def importance_sampling_ratio(new_log_probs, old_log_probs):
            return (new_log_probs - old_log_probs).exp()
        ''', mode='autograd', sources=('ppo',), prerequisites=('reinforce_loss',), minutes=15)

    put('ppo_clipped_loss', 'PPO Actor · Clipped Surrogate', '强化学习 · 策略优化', '进阶',
        'ppo_clipped_loss(new_log_probs, old_log_probs, advantages, clip_epsilon=0.2)',
        '实现 PPO actor 的悲观 clipped surrogate，兼顾正负优势的裁剪方向；本题只返回 actor loss。',
        'r=exp(new_logp−old_logp)；L=−mean[min(r·A, clip(r,1−ε,1+ε)·A)]',
        ['new_log_probs、old_log_probs、advantages：[B] 同 dtype/device，B≥1；log概率有限非正，|new−old|≤30。',
         '0<clip_epsilon<1；advantages 可正可负或零；返回标量，只检查 new_log_probs 梯度。',
         'old_log_probs 和 advantages 固定；不归一化优势，不加 value loss、entropy、KL；只裁剪比率分支，不能只返回 clipped 分支。',
         '分支相等时采用 torch.minimum 的平均子梯度；clamp 边界使用 PyTorch 的子梯度约定。'],
        '例 1：r=[1.5],A=[2],ε=0.2 → loss=-2.4（正优势上限）。\n例 2：r=[0.5],A=[-2],ε=0.2 → loss=1.6（负优势下限）。',
        ['分别保留原始 ratio 和裁剪后的 ratio。', '先各自乘 advantage，再做 minimum，最后负号和 mean。'],
        ['用 maximum，或遗漏最小化目标的负号。', '只 clamp ratio 后乘优势，错误裁掉本应惩罚的方向。'],
        ['为什么正负优势的裁剪方向不同？', 'PPO clipping 是否严格保证 KL 不超过某个阈值？'],
        '''
        import torch
        def ppo_clipped_loss(new_log_probs, old_log_probs, advantages, clip_epsilon=0.2):
            ratio = (new_log_probs - old_log_probs).exp()
            clipped = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon)
            return -torch.minimum(ratio * advantages, clipped * advantages).mean()
        ''', mode='autograd', sources=('ppo',), prerequisites=('importance_sampling_ratio', 'gae_advantages'), minutes=30)

    put('ppo_value_loss', 'PPO Value · 明确裁剪变体', '强化学习 · 策略优化', '进阶',
        'ppo_value_loss(new_values, old_values, returns, clip_epsilon=0.2)',
        '实现 PPO 常见的 value clipping 变体，采用 OpenAI Baselines 的双误差取较大值口径；不是所有 PPO 实现的统一定义。',
        'v_clip=v_old+clip(v_new−v_old,−ε,ε)\nL=1/2·mean[max((v_new−R)²,(v_clip−R)²)]',
        ['new_values、old_values、returns：[B] 同 dtype/device 有限浮点，B≥1；clip_epsilon>0。',
         '返回标量；只检查 new_values 梯度；旧预测与 return target 固定；不对 returns 归一化。',
         '这里裁剪的是 value 的绝对变化，不是 policy ratio；epsilon 与 reward/value 的尺度相关。',
         '相等分支使用 torch.maximum 的平均子梯度；无额外 value coefficient。'],
        '例 1：new=old=0,R=2,ε=0.2 → loss=2。\n例 2：new=1,old=0,R=2,ε=0.2 → max(1,3.24)/2=1.62。',
        ['以 old_values 为中心裁剪变化量，再加回 old_values。', '比较两个平方误差后取 max，最后乘 0.5。'],
        ['只计算裁剪后的误差，或取 min。', '裁剪 value 自身，而不是相对 old_values 的变化。'],
        ['为什么 reward scaling 会影响这个裁剪变体？', 'value clipping 和 policy clipping 能否共用相同的数值尺度？'],
        '''
        import torch
        def ppo_value_loss(new_values, old_values, returns, clip_epsilon=0.2):
            clipped = old_values + (new_values - old_values).clamp(-clip_epsilon, clip_epsilon)
            raw_error = (new_values - returns).square()
            clipped_error = (clipped - returns).square()
            return 0.5 * torch.maximum(raw_error, clipped_error).mean()
        ''', priority='P1', mode='autograd', sources=('ppo_impl',), prerequisites=('mse_loss', 'ppo_clipped_loss'))

    put('dqn_loss', 'DQN · 固定 Target 的 Huber Loss', '强化学习 · 价值学习', '进阶',
        'dqn_loss(q_values, actions, rewards, next_values, terminated, gamma=0.99)',
        '提取当前网络已执行动作的 Q 值，与固定 bootstrap target 比较，使用 delta=1 的 Huber 损失。',
        'y=r+γ(1−terminal)next_value；e=Q(s,a)−y\nHuber(e)=0.5e²（|e|≤1），否则 |e|−0.5；L=mean(Huber)',
        ['q_values：[B,A]；actions：[B] long 合法动作；rewards/next_values：[B]；terminated：[B] bool；B,A≥1。',
         '0≤gamma≤1；next_values 已由调用方的目标网络求出（如 max_a Q_target(s′,a)），不在本题再次选动作。',
         '返回标量；只检查 q_values 梯度，奖励、next_values、动作与标志固定；无 replay buffer、target 更新或训练循环。'],
        '例 1：Q=[[0,5]],a=[0],r=[2],terminal=True → |e|=2，loss=1.5。\n例 2：Q=[[0.5]],a=[0],r=[0],next=0 → loss=0.125。',
        ['先 gather 当前动作列。', 'target 不含当前 q_values；再按绝对误差分段求 Huber。'],
        ['使用当前最大 Q 而非 actions 指定的 Q。', '误用 MSE，或在 terminal 处继续 bootstrap。'],
        ['Huber 相比 MSE 对离群 TD error 有什么变化？', '为何目标网络与经验回放是完整 DQN 的另外两个组件？'],
        '''
        import torch
        def dqn_loss(q_values, actions, rewards, next_values, terminated, gamma=0.99):
            target = rewards + gamma * (~terminated).to(rewards.dtype) * next_values
            chosen = q_values.gather(1, actions[:, None]).squeeze(-1)
            error = chosen - target
            losses = torch.where(error.abs() <= 1, 0.5 * error.square(), error.abs() - 0.5)
            return losses.mean()
        ''', mode='autograd', sources=('dqn',), prerequisites=('td0_target', 'mse_loss'))

    put('double_dqn_target', 'Double DQN · Online 选、Target 评', '强化学习 · 价值学习', '进阶',
        'double_dqn_target(rewards, online_next_q, target_next_q, terminated, gamma=0.99)',
        '把下一动作选择与价值评估分开：online 网络选择 argmax，target 网络评估那个动作。这里只返回固定 target。',
        'a*=argmax_a Q_online(s′,a)；y=r+γ(1−terminal)Q_target(s′,a*)',
        ['online_next_q、target_next_q：[B,A]；rewards：[B]；terminated：[B] bool；B,A≥1，0≤gamma≤1。',
         '返回 [B]，dtype/device 与 rewards 相同；所有浮点输入有限；只评数值，不求网络梯度。',
         'online 最大值并列时取最小动作索引；不能用 target argmax 选动作；禁止修改输入。'],
        '例 1：online=[[5,1]],target=[[2,10]],r=[1],γ=0.5,非终止 → [2]，不是 [6]。\n例 2：online=[[3,3]],target=[[2,10]] → 选择动作0；若终止则只返回 r。',
        ['先在 online_next_q 的动作维 argmax。', '用该索引 gather target_next_q，再加 reward 和终止 mask。'],
        ['直接 target_next_q.max，退回普通 DQN target。', '用 online 网络既选又评。'],
        ['max 操作为什么可能引入过估计？', 'Double DQN 是否需要两个独立在线优化器？'],
        '''
        def double_dqn_target(rewards, online_next_q, target_next_q, terminated, gamma=0.99):
            actions = online_next_q.argmax(dim=-1)
            evaluated = target_next_q.gather(1, actions[:, None]).squeeze(-1)
            return rewards + gamma * (~terminated).to(rewards.dtype) * evaluated
        ''', priority='P1', sources=('ddqn',), prerequisites=('dqn_loss',))

    put('polyak_update', 'Polyak Update · 目标参数软更新', '强化学习 · 价值学习', '基础',
        'polyak_update(online, target, tau=0.005)',
        '用明确的 tau 方向产生新的 target 参数；函数返回新 Tensor，不原地写入网络。',
        'target_new = (1−tau)·target + tau·online',
        ['online、target 为相同非空 shape/dtype/device 有限浮点 Tensor，允许标量、非连续张量；0≤tau≤1。',
         '返回同 shape Tensor，仅核验数值；online/target 都不可被原地修改。',
         'tau 是新 online 参数的比例：tau=0 保持旧 target，tau=1 完全复制 online；这里不用有些文献中的保留系数 rho 命名。'],
        '例 1：online=[10],target=[2],tau=0.25 → [4]。\n例 2：tau=0 → target；tau=1 → online。',
        ['把 tau=0 和 tau=1 代入公式检查方向。', '用表达式创建新输出，不调用输入的 mul_ / add_。'],
        ['把 tau 当成旧 target 的比例。', '直接覆盖 target，破坏调用方状态。'],
        ['硬更新每隔 K 步复制，与软更新有什么差别？', 'BatchNorm 等非浮点状态是否应照同一公式更新？'],
        '''
        def polyak_update(online, target, tau=0.005):
            return (1 - tau) * target + tau * online
        ''', sources=('dqn',), prerequisites=('dqn_loss',), minutes=15)

    put('value_iteration_step', 'Value Iteration · Bellman 最优备份', '强化学习 · 价值学习', '进阶',
        'value_iteration_step(values, transition, rewards, terminated, gamma=0.99)',
        '对已知小型 MDP 执行一次同步 Bellman optimality backup，先对下一状态求期望，再对动作取最大值。',
        'Q(s,a)=Σ_s′ P(s′|s,a)[R(s,a,s′)+γ(1−terminal(s,a,s′))V_old(s′)]\nV_new(s)=max_a Q(s,a)',
        ['values：[S]；transition/rewards：[S,A,S]；terminated：[S,A,S] bool；S,A≥1，0≤gamma≤1。',
         'transition 非负，每个 [s,a,:] 和为1；reward 可以依赖完整 transition；terminal 是该 transition 是否结束。',
         '返回 [S] 浮点；只做一次同步更新，所有状态都读同一份旧 values；仅数值评测，不返回 long 动作索引。',
         '不包含求解到收敛的外循环、环境采样或策略评估；不原地修改输入。'],
        '例 1：S=1,A=1,P=1,R=2,V_old=4,γ=0.5,非终止 → V_new=4。\n例 2：S=2,A=2，所有 transition 终止；P=[[[0.5,0.5],[1,0]],[[0,1],[0.25,0.75]]]，R=[[[2,4],[2,0]],[[0,5],[8,0]]] → Q=[[3,2],[5,2]]，V_new=[3,5]，与 V_old 无关。',
        ['values 广播到最后一个 next-state 维。', '先乘 transition 并沿最后维求和，再沿动作维 max。'],
        ['先对 next-state 取 max，而不是概率期望。', '逐个状态覆盖 values，误变为异步更新。'],
        ['为什么这是 optimality backup，而不是固定策略的 evaluation backup？', 'gamma=1 时迭代收敛需要什么额外条件？'],
        '''
        def value_iteration_step(values, transition, rewards, terminated, gamma=0.99):
            target = rewards + gamma * (~terminated).to(values.dtype) * values[None, None, :]
            q = (transition * target).sum(dim=-1)
            return q.max(dim=-1).values
        ''', priority='P1', sources=('vi',), prerequisites=('td0_target',), minutes=30)

    put('c51_projection', 'C51 · 均匀支撑上的分布投影', '强化学习 · 分布与语言策略', '挑战',
        'c51_projection(next_probabilities, rewards, terminated, support, gamma=0.99)',
        '把下一状态的离散回报分布经 Bellman 变换后投影回均匀支撑。只实现 categorical projection，可用任意 N≥2，而非固定51个原子。',
        'z′_j=clip(r+γ(1−terminal)z_j,z_min,z_max)，b_j=(z′_j−z_min)/Δz\nl=floor(b)，u=ceil(b)；质量按(u−b,b−l)分给l/u；l=u时全部留在该atom。',
        ['next_probabilities：[B,N] 非负，每行和为1；rewards：[B]；terminated：[B] bool；support：[N] 严格递增、等间距；B≥1,N≥2。',
         '所有浮点同 dtype/device 且有限，0≤gamma≤1；返回 [B,N]，非负且每行质量和约为1。',
         '超出支撑的回报裁到两端；terminal 时未来分布被折叠到 reward 的位置，再按邻居插值。',
         '仅数值评测，不求 support 或投影索引的梯度；动作选择与交叉熵训练不在本题范围。'],
        '例 1：support=[-1,0,1],p=[0.2,0.3,0.5],r=0,γ=1,非终止 → 原分布不变。\n例 2：同support，terminal=True,r=0.5 → [0,0.5,0.5]；r=3 → [0,0,1]。',
        ['计算变换后的位置并裁剪，随后转成分数索引 b。', '用 scatter_add 累加多个来源 atom 的质量。', '单独处理 floor=ceil；两个插值权重都为零会丢掉整份质量。'],
        ['恰落 atom 时质量消失。', '直接四舍五入，失去线性插值。', '忘记 terminal 或两端裁剪。'],
        ['固定 support 太窄会损失哪些分布信息？', '为何先选动作与直接混合所有动作分布是不同算法？'],
        '''
        import torch
        def c51_projection(next_probabilities, rewards, terminated, support, gamma=0.99):
            step = (support[-1] - support[0]) / (support.numel() - 1)
            shifted = rewards[:, None] + gamma * (~terminated).to(rewards.dtype)[:, None] * support[None, :]
            shifted = shifted.clamp(support[0], support[-1])
            position = ((shifted - support[0]) / step).clamp(0, support.numel() - 1)
            lower, upper = position.floor().long(), position.ceil().long()
            low_weight = upper.to(position.dtype) - position + (lower == upper).to(position.dtype)
            high_weight = position - lower.to(position.dtype)
            output = torch.zeros_like(next_probabilities)
            output = output.scatter_add(1, lower, next_probabilities * low_weight)
            return output.scatter_add(1, upper, next_probabilities * high_weight)
        ''', priority='P2', sources=('c51',), prerequisites=('dqn_loss',), minutes=40)

    put('grpo_advantages', 'GRPO · 同 Prompt 组内优势', '强化学习 · 分布与语言策略', '进阶',
        'grpo_advantages(rewards, eps=1e-6)',
        '对同一 prompt 的多条回答奖励进行组内标准化，得到 outcome-supervision 的序列优势。此题显式指定总体标准差及 epsilon。',
        'μ_b=mean_g r_bg；σ_b=sqrt(mean_g (r_bg−μ_b)²)；A_bg=(r_bg−μ_b)/(σ_b+eps)',
        ['rewards：[B,G] 有限浮点；B 为 prompt 数、G 为每个 prompt 的回答数，B,G≥1；eps>0。',
         '返回 [B,G]；每个 prompt 独立计算；方差分母为 G，eps 在标准差外；常量组与 G=1 返回零。',
         '全部为固定奖励数据，只数值评测；不训练 reward model、critic 或策略；不采用 process-reward 累加。',
         '总体标准差、eps 和退化组约定属于本练习口径，不代表所有名为 GRPO 的实现。'],
        '例 1：rewards=[[1,3],[10,10]],eps=1 → [[-0.5,0.5],[0,0]]。\n例 2：每组只有一条回答 rewards=[[2],[7]] → [[0],[0]]。',
        ['只沿组内 G 维求统计量，并 keepdim=True。', '先中心化再平方平均；不要默认使用无偏标准差。'],
        ['把不同 prompt 混在一起标准化。', '用 G−1 导致单样本组 NaN，或忘记 eps。'],
        ['没有 critic 时，组均值如何充当 baseline？', '奖励全相同的 prompt 为何没有相对优势信号？'],
        '''
        def grpo_advantages(rewards, eps=1e-6):
            centered = rewards - rewards.mean(dim=1, keepdim=True)
            std = centered.square().mean(dim=1, keepdim=True).sqrt()
            return centered / (std + eps)
        ''', priority='P1', sources=('grpo',), prerequisites=('normalize_advantages',), minutes=20)

    put('grpo_loss', 'GRPO · Masked Token 策略目标', '强化学习 · 分布与语言策略', '挑战',
        'grpo_loss(new_log_probs, old_log_probs, reference_log_probs, advantages, mask, clip_epsilon=0.2, beta=0.04)',
        '实现 GRPO outcome-supervision 的明确教学损失：固定序列优势、token PPO clipping 与 k3 KL 项。每条有效回答等权，不把长回答自动赋予更大权重。',
        'r_it=exp(new_it−old_it)，s_it=min(r_it A_i,clip(r_it,1−ε,1+ε)A_i)\nd_it=ref_it−new_it；k3_it=exp(d_it)−d_it−1\nL=mean_{i:有效长度>0}[mean_{t:mask_it}(-s_it+β·k3_it)]',
        ['new/old/reference_log_probs：[B,T] 同 dtype/device 的有限非正 log概率；advantages：[B] 固定序列优势；mask：[B,T] bool，True为有效生成token；B,T≥1。',
         '只检查 new_log_probs 梯度；old/reference/advantages 都是固定数据。所有位置均保证 |new−old|≤30、|ref−new|≤30；0<ε<1，beta≥0。',
         '先对每条序列有效 token 平均，再对至少有一个有效 token 的序列平均；全 padding 序列不进入外层分母。',
         '整个 batch 全 padding 时返回与 new_log_probs 保持连接的标量零；masked token 的梯度必须为零。',
         'KL 项可用 expm1(d)−d 稳定计算；不把 KL 放入优势，不反向传播旧策略；不加入其他奖励、熵、长度修正或采样过程。',
         '这是固定归约与 KL 估计器的教学版本，不是完整大模型 RL 训练器，也不把有限样本的 k3 值宣称为精确分布 KL。'],
        '例 1：new=old=ref，β任意，A=[1,-1]，两行均有有效token → loss=0。\n例 2：β=0，ratio=1，A=[2,-1]，有效长度=[1,3] → (-2+1)/2=-0.5；不能按4个token平均。',
        ['把 advantages[:,None] 广播到 token 维。', '分别计算 clipped surrogate 和 expm1(ref-new)−(ref-new)。', '用 mask 求各序列和，除以有效长度；再排除全 padding 行，外层零分母用1保护。'],
        ['对全部有效 token 直接平均，改变长度权重。', 'KL 的 log比方向反了，或把 min 写成 max。', '全 padding 返回独立新零 Tensor，使计算图断开。'],
        ['为什么不同长度归约会改变训练权重？', 'reference policy 与 old rollout policy 分别承担什么角色？', '单个 token 的 k3 样本与真实 KL 的期望关系需要哪些采样假设？'],
        '''
        import torch
        def grpo_loss(new_log_probs, old_log_probs, reference_log_probs, advantages, mask, clip_epsilon=0.2, beta=0.04):
            ratio = (new_log_probs - old_log_probs).exp()
            clipped = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon)
            advantage = advantages[:, None]
            surrogate = torch.minimum(ratio * advantage, clipped * advantage)
            difference = reference_log_probs - new_log_probs
            kl = torch.expm1(difference) - difference
            token_loss = -surrogate + beta * kl
            valid = mask.to(new_log_probs.dtype)
            lengths = valid.sum(dim=1)
            per_sequence = (token_loss * valid).sum(dim=1) / lengths.clamp_min(1)
            return per_sequence.sum() / (lengths > 0).sum().clamp_min(1)
        ''', priority='P2', mode='autograd', sources=('grpo', 'ppo'), prerequisites=('grpo_advantages', 'ppo_clipped_loss'), minutes=45)
