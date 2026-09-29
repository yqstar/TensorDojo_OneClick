"""LLM decoding, training, alignment and optimization exercise authoring."""


def register(add):
    sources = {
        'sampling': ('Nucleus / Top-k 原论文', 'https://arxiv.org/html/1904.09751v2'),
        'repetition': ('Hugging Face RepetitionPenaltyLogitsProcessor', 'https://huggingface.co/docs/transformers/v4.46.3/en/internal/generation_utils#transformers.RepetitionPenaltyLogitsProcessor'),
        'causal': ('Hugging Face Causal Language Modeling', 'https://huggingface.co/docs/transformers/en/tasks/language_modeling'),
        'distill': ('PyTorch 知识蒸馏教程', 'https://docs.pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html'),
        'dpo': ('DPO 原论文与附录实现', 'https://arxiv.org/html/2305.18290v3'),
        'adamw': ('Decoupled Weight Decay Regularization', 'https://arxiv.org/html/1711.05101v3'),
        'accumulation': ('Accelerate 梯度累积与不同有效样本数', 'https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation'),
        'perplexity': ('Hugging Face Perplexity 定义', 'https://huggingface.co/docs/transformers/en/perplexity'),
    }

    def put(pid, title, topic, level, signature, summary, formula, contract,
            example, hints, pitfalls, followups, solution, *, source,
            priority='P0', check_mode='autograd', prerequisites=(), minutes=20):
        title_, url = sources[source]
        add(pid, title, topic, level, signature, summary, formula, contract,
            example, hints, pitfalls, followups, solution,
            sources=({'title': title_, 'url': url},), track='llm', priority=priority,
            check_mode=check_mode, prerequisites=prerequisites, minutes=minutes)

    put('temperature_softmax', 'Temperature Softmax · 温度分布', '解码与采样', '基础',
        'temperature_softmax(logits, temperature=1.0, dim=-1)',
        '从模型 logits 计算指定温度下的 token 分布。明确温度作用在归一化之前，并保持数值和梯度稳定。',
        'pᵢ = exp(zᵢ/T) / Σⱼ exp(zⱼ/T)，T>0',
        ['logits：非空、至少一维的有限 float32/float64 Tensor；dim 为合法维度，temperature 为正 Python 数值。',
         '返回与 logits 同 shape/dtype/device 的概率，沿 dim 和为 1。只对 logits 求导；不修改输入。',
         '测试含非连续输入和 ±1000；建议先减该维最大 logits 再除温度，避免直接指数上溢。',
         '不抽样 token，不做 Top-k / Top-p 截断；温度是固定超参数。'],
        '例 1：logits=[0,log(4)]，T=2 → [1/3,2/3]\n例 2：logits=[[5,5,5]]，T=0.1 → [[1/3,1/3,1/3]]',
        ['同一行 logits 减去常数不改变分布。', '先缩放 logits，再对指定维度指数归一化。'],
        ['先 softmax 再除以 T，概率不再归一化。', '把除温度写成乘温度，改变温度高低的含义。'],
        ['T→0 与 T→∞ 时分布如何变化？', '温度会改变候选 token 的排序吗？', '低温对 logits 梯度有什么影响？'],
        '''
        import torch

        def temperature_softmax(logits, temperature=1.0, dim=-1):
            shifted = (logits - logits.amax(dim=dim, keepdim=True)) / temperature
            numerator = shifted.exp()
            return numerator / numerator.sum(dim=dim, keepdim=True)
        ''', source='sampling', prerequisites=('softmax',), minutes=15)

    put('topk_sampling_probs', 'Top-k · 截断采样分布', '解码与采样', '基础',
        'topk_sampling_probs(logits, k, temperature=1.0)',
        '保留每行得分最高的 k 个 token，再归一化成可用于抽样的概率。只计算分布，不调用随机采样。',
        'K=min(k,V)，S 为稳定降序前 K 个索引；pⱼ=1[j∈S] exp(zⱼ/T)/Σₗ∈S exp(zₗ/T)',
        ['logits：[B,V] 有限浮点，B,V>0；整数 k>=1，temperature>0。',
         '同分按原 token 索引升序；每行恰好保留 min(k,V) 个位置，不能因为阈值并列而多保留。',
         '返回 [B,V] 同 dtype/device 的有限概率，未选位置为 0，每行和为 1。检查数值与输入契约，不检查排序的梯度。',
         'k>=V 退化为温度 softmax；允许内部用 −inf 屏蔽，但不要返回含 −inf 的 logits。'],
        '例 1：logits=[[0,log(2),log(4)]]，k=2，T=1 → [[0,1/3,2/3]]\n例 2：logits=[[1,1,1]]，k=1 → [[1,0,0]]',
        ['使用 descending=True、stable=True 排序，明确同分行为。', '在被选 logits 内做稳定 softmax，再 scatter 回原词表位置。'],
        ['用 logits>=第k大值 会多保留边界同分项。', '截断后忘记重新归一化。', '输出排序后的概率而未恢复原 token 索引。'],
        ['固定 k 对尖锐与平坦分布的效果有什么区别？', '为什么 Top-k 和温度可在排序阶段交换，但归一化仍需温度？'],
        '''
        import torch

        def topk_sampling_probs(logits, k, temperature=1.0):
            order = torch.argsort(logits, dim=-1, descending=True, stable=True)
            selected = order[:, :min(k, logits.shape[-1])]
            kept = logits.gather(1, selected)
            shifted = (kept - kept.amax(dim=1, keepdim=True)) / temperature
            probabilities = shifted.exp()
            probabilities = probabilities / probabilities.sum(1, keepdim=True)
            return torch.zeros_like(logits).scatter(1, selected, probabilities)
        ''', source='sampling', check_mode='numeric', prerequisites=('softmax',), minutes=20)

    put('topp_sampling_probs', 'Top-p / Nucleus · 累计概率截断', '解码与采样', '进阶',
        'topp_sampling_probs(logits, top_p, temperature=1.0)',
        '按温度分布的累计质量动态选择候选集。保留让累计概率首次达到或超过阈值的 token，随后重新归一化。',
        'q=softmax(z/T)；按 q 降序取最短前缀 S，使 Σⱼ∈S qⱼ≥p；输出 qⱼ/Σₗ∈S qₗ（j∈S），其余为0',
        ['logits：[B,V] 有限浮点，B,V>0；0<top_p<=1，temperature>0。',
         '相等 logits 按原索引升序；根据截断前、加温度后的概率决定前缀，每行至少保留一个 token。',
         '精确等于阈值就停止；保留第一个达到或跨过阈值的 token。top_p=1 明确保留整个词表。',
         '返回同 shape/dtype/device 的有限概率，每行和为 1；只检查数值与输入契约，不抽样、不检查截断梯度。',
         '本题采用论文“最小集合累计≥p”的边界；部分库在累计恰等于 p 时多留一个 token，按本题约定实现。'],
        '例 1：logits=log([[0.5,0.3,0.2]])，p=0.6 → [[0.625,0.375,0]]\n例 2：logits=[[0,0,0,0]]，p=0.5 → [[0.5,0.5,0,0]]',
        ['先稳定排序，再 softmax 和 cumsum。', '第 j 项保留条件是前 j−1 项的累计概率小于 p。', '构造前缀累计时可以在 cumsum 前补一个零，避免减法抵消。'],
        ['只保留 cumsum<=p，丢掉跨阈值的必要 token，甚至产生空集。', '在温度处理前就选 nucleus。', '忽略同分稳定顺序或未 scatter 回原索引。'],
        ['Top-p 为什么比固定 k 更能适应分布的形状？', '同分 token 的边界策略会影响复现吗？', 'T 与 p 同时变化时，候选数量是否一定单调？'],
        '''
        import torch

        def topp_sampling_probs(logits, top_p, temperature=1.0):
            order = torch.argsort(logits, dim=-1, descending=True, stable=True)
            ranked = logits.gather(1, order)
            ranked = (ranked - ranked[:, :1]) / temperature
            probabilities = torch.softmax(ranked, dim=1)
            previous = torch.cat((torch.zeros_like(probabilities[:, :1]),
                                  probabilities.cumsum(1)[:, :-1]), dim=1)
            keep = torch.ones_like(previous, dtype=torch.bool) if top_p == 1 else previous < top_p
            truncated = probabilities * keep.to(probabilities.dtype)
            truncated = truncated / truncated.sum(1, keepdim=True)
            return torch.zeros_like(logits).scatter(1, order, truncated)
        ''', source='sampling', check_mode='numeric', prerequisites=('softmax',), minutes=25)

    put('repetition_penalty', 'Repetition Penalty · 符号与去重', '解码与采样', '进阶',
        'repetition_penalty(logits, token_ids, history_mask, penalty=1.2)',
        '根据每行已有历史，对出现过的 token 调整生成 logits。练习正负号处理、mask 和“每个 ID 只处理一次”。',
        '已出现token：z<0 时 z′=z×penalty，否则 z′=z/penalty；未出现token：z′=z',
        ['logits：[B,V] 浮点；token_ids：[B,L] long；history_mask：[B,L] bool，True 表示有效历史，B,V>0、L>=0。',
         '所有 IDs 都在 [0,V) 内，包括被 mask 的位置；penalty>=1，有限 Python 数值。',
         '每个 token ID 在同一行出现多次也只惩罚一次；不同 batch 行单独处理；本题历史范围完全由输入 mask 定义。',
         '返回同 shape/dtype/device 的有限调整后 logits，不做 softmax；只检查数值，不修改输入。L=0 或全 mask 时数值不变。'],
        '例 1：logits=[[2,-3,0,4]]，IDs=[[0,1,1]]，mask全True，penalty=2 → [[1,-6,0,4]]\n例 2：相同 logits，mask全False → [[2,-3,0,4]]',
        ['先把历史位置合并为 [B,V] 的“是否出现过”布尔标记。', '可以 scatter_add 出次数，再用 >0 去重。', '正数除、负数乘，才能都让该 token 变得更不受偏好。'],
        ['所有 logits 都除 penalty，会把负数变大。', '逐历史位置反复惩罚重复 ID。', '把 padding ID 当作真实历史。'],
        ['该方法与基于次数的 frequency penalty 有什么区别？', '为什么连 prompt token 一起惩罚可能影响输出？', '如果允许 penalty<1，它代表什么？'],
        '''
        import torch

        def repetition_penalty(logits, token_ids, history_mask, penalty=1.2):
            counts = torch.zeros_like(logits, dtype=torch.long)
            counts.scatter_add_(1, token_ids, history_mask.long())
            adjusted = torch.where(logits < 0, logits * penalty, logits / penalty)
            return torch.where(counts > 0, adjusted, logits)
        ''', source='repetition', priority='P1', check_mode='numeric', prerequisites=('softmax',), minutes=20)

    put('causal_lm_loss', 'Causal LM Loss · Shift 与有效 Token', '训练与对齐', '进阶',
        'causal_lm_loss(logits, labels, ignore_index=-100)',
        '手写自回归语言建模损失：当前位置预测下一个 token，按全部有效目标 token 平均，而不是按序列等权。',
        'L = Σ有效(b,t) −log softmax(logits[b,t])[labels[b,t+1]] / N有效，t=0…T−2',
        ['logits：[B,T,V] 有限浮点，B,V>0、T>=1；labels：[B,T] long，值为 [0,V) 或 ignore_index。ignore_index 不在 [0,V) 内。',
         '对齐 logits[:,:-1,:] 与 labels[:,1:]。labels 的首位置、logits 的末位置不参与目标。',
         '忽略等于 ignore_index 的目标；分母是整个 batch 的有效目标数量，不是 B、T 或每条序列均值。',
         '返回标量，dtype/device 与 logits 相同；只检查 logits 梯度。T=1 或全部忽略时，返回与 logits 计算图相连的零。',
         '允许不同 ignore_index；不要用非法负标签直接 gather，也不修改输入。'],
        '例 1：logits=zeros([1,3,2])，labels=[[1,0,1]] → log(2)≈0.693147\n例 2：同样 logits，labels=[[1,-100,-100]] → 0，logits梯度全0',
        ['先把 logits 和 labels 做相反方向的切片。', '把无效标签临时换成合法索引，gather 后再 mask。', '用有效数 clamp_min(1) 归约，可以保留全忽略时的零梯度连接。'],
        ['不 shift 或把 labels 向左错位。', '把被忽略 token 计入分母。', '全忽略时直接创建新的常量零，断开计算图。'],
        ['SFT 中为何常把 prompt 部分标签设为 ignore_index？', 'packing 多条样本时还要约束哪些 attention 边界？', '跨 microbatch 累积时应该按序列数还是有效 token 数加权？'],
        '''
        import torch

        def causal_lm_loss(logits, labels, ignore_index=-100):
            scores, targets = logits[:, :-1], labels[:, 1:]
            valid = targets != ignore_index
            safe_targets = torch.where(valid, targets, torch.zeros_like(targets))
            shifted = scores - scores.amax(dim=-1, keepdim=True)
            log_probs = shifted - torch.logsumexp(shifted, dim=-1, keepdim=True)
            nll = -log_probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
            return (nll * valid.to(logits.dtype)).sum() / valid.sum().clamp_min(1)
        ''', source='causal', prerequisites=('cross_entropy',), minutes=25)

    put('sequence_log_probs', 'Sequence Log Prob · Masked Sum', '训练与对齐', '进阶',
        'sequence_log_probs(logits, labels, mask)',
        '计算每条回答在模型下的序列对数概率，为偏好优化准备输入。明确 token 对齐已由调用方完成，只对有效位置求和。',
        'logp[b] = Σₜ mask[b,t] × log softmax(logits[b,t])[labels[b,t]]',
        ['logits：[B,T,V] 有限浮点，B,V>0、T>=0；labels：[B,T] long；mask：[B,T] bool，True 表示要计入的 token。',
         '本题不额外 shift：logits[b,t] 直接预测 labels[b,t]。调用方可先对因果模型结果做切片。',
         'mask=True 时标签在 [0,V)；mask=False 时允许任意整数哨兵（例如 −100 或 999），不得用这些值直接索引。',
         '返回 [B]，是有效 token 的 log 概率之和，不取长度平均。只检查 logits 梯度；全 mask 或 T=0 的行返回图连零。'],
        '例 1：logits=zeros([1,3,2])，labels=[[0,1,0]]，mask=[[True,True,False]] → [−2log(2)]\n例 2：相同 logits，labels=[[999,-100,999]]，mask全False → [0]',
        ['使用稳定 log_softmax，再沿词表维 gather。', '先用 mask 把非法标签替换成零，再进行索引。', '最后沿时间维 sum，保留 batch 维。'],
        ['自动 shift 一次导致答案错位。', '为了比较不同长度而擅自 mean，改变序列概率定义。', 'gather 后才处理非法 label，已经越界。'],
        ['为什么长序列通常有更小的 log 概率和？', 'DPO 用 sum 与长度归一化版本有何不同？', '如何屏蔽 prompt 而只统计回答？'],
        '''
        import torch

        def sequence_log_probs(logits, labels, mask):
            safe_labels = torch.where(mask, labels, torch.zeros_like(labels))
            shifted = logits - logits.amax(dim=-1, keepdim=True)
            log_probs = shifted - torch.logsumexp(shifted, dim=-1, keepdim=True)
            selected = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
            return (selected * mask.to(logits.dtype)).sum(dim=1)
        ''', source='causal', prerequisites=('cross_entropy',), minutes=20)

    put('kl_distillation_loss', 'KL 蒸馏 · 教师到学生与 T²', '训练与对齐', '进阶',
        'kl_distillation_loss(student_logits, teacher_logits, temperature=2.0)',
        '用冻结教师的软分布监督学生。固定 KL 方向、温度及归约方式，避免把 KL 蒸馏写成对称距离或硬标签 CE。',
        'q=softmax(teacher/T)，p=softmax(student/T)；L=T²/B × ΣᵦΣ𝚌 qᵦ𝚌(log qᵦ𝚌−log pᵦ𝚌)',
        ['student_logits/teacher_logits：[B,C] 有限浮点，B,C>0，同 dtype/device；temperature>0。',
         '教师固定，只检查 student_logits 梯度；先除 T 得到软分布，最后将 KL 乘 T²。',
         '沿 C 求和、沿 B 平均，相当于 batchmean；不再除 C，不包含任何硬标签交叉熵项。',
         '返回标量。包含极端 logits，使用稳定 log 概率；C=1 或两路相同分布时损失为零（浮点容忍内）。'],
        '例 1：student=teacher=[[1,2,3]]，T=2 → 0\n例 2：student=[[0,0]]，teacher=[[log(3),0]]，T=1 → 0.75log(1.5)+0.25log(0.5)≈0.130812',
        ['先分别计算两路稳定 log_softmax。', '用教师 log 概率 exp 得到权重 q，保持 KL(teacher||student) 的方向。', 'T² 是目标定义的一部分，不是只影响数值显示。'],
        ['把 KL 方向颠倒或直接算两路 logits 的 MSE。', '漏乘 T² 或 mean 覆盖所有 B×C 元素。', '教师 softmax 下溢到0后再 log，出现 0×−inf。'],
        ['为什么温度升高后常乘 T²？', '软标签相比 argmax 硬标签保留了什么信息？', '加入硬标签 CE 时如何安排两部分权重？'],
        '''
        import torch

        def kl_distillation_loss(student_logits, teacher_logits, temperature=2.0):
            def log_distribution(x):
                z = (x - x.amax(dim=1, keepdim=True)) / temperature
                return z - torch.logsumexp(z, dim=1, keepdim=True)
            log_p = log_distribution(student_logits)
            log_q = log_distribution(teacher_logits)
            per_row = (log_q.exp() * (log_q - log_p)).sum(dim=1)
            return per_row.mean() * temperature ** 2
        ''', source='distill', priority='P1', prerequisites=('cross_entropy',), minutes=25)

    put('dpo_loss', 'DPO · 策略与参考的偏好差', '训练与对齐', '综合',
        'dpo_loss(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta=0.1)',
        '给定策略和冻结参考模型的回答 log 概率，手写原始 sigmoid DPO 目标。无需加载或生成模型，专注偏好差、参考校正与稳定梯度。',
        'Δ=(logπ(chosen)−logπ(rejected))−(logπref(chosen)−logπref(rejected))；L=mean(−logσ(βΔ))',
        ['四个 Tensor 均为 [B] 有限浮点、B>0、同 dtype/device；输入为已按回答有效 token 求和的 log 概率，不是概率或 logits。',
         'beta>0 固定；只检查 policy_chosen_logps 和 policy_rejected_logps 梯度，两个 reference 输入固定。',
         '返回标量平均损失；只实现原始无 label smoothing 的 sigmoid DPO，不再添加显式 KL、奖励模型或长度归一化。',
         '策略与参考的偏好差相同时，每项损失 log(2)；不修改输入，支持极端正负 βΔ。'],
        '例 1：policy_chosen=[−2]，policy_rejected=[−4]，reference_chosen=[−2]，reference_rejected=[−4] → log(2)\n例 2：policy=[chosen −1,rejected −3]，reference=[−2,−2]，beta=1 → log(1+exp(−2))≈0.126928',
        ['分别先算 policy 的 chosen−rejected 与 reference 的 chosen−rejected。', '使用 logaddexp(0,−beta×差值) 稳定表达负 log-sigmoid。'],
        ['漏掉 reference 差，把 DPO 写成普通奖励 pairwise loss。', 'chosen/rejected 符号颠倒。', '把 beta 当成除数或擅自加一项显式 KL。'],
        ['为什么原始 DPO 不必先拟合独立 reward model？', 'beta 与参考策略约束有什么联系？', '参考差是零时，与 pairwise logistic loss 有什么关系？'],
        '''
        import torch

        def dpo_loss(policy_chosen_logps, policy_rejected_logps,
                     reference_chosen_logps, reference_rejected_logps, beta=0.1):
            policy_gap = policy_chosen_logps - policy_rejected_logps
            reference_gap = reference_chosen_logps - reference_rejected_logps
            scaled_gap = beta * (policy_gap - reference_gap)
            return torch.logaddexp(torch.zeros_like(scaled_gap), -scaled_gap).mean()
        ''', source='dpo', priority='P2', prerequisites=('bce_with_logits',), minutes=25)

    put('reward_pairwise_loss', 'Reward Model · 成对偏好损失', '训练与对齐', '进阶',
        'reward_pairwise_loss(chosen_rewards, rejected_rewards, margin=0.0)',
        '从偏好对训练奖励分数。以 Bradley–Terry 风格 logistic 目标为基础，明确本题增加固定 margin 的教学变体。',
        'L = mean(log(1+exp(rejected−chosen+margin)))',
        ['chosen_rewards/rejected_rewards：[B] 有限浮点，B>0、同 dtype/device；margin>=0 是固定 Python 数值。',
         '每个位置是一对同一 prompt 的偏好回答；分数是任意实数，不先 sigmoid。返回标量平均损失。',
         '同时检查两路 reward 梯度；margin=0 是标准成对 logistic 目标，正 margin 要求 chosen 超过 rejected 更多。',
         '不包含额外 reward centering 正则、reference 模型或 DPO beta；稳定处理 ±1000。'],
        '例 1：chosen=[0,0]，rejected=[0,0]，margin=0 → log(2)\n例 2：chosen=[2]，rejected=[0]，margin=2 → log(2)',
        ['分差正负方向用“chosen 越大，loss 越小”检查。', 'margin 加在 rejected−chosen 上，而不是减去 margin。'],
        ['把奖励先变为概率，丢掉原分差。', 'margin 方向写反。', '直接 exp 大分差导致溢出。'],
        ['奖励整体平移是否改变这个目标？', '只有相对偏好时，奖励的绝对零点能被识别吗？', '与 DPO 输入和 reference 校正有什么区别？'],
        '''
        import torch

        def reward_pairwise_loss(chosen_rewards, rejected_rewards, margin=0.0):
            difference = rejected_rewards - chosen_rewards + margin
            return torch.logaddexp(torch.zeros_like(difference), difference).mean()
        ''', source='dpo', priority='P1', prerequisites=('bpr_loss',), minutes=20)

    put('adamw_step', 'AdamW · 解耦权重衰减', '训练优化与评估', '进阶',
        'adamw_step(param, grad, exp_avg, exp_avg_sq, step, lr, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01)',
        '在已有 Adam 更新的基础上实现解耦 weight decay。显式传入状态，核对衰减位置、偏差修正与新旧状态。',
        'm′=β₁m+(1−β₁)g；v′=β₂v+(1−β₂)g²；p′=(1−lr×wd)p−lr×[m′/(1−β₁^step)]/[√(v′/(1−β₂^step))+eps]',
        ['param/grad/exp_avg/exp_avg_sq：同 shape/dtype/device 的有限浮点，非空；exp_avg_sq>=0。',
         'step 为当前更新步数（整数>=1）；lr>=0，0<=beta1,beta2<1，eps>0，weight_decay>=0，均为固定 Python 数值。',
         '返回 (new_param,new_exp_avg,new_exp_avg_sq)，各自与原参数同 shape/dtype/device；只检查数值，不求二阶导，不修改输入。',
         '衰减直接作用于旧 param，不进入梯度或一二阶矩；epsilon 放在 sqrt(v_hat) 外。没有 AMSGrad、maximize 或调度器。'],
        '例 1：p=[2]，g=m=v=[0]，step=1，lr=0.1，wd=0.5 → p′=[1.9]，m′=v′=[0]\n例 2：lr=0 → p′=p，但 m′/v′ 仍按当前梯度更新',
        ['先完全按纯 grad 更新 m 和 v，再偏差修正。', '参数更新拆成旧参数乘衰减系数与 Adam 步长两项。', '非零旧状态与 step>1 能检查“每步都当第一步”的错误。'],
        ['先 g+=wd×p，使衰减混入 Adam 自适应状态。', '对已经做完 Adam 更新的参数整体乘衰减，错误衰减当前步长。', '漏偏差修正或把 eps 放平方根内。'],
        ['为什么 Adam 的 L2 梯度与 AdamW 不等价？', '哪些参数通常考虑排除 weight decay？', 'lr 调度如何同时影响衰减和梯度更新？'],
        '''
        import torch

        def adamw_step(param, grad, exp_avg, exp_avg_sq, step, lr,
                       beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
            new_m = beta1 * exp_avg + (1 - beta1) * grad
            new_v = beta2 * exp_avg_sq + (1 - beta2) * grad.square()
            m_hat = new_m / (1 - beta1 ** step)
            v_hat = new_v / (1 - beta2 ** step)
            new_param = param * (1 - lr * weight_decay) - lr * m_hat / (v_hat.sqrt() + eps)
            return new_param, new_m, new_v
        ''', source='adamw', priority='P1', check_mode='numeric', prerequisites=('adam_step',), minutes=25)

    put('gradient_accumulation', '梯度累积 · 不等 Microbatch 权重', '训练优化与评估', '进阶',
        'gradient_accumulation(microbatch_grads, batch_sizes)',
        '给定同一参数在各 microbatch 的平均损失梯度，计算合并样本后平均损失的梯度。专门处理最后一个小 batch，避免简单平均 batch 均值。',
        'g_global = Σₖ nₖ gₖ / Σₖ nₖ',
        ['microbatch_grads：非空 tuple，包含 K 个同 shape/dtype/device 的有限浮点 Tensor，代表同一个参数的梯度；参数可为标量或多维。',
         'batch_sizes：长度 K 的正整数 tuple；每个 g_k 已除以其 n_k，即对应该 microbatch 的样本平均损失。',
         '返回单个同 shape/dtype/device 的平均梯度 Tensor；不修改输入，不执行 optimizer.step，只检查数值。',
         '假设各 microbatch 使用同一组模型参数且损失可按样本相加。token 平均目标应传有效 token 数，本题传入值已代表正确的归约计数。'],
        '例 1：grads=([1,3],[5,7])，sizes=(1,3) → [4,6]\n例 2：grads=([2],[8])，sizes=(2,2) → [5]',
        ['先用 batch size 把每个 mean gradient 恢复成 sum gradient。', '累加后只除一次总样本数。', '不同 batch size 时不能统一除以 microbatch 个数。'],
        ['简单 stack(...).mean(0)，把大小不同的 batch 等权。', '输入已经 mean，却重复除以各自 batch size。', '累积期间更新模型参数，使其不再对应同一点的总梯度。'],
        ['token 数不同的语言模型 batch，计数为何不能只用序列数？', '梯度裁剪应在累积前还是累积后执行？', 'DDP 的梯度平均和 AMP scale 会如何影响实现？'],
        '''
        import torch

        def gradient_accumulation(microbatch_grads, batch_sizes):
            total = torch.zeros_like(microbatch_grads[0])
            for gradient, count in zip(microbatch_grads, batch_sizes):
                total = total + gradient * count
            return total / sum(batch_sizes)
        ''', source='accumulation', priority='P1', check_mode='numeric', prerequisites=('sgd_momentum',), minutes=20)

    put('token_perplexity', 'Token Perplexity · 有效 Token 归约', '训练优化与评估', '基础',
        'token_perplexity(token_nll, mask)',
        '从逐 token 的自然对数负对数似然计算语料级 perplexity。按有效 token 等权，不能先算每条序列的 PPL 再平均。',
        'PPL = exp(Σ mask×NLL / N有效)；本题 N有效=0 时约定返回1',
        ['token_nll：[B,T] 浮点，B>0、T>=0；mask：[B,T] bool，True 表示统计位置。',
         '输入是已经对齐好的逐 token NLL，使用自然对数；本题所有值限定 [0,50]，以确保 float32 输出有限。',
         '返回同 dtype/device 的标量，按整个 batch 的有效 token 数平均后取 exp；只检查数值，不做额外 shift。',
         '全 mask 或 T=0 返回1，这是空集合的显式程序约定，不表示空语料有可比较的模型质量。'],
        '例 1：NLL=[[log(2),log(8)]]，mask全True → exp(log(4))=4\n例 2：NLL=[[5,7]]，mask全False → 1',
        ['先 mask 并求 NLL 总和，再除有效数，最后 exp。', '使用有效数 clamp_min(1) 可以得到空集合的指定结果。'],
        ['先对每个 NLL 取 exp 再平均。', '分母使用 B×T，导致 padding 稀释损失。', '把自然对数与 log₂ 混用。'],
        ['不同 tokenizer 下的 PPL 为什么不能直接比较？', '固定上下文窗口如何影响长文本 PPL 估计？', '数据混合时为什么应累加 NLL 与 token 数后再计算？'],
        '''
        import torch

        def token_perplexity(token_nll, mask):
            total = (token_nll * mask.to(token_nll.dtype)).sum()
            count = mask.sum().clamp_min(1)
            return (total / count).exp()
        ''', source='perplexity', check_mode='numeric', prerequisites=('cross_entropy',), minutes=15)
