"""Recommendation interaction, sequence and multitask exercises (R20–R31)."""


def register(add):
    def problem(pid, title, topic, level, signature, summary, formula, contract,
                example, hints, pitfalls, followups, solution, priority, source,
                prerequisites=(), minutes=30):
        add(pid, title, topic, level, signature, summary, formula, contract,
            example, hints, pitfalls, followups, solution,
            sources=({'title': source[0], 'url': source[1]},),
            prerequisites=prerequisites, minutes=minutes, track='recsys',
            priority=priority, check_mode='autograd')

    problem('fm_second_order', 'FM 二阶交互', '交互', '进阶', 'fm_second_order(x, factors)',
        '实现推荐排序中的 FM 二阶交互项，并从两两枚举推导 O(BFK) 的向量化计算。',
        'y_b = Σ_{i<j} x_bi x_bj ⟨v_i,v_j⟩ = 1/2 Σ_k [(Σ_i x_bi v_ik)² − Σ_i x_bi² v_ik²]',
        ['x：[B,F]；factors：[F,K]；B,F,K≥1；返回 [B]。',
         '特征值可以为负、为零或非单位值；F=1 时没有交互，返回零并保留计算图。',
         '不包含线性项、偏置、sigmoid；对 x 和 factors 检查梯度；不得修改输入。'],
        '样例 1：x=[[2,3]], factors=[[1],[4]] → [24]。\n样例 2：x=[[5]], factors=[[2,3]] → [0]。',
        ['先计算 x @ factors，再逐元素平方。', '扣除 x.square() @ factors.square()，最后沿 K 求和并乘 0.5。'],
        ['漏掉 1/2 会把交互计算两次。', '没有扣掉自交互或忽略 x_i*x_j。'],
        ['为什么只需 O(BFK)？', 'FM、矩阵分解和 FFM 的参数共享方式有什么区别？'],
        '''
        import torch
        def fm_second_order(x, factors):
            total = (x @ factors).square()
            diagonal = x.square() @ factors.square()
            return 0.5 * (total - diagonal).sum(dim=-1)
        ''', 'P0', ('libFM 作者资料', 'https://www.libfm.org/'))

    problem('dcn_cross_layer', 'DCN 原版 Cross Layer', '交互', '进阶', 'dcn_cross_layer(x0, xl, weight, bias)',
        '实现原版 DCN 的一个向量权重交叉层，区分原始输入 x0 与当前层输入 xl。',
        'x_{l+1} = x0 · (xlᵀ w) + b + xl；内积逐 batch 计算，标量广播到特征维。',
        ['x0、xl：[B,D]；weight、bias：[D]；返回 [B,D]，B,D≥1。',
         'bias 在乘 x0 之后相加；weight 是向量，不是 DCN-V2 的矩阵。',
         '检查全部浮点输入梯度；禁止原地更新 xl；本题只做一层。'],
        '样例 1：x0=[[2]], xl=[[3]], w=[4], b=[5] → [[32]]。\n样例 2：w=0 时输出恒为 xl+b，与 x0 无关。',
        ['计算 (xl @ weight).unsqueeze(-1)。', '与 x0 相乘后，再加 bias 和 xl。'],
        ['把 bias 放进被 x0 相乘的括号。', '遗漏残差 xl，或误把 x0 用成 xl。'],
        ['堆叠多层为什么能增加交互阶数？', 'DCN-V2 的矩阵形式提高了哪些表达能力？'],
        '''
        def dcn_cross_layer(x0, xl, weight, bias):
            return x0 * (xl @ weight).unsqueeze(-1) + bias + xl
        ''', 'P0', ('Deep & Cross Network', 'https://arxiv.org/abs/1708.05123'), ('fm_second_order',))

    problem('deepfm', 'DeepFM · 共享 Embedding 前向', '交互', '挑战',
        'deepfm(ids, values, table, linear, bias, w1, b1, w2, b2)',
        '实现固定一层隐藏 MLP 的 DeepFM 教学结构；FM 与 deep 分支必须使用同一份按特征值加权的 embedding。',
        'e_bf = table[ids_bf] · values_bf\nlogit = bias + Σ_f linear[ids_bf] values_bf + Σ_{i<j}⟨e_bi,e_bj⟩ + w2·ReLU(w1·flatten(e)+b1)+b2',
        ['ids：[B,F] int64，范围 [0,V)；values：[B,F]；table：[V,K]；linear：[V]；bias：标量 Tensor。',
         'w1：[H,F*K]；b1：[H]；w2：[1,H]；b2：[1]；所有尺寸≥1；返回 [B] logits。',
         'flatten 按 field 后 embedding 维顺序；重复 ID 合法，交互按不同 field 槽位 i<j 计算，梯度累加。',
         '检查 values、共享 table、线性和 MLP 参数的梯度；不输出 sigmoid 概率，无 dropout/正则项。'],
        '样例 1：B=2,F=3,K=4,H=5 → embedding [2,3,4] → deep输入 [2,12] → logits [2]。\n样例 2：只有一个 field 时 FM 项为零，但线性与 deep 项仍需保留。',
        ['先只 lookup 一次，并乘 values[...,None]。', 'FM 用平方和差；deep 将同一 e 从第 1 维开始 flatten。', '三路均是 logits，最后直接相加。'],
        ['FM 忽略 values，或 deep 使用另一份 embedding。', '先对各分支 sigmoid，再相加。', 'squeeze() 会误删 B=1 的 batch 维。'],
        ['共享 embedding 怎样连接显式和隐式交互？', '重复 ID 的 table 梯度为何需要累加？'],
        '''
        import torch
        def deepfm(ids, values, table, linear, bias, w1, b1, w2, b2):
            e = table[ids] * values.unsqueeze(-1)
            fm = 0.5 * (e.sum(1).square() - e.square().sum(1)).sum(-1)
            hidden = torch.relu(e.flatten(1) @ w1.T + b1)
            deep = (hidden @ w2.T + b2).squeeze(-1)
            wide = (linear[ids] * values).sum(-1) + bias
            return wide + fm + deep
        ''', 'P1', ('DeepFM 原始论文', 'https://arxiv.org/abs/1703.04247'), ('fm_second_order', 'linear'), 40)

    problem('ffm', 'FFM · Field-aware 二阶交互', '交互', '挑战', 'ffm(x, fields, factors)',
        '实现 field-aware FM 的二阶项。每个特征面向不同的对方 field 使用不同的向量。',
        'y_b = Σ_{i<j} x_bi x_bj ⟨factors[i,fields[j],:], factors[j,fields[i],:]⟩',
        ['x：[B,F]；fields：[F] int64，范围 [0,C)；factors：[F,C,K]；返回 [B]。',
         '所有维度≥1；field ID 可无序、重复；同 field 的不同特征也按 i<j 参与交互。',
         'F=1 返回零；不加线性项、bias 或 sigmoid；只对 x、factors 求导。'],
        '样例 1：fields=[0,1] 时唯一交互使用 factors[0,1] 与 factors[1,0]。\n样例 2：F=1 时返回每个样本一个 0，不得包含自交互。',
        ['枚举 i<j，先根据对方 field 选择向量。', '向量点积乘 x[:,i]*x[:,j]，累加到 [B]。'],
        ['使用 factors[i,fields[i]] 取自身 field。', '同时计算 (i,j) 和 (j,i)，导致翻倍。'],
        ['与 FM 相比，参数量如何变化？', '稀疏输入如何减少需要枚举的特征对？'],
        '''
        def ffm(x, fields, factors):
            out = x.sum(-1) * 0 + factors.sum() * 0
            for i in range(x.shape[1]):
                for j in range(i + 1, x.shape[1]):
                    dot = (factors[i, fields[j]] * factors[j, fields[i]]).sum()
                    out = out + x[:, i] * x[:, j] * dot
            return out
        ''', 'P2', ('LIBFFM 作者资料', 'https://www.csie.ntu.edu.tw/~cjlin/libffm/'), ('fm_second_order',))

    problem('wide_deep', 'Wide & Deep · 双路 Logits 融合', '交互', '进阶',
        'wide_deep(wide_x, deep_x, wide_weight, bias, w1, b1, w2, b2)',
        '实现已有 wide 特征与 deep 特征的联合前向。交叉特征由输入提供，本题不自动构造特征工程。',
        'logit = wide_x @ wide_weight + bias + Linear₂(ReLU(Linear₁(deep_x)))',
        ['wide_x：[B,F]；deep_x：[B,D]；wide_weight：[F]；bias：标量 Tensor。',
         'w1：[H,D]；b1：[H]；w2：[1,H]；b2：[1]；所有维度≥1；返回 [B] logits。',
         '两路相加后仍不做 sigmoid；只有一层 ReLU 隐藏层，无 dropout；所有浮点输入求导。'],
        '样例 1：wide输出 [2,3]，deep输出 [-1,4]，bias=0 → logits [1,7]。\n样例 2：B=1 时输出仍为 [1]，不是零维 Tensor。',
        ['wide 是向量投影，deep 是 Linear-ReLU-Linear。', 'deep 输出只 squeeze 最后一维，再相加。'],
        ['把两个概率相加，或只保留 deep 分支。', '误把 wide 特征当作 deep 输入。'],
        ['wide 分支与 deep 分支分别适合表示什么？', '如何把共享 embedding 接入这个末端结构？'],
        '''
        import torch
        def wide_deep(wide_x, deep_x, wide_weight, bias, w1, b1, w2, b2):
            wide = wide_x @ wide_weight + bias
            deep = (torch.relu(deep_x @ w1.T + b1) @ w2.T + b2).squeeze(-1)
            return wide + deep
        ''', 'P1', ('Wide & Deep 原始论文', 'https://arxiv.org/abs/1606.07792'), ('linear',))

    problem('cin_layer', 'CIN · 单层向量级交互', '交互', '挑战', 'cin_layer(x0, xl, weight, bias)',
        '实现 CIN 的一个教学提取层：每个 embedding 坐标独立做通道外积，再用共享权重压缩通道。',
        'out[b,c,d] = Σ_f Σ_h weight[c,f,h] · x0[b,f,d] · xl[b,h,d] + bias[c]',
        ['x0：[B,F,D]；xl：[B,H,D]；weight：[C,F,H]；bias：[C]；返回 [B,C,D]。',
         '所有维度≥1；weight 沿 embedding 坐标 D 共享；本题无激活、无池化、无 split-half。',
         '只实现一层，不称为完整 xDeepFM；检查全部浮点输入梯度。'],
        '样例 1：F=H=C=1，x0=[[[2,3]]], xl=[[[4,5]]], weight=[[[2]]], bias=[1] → [[[17,31]]]。\n样例 2：D=4 时输出仍保留 4 个坐标，不能提前沿 D 求和。',
        ['先用广播得到 [B,F,H,D] 的逐坐标交互。', '用 einsum 或 reshape 后线性投影把 F*H 压缩到 C。'],
        ['把 embedding 维提前求和。', '把 F/H 的权重索引顺序写反。'],
        ['堆叠 CIN 如何得到更高阶显式交互？', '为何称为向量级而不是标量级交互？'],
        '''
        import torch
        def cin_layer(x0, xl, weight, bias):
            interaction = x0.unsqueeze(2) * xl.unsqueeze(1)
            return torch.einsum('bfhd,cfh->bcd', interaction, weight) + bias[None, :, None]
        ''', 'P2', ('xDeepFM / CIN 原始论文', 'https://arxiv.org/abs/1803.05170'), ('fm_second_order',))

    problem('din_pool', 'DIN · 目标相关兴趣聚合', '序列', '进阶', 'din_pool(query, history, mask, w1, b1, w2, b2)',
        '用候选 query 为每个历史行为生成未归一化实数权重，再加权聚合。使用题目指定的教学评分器，无 Dice、CTR tower 或辅助正则。',
        'z_t = concat(q,h_t,q−h_t,q⊙h_t)\na_t = Linear₂(ReLU(Linear₁(z_t)))\noutput = Σ_t mask_t · a_t · h_t；不做 softmax。',
        ['query：[B,D]；history：[B,L,D]；mask：[B,L] bool，True 表示有效；B,D≥1，L≥0。',
         'w1：[H,4D]；b1：[H]；w2：[1,H]；b2：[1]；H≥1；返回 [B,D]。',
         '实数权重可以为负；全 padding 或空历史返回零，梯度有限；query/history/全部评分参数求导。'],
        '样例 1：有效 history=[[1,2],[3,4]]，评分都为 2 → [8,12]。\n样例 2：把有效历史整体复制一次，聚合输出翻倍；全 padding 返回零。',
        ['把 query 扩展为 [B,L,D] 后拼接四组特征。', '末层只去掉最后的单元素维度。', 'mask 后按历史维加权求和，不除长度。'],
        ['擅自 softmax 抹去历史强度信息。', '忽略 query、反转 mask 或用 mean pooling。'],
        ['未归一化权重怎样表达兴趣强度？', '若加入 Dice 和 CTR tower，需要额外定义哪些参数与状态？'],
        '''
        import torch
        def din_pool(query, history, mask, w1, b1, w2, b2):
            q = query[:, None, :].expand_as(history)
            features = torch.cat((q, history, q - history, q * history), dim=-1)
            scores = (torch.relu(features @ w1.T + b1) @ w2.T + b2).squeeze(-1)
            scores = scores * mask.to(scores.dtype)
            return (scores.unsqueeze(-1) * history).sum(dim=1)
        ''', 'P0', ('DIN 原始论文', 'https://arxiv.org/html/1706.06978v4#S4.SS3'), ('linear',), 35)

    problem('augru_cell', 'AUGRU Cell · Attention 调制更新门', '序列', '挑战', 'augru_cell(x, hidden, attention, weights)',
        '实现一个固定门约定的 AUGRU 教学单元。z 表示写入新候选的比例，attention 只调制这个 update gate。',
        'r=σ(xWrᵀ+hUrᵀ+br)，z=σ(xWzᵀ+hUzᵀ+bz)\nn=tanh(xWnᵀ+(r⊙h)Unᵀ+bn)\nz′=attention·z；h_new=(1−z′)⊙h+z′⊙n',
        ['x：[B,D]；hidden：[B,H]；attention：[B]，取值 [0,1]；返回 [B,H]。',
         'weights 包含 wr/wz/wn：[H,D]，ur/uz/un：[H,H]，br/bz/bn：[H]。',
         'reset 在 hidden 乘矩阵之前生效；不是 PyTorch GRUCell 的 reset-after 变体；全部浮点输入含 attention 求导。',
         '所有维度≥1；只做一个 cell，不含完整 DIEN 的兴趣提取器、attention 网络和辅助损失。'],
        '样例 1：attention=0 → h_new=hidden；attention=1 → 固定公式的普通 GRU 更新。\n样例 2：所有权重/bias=0 时 z=0.5,n=0，输出=(1−0.5*attention)*hidden。',
        ['分别计算 reset gate 和 update gate。', '先把 r 乘 hidden，再投影得到候选。', '将 attention[:,None] 乘 z，最后插值。'],
        ['attention 乘到整个 hidden 输出。', '把 z 当保留旧状态的比例，门方向反了。', '把 reset-before 改成 reset-after。'],
        ['AIGRU、AGRU 和 AUGRU 调制的位置有什么不同？', 'attention 接近零时隐藏状态为何应保留？'],
        '''
        import torch
        def augru_cell(x, hidden, attention, weights):
            w = weights
            r = torch.sigmoid(x @ w['wr'].T + hidden @ w['ur'].T + w['br'])
            z = torch.sigmoid(x @ w['wz'].T + hidden @ w['uz'].T + w['bz'])
            n = torch.tanh(x @ w['wn'].T + (r * hidden) @ w['un'].T + w['bn'])
            update = attention[:, None] * z
            return (1 - update) * hidden + update * n
        ''', 'P2', ('DIEN / AUGRU 原始论文', 'https://ojs.aaai.org/index.php/AAAI/article/download/4545/4423'), ('din_pool',), 40)

    problem('sasrec_block', 'SASRec 风格 · 因果序列 Block', '序列', '挑战', 'sasrec_block(x, mask, weights, n_heads, eps=1e-5)',
        '实现一个明确双 pre-norm 口径的 SASRec 风格教学 Block。输入已含 item/position 表示；不复刻原论文全部细节，不包含 embedding、预测头或 dropout。',
        'x0=x·mask；z=LN₁(x0)\nh=(x0+MHA(z, causal AND key_valid))·mask\nout=(h+Linear₂(ReLU(Linear₁(LN₂(h)))))·mask',
        ['x：[B,T,D]；mask：[B,T] bool，True 为有效 token；B,T,D≥1；n_heads 整除 D；输出 [B,T,D]。',
         'weights 的 wq/wk/wv/wo：[D,D]，按 out×in；无 QKV/O bias。',
         'norm1_w/norm1_b/norm2_w/norm2_b：[D]；w1：[F,D]，b1：[F]，w2：[D,F]，b2：[D]。',
         'LayerNorm 总体方差、eps 在平方根内；Q/K/V 均来自 LN1(x0)；缩放 sqrt(D/n_heads)。',
         '仅可见 j≤i 且 mask[b,j]=True；全屏蔽行 attention 为零。输入及两次残差后 padding query 必须归零。',
         '全部浮点输入和参数求导；dropout=0，不做最终额外 LayerNorm。'],
        '样例 1：mask=[False,True,True]，token 2 只能看 token 1、2；token 0 输出为零。\n样例 2：整条历史全 padding 时所有输出为零且梯度有限。',
        ['先清零输入 padding；按最后一维拆头。', '组合因果与 key mask；全屏蔽行先用安全 scores，再把概率归零。', '明确在 MHA 与 FFN 之前归一化，并在每次残差后清零 query。'],
        ['使用双向 attention，泄漏未来行为。', '只屏蔽 key，却让 padding query 保留残差或 bias。', '误写为 post-norm。'],
        ['位置表示为什么需由输入提供？', '为何只有 key padding mask 不足以保证 padding 输出为零？'],
        '''
        import torch
        def sasrec_block(x, mask, weights, n_heads, eps=1e-5):
            w = weights
            keep = mask.unsqueeze(-1).to(x.dtype)
            x0 = x * keep
            def norm(a, prefix):
                centered = a - a.mean(-1, keepdim=True)
                return centered * torch.rsqrt(centered.square().mean(-1, keepdim=True) + eps) * w[prefix+'_w'] + w[prefix+'_b']
            z = norm(x0, 'norm1')
            b, t, d = x.shape
            q, k, v = [(z @ w[name].T).reshape(b, t, n_heads, d // n_heads).transpose(1, 2) for name in ('wq', 'wk', 'wv')]
            scores = q @ k.transpose(-2, -1) / (d // n_heads) ** 0.5
            causal = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
            visible = causal[None, None] & mask[:, None, None, :]
            has_key = visible.any(-1, keepdim=True)
            scores = scores.masked_fill(~visible, float('-inf'))
            scores = torch.where(has_key, scores, torch.zeros_like(scores))
            probability = torch.softmax(scores, -1) * has_key.to(x.dtype)
            attention = (probability @ v).transpose(1, 2).reshape(b, t, d) @ w['wo'].T
            h = (x0 + attention) * keep
            feedforward = torch.relu(norm(h, 'norm2') @ w['w1'].T + w['b1']) @ w['w2'].T + w['b2']
            return (h + feedforward) * keep
        ''', 'P2', ('SASRec 作者实现（结构背景）', 'https://github.com/kang205/SASRec'), ('multihead', 'layernorm'), 50)

    problem('mmoe', 'MMoE · 多任务独立 Gate', '多任务', '挑战', 'mmoe(x, weights)',
        '实现共享单层 ReLU experts 与每任务独立 gate 的 MMoE 提取层；不含各任务 tower 和预测损失。',
        'expert[b,e,h]=ReLU(W_e x_b+b_e)\ngate[b,t,:]=softmax(G_t x_b+c_t, expert维)\nout[b,t,h]=Σ_e gate[b,t,e] expert[b,e,h]',
        ['x：[B,D]；weights：expert_w [E,H,D]、expert_b [E,H]、gate_w [T,E,D]、gate_b [T,E]。',
         '返回 [B,T,H]；B,D,E,H,T≥1；expert 对所有任务共享，gate 对任务独立。',
         '沿 E 维 softmax，非 task 维；全部浮点输入和参数求导，无 dropout。'],
        '样例 1：E=1 时每个任务输出同一个 expert；gate 参数梯度为零。\n样例 2：两个 expert 输出 2、6，某任务 gate logits 相同 → 该任务输出 4。',
        ['批量算出 [B,E,H] experts 和 [B,T,E] gate logits。', '沿最后 E 维归一化，再按 expert 加权求和。'],
        ['沿 T 维 softmax。', '所有任务共享同一个 gate 或省略 expert ReLU。'],
        ['共享 experts 为什么不代表所有任务使用相同比例？', '如何观察负迁移以及 gate 塌缩？'],
        '''
        import torch
        def mmoe(x, weights):
            w = weights
            experts = torch.relu(torch.einsum('bd,ehd->beh', x, w['expert_w']) + w['expert_b'])
            gates = torch.softmax(torch.einsum('bd,ted->bte', x, w['gate_w']) + w['gate_b'], dim=-1)
            return torch.einsum('bte,beh->bth', gates, experts)
        ''', 'P1', ('MMoE 原始论文', 'https://research.google/pubs/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-of-experts/'), ('softmax', 'linear'), 40)

    problem('esmm_loss', 'ESMM · 全曝光空间联合损失', '多任务', '挑战', 'esmm_loss(ctr_logits, cvr_logits, clicks, conversions)',
        '只实现 ESMM 的联合目标。输入来自两个 tower，利用 pCTCVR=pCTR*pCVR，对全曝光样本监督 CTR 和 CTCVR。',
        'pCTR=σ(c)，pCVR=σ(v)，pCTCVR=pCTR·pCVR\nloss=mean(BCE(pCTR,click))+mean(BCE(pCTCVR,conversion))',
        ['四个输入均为 [B]，B≥1；两种标签为浮点 0/1，conversions≤clicks；未点击样本 conversion=0。',
         '返回标量 Tensor；两项损失等权相加，每项均除完整 B；不单独监督 CVR，不添加正则项。',
         '只对 ctr_logits、cvr_logits 求导，标签固定；支持 ±1000，禁止直接概率乘积后 log 导致下溢。',
         '可用 log-sigmoid / softplus / logaddexp 计算；不 clamp 概率来改变数学目标。'],
        '样例 1：c=v=0、click=conversion=1 → loss=ln2+ln4=ln8。\n样例 2：c=v=0、click=conversion=0 → loss=ln2−ln(3/4)。',
        ['正 CTCVR 标签的负 log 概率是 softplus(-c)+softplus(-v)。', '1−pCTR*pCVR=(1−pCTR)+pCTR*(1−pCVR)，两项在 log 域用 logaddexp 相加。', 'CTR 可用 logaddexp(0,c)−click*c。'],
        ['把 CVR 当 CTCVR 直接监督。', '只在点击样本计算第二项；或把全曝光 conversion 当 CVR 标签。', 'sigmoid 饱和后再 log，造成 Inf/NaN。'],
        ['为何 conversion 标签必须与点击保持逻辑一致？', '样本选择偏差与标签延迟分别需要哪些额外考虑？'],
        '''
        import torch
        import torch.nn.functional as F
        def esmm_loss(ctr_logits, cvr_logits, clicks, conversions):
            c, v = ctr_logits, cvr_logits
            ctr = torch.logaddexp(torch.zeros_like(c), c) - clicks * c
            positive = F.softplus(-c) + F.softplus(-v)
            log_not_click = -F.softplus(c)
            log_click_not_convert = -F.softplus(-c) - F.softplus(v)
            negative = -torch.logaddexp(log_not_click, log_click_not_convert)
            joint = conversions * positive + (1 - conversions) * negative
            return ctr.mean() + joint.mean()
        ''', 'P1', ('ESMM 原始论文', 'https://arxiv.org/abs/1804.07931'), ('bce_with_logits',), 35)

    problem('ple_layer', 'PLE · 双任务单提取层', '多任务', '挑战', 'ple_layer(task1, task2, shared, weights)',
        '实现两个任务与一个共享流的 PLE 教学提取层。专属与共享专家可见集合固定，不含多层渐进堆叠和最终 towers。',
        'E1=ReLU(task1专家(task1))；E2=ReLU(task2专家(task2))；Es=ReLU(shared专家(shared))\ny1=gate1(task1)混合[E1,Es]；y2=gate2(task2)混合[E2,Es]\nys=gate_shared(shared)混合[E1,E2,Es]；各 gate 在自己的专家维 softmax。',
        ['task1、task2、shared：[B,D]；返回 tuple：(y1,y2,ys)，每项 [B,H]；所有尺寸≥1。',
         'weights 的 task1_w/task2_w/shared_w 分别为 [E1,H,D]/[E2,H,D]/[Es,H,D]；对应 *_b 为 [E,H]。',
         'gate1_w：[E1+Es,D]，gate2_w：[E2+Es,D]，gate_shared_w：[E1+E2+Es,D]；各 bias 为 [可见专家数]。',
         '专家顺序严格为本任务后 shared；shared gate 顺序为 task1、task2、shared；每个专家是单层 Linear-ReLU。',
         'gate1 不能见 task2 专家，gate2 不能见 task1 专家；所有浮点输入和参数求导，无 dropout。'],
        '样例 1：各流只有一个 expert，输出分别 2、6、10，所有 gate logits=0 → y1=6，y2=8，ys=6。\n样例 2：E1=2,E2=1,Es=3 时三个 gate 的长度依次为 5、4、6。',
        ['三个输入流分别计算自己的 experts。', '按题面顺序拼接可见集合，然后用对应流计算 gate。', '混合沿 expert 维归约，返回固定顺序 tuple。'],
        ['每个任务 gate 都看全部任务专家，失去本题规定结构。', '拼接顺序与 gate 权重不对应。', '用 shared 输入替代 task-specific 输入。'],
        ['为什么 shared gate 可见集合比 task gate 更大？', '堆叠多个提取层时三个输出如何成为下一层输入？'],
        '''
        import torch
        def ple_layer(task1, task2, shared, weights):
            w = weights
            def experts(x, prefix):
                return torch.relu(torch.einsum('bd,ehd->beh', x, w[prefix+'_w']) + w[prefix+'_b'])
            e1, e2, es = experts(task1, 'task1'), experts(task2, 'task2'), experts(shared, 'shared')
            def mix(x, visible, prefix):
                gate = torch.softmax(x @ w[prefix+'_w'].T + w[prefix+'_b'], dim=-1)
                return (gate.unsqueeze(-1) * visible).sum(1)
            y1 = mix(task1, torch.cat((e1, es), 1), 'gate1')
            y2 = mix(task2, torch.cat((e2, es), 1), 'gate2')
            ys = mix(shared, torch.cat((e1, e2, es), 1), 'gate_shared')
            return y1, y2, ys
        ''', 'P2', ('PLE 原始论文', 'https://dl.acm.org/doi/10.1145/3383313.3412236'), ('mmoe',), 45)
