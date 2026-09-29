"""Authoring source for recommendation metrics and retrieval exercises (R11–R19)."""


def register(add):
    def source(title, url):
        return ({'title': title, 'url': url},)

    numeric = dict(track='recsys', priority='P0', check_mode='numeric')
    differentiable = dict(track='recsys', priority='P0', check_mode='autograd')
    auc_source = source('ROC AUC 定义与背景', 'https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html')

    add('binary_auc', 'AUC · 并列分数与排序', '推荐 · 评估指标', '进阶', 'binary_auc(labels, scores)',
        '衡量随机抽取的正样本排在负样本之前的概率。实现带并列分数处理的二分类 AUC，区分排序指标与分类准确率。',
        'AUC = Σ正样本p Σ负样本n (1[sₚ>sₙ] + 0.5×1[sₚ=sₙ]) / (N₊N₋)',
        ['labels：[N] 的 torch.long，取值 0/1；scores：[N] 的有限 float32/float64，N>1 且保证两类都存在。',
         '返回标量 Tensor，dtype/device 与 scores 相同；不检查梯度，不修改输入。分数越大表示越可能为正例。',
         '相同分数的正负配对记 0.5；不能按同分样本的原始顺序决定胜负。',
         '建议排序后按同分组累计正负数，时间 O(N log N)、额外空间 O(N)；小测试不代替复杂度复盘。单类输入不属于本题合法范围。'],
        '例 1：labels=[0,1,0,1]，scores=[0.1,0.4,0.35,0.8] → 0.75\n例 2：labels=[0,1,1,0]，scores=[2,2,2,2] → 0.5',
        ['按 score 从小到大排序；同分样本必须整体处理。',
         '该分数组的每个正例都胜过之前所有负例，与本组负例各打平一次。',
         '累计 wins += positives_in_group × (previous_negatives + 0.5 × negatives_in_group)。'],
        ['使用 > 而没有补上同分的半分。', '把 AUC 当作阈值 0.5 下的 accuracy。', '只对分数排序，忘记同步排列 labels。'],
        ['为什么 AUC 对严格单调递增的分数变换不变？', '全局 AUC 提高，为什么不必然意味着每个用户的排序更好？', '如何通过正样本平均 rank 推导等价公式？'],
        '''
        import torch

        def binary_auc(labels, scores):
            order = torch.argsort(scores)
            sorted_scores, sorted_labels = scores[order], labels[order]
            _, group_index, group_size = torch.unique_consecutive(
                sorted_scores, return_inverse=True, return_counts=True)
            positives = scores.new_zeros(group_size.numel())
            positives.scatter_add_(0, group_index, sorted_labels.to(scores.dtype))
            negatives = group_size.to(scores.dtype) - positives
            previous_negatives = negatives.cumsum(0) - negatives
            wins = (positives * (previous_negatives + 0.5 * negatives)).sum()
            return wins / (positives.sum() * negatives.sum())
        ''', sources=auc_source, minutes=25, **numeric)

    add('group_auc', 'GAUC · 有效组加权', '推荐 · 评估指标', '进阶', 'group_auc(labels, scores, group_ids)',
        '按用户分别评估排序，再按有效用户的曝光数汇总。本题固定一种 GAUC 口径，重点是单类组过滤与分母。',
        'GAUC = Σ有效组g n_g × AUC_g / Σ有效组g n_g；有效组同时含有正、负样本',
        ['labels/scores/group_ids 均为 [N]；labels 为 long 0/1，scores 为有限浮点，group_ids 为 long，允许无序、负数与不连续 ID。',
         '组内 AUC 的同分正负配对记 0.5；仅含正例或仅含负例的组全部跳过。保证至少一个有效组。',
         '按有效组的总样本数 n_g 加权；分母不包含被跳过组的任何样本。返回与 scores 同 dtype/device 的标量 Tensor；不检查梯度。',
         '这不是用户等权、按正例数加权或直接全局 AUC；实际业务须先约定口径。'],
        '例 1：labels=[0,1,0,1,1,1]，scores=[0,1,2,0,0,0]，groups=[10,10,20,20,20,20] → (2×1+4×0)/6=1/3\n例 2：在 labels=[0,1]、scores=[0,1]、groups=[5,5] 后追加任意只含负例的新组 → 仍为 1',
        ['先遍历 unique(group_ids)，分别取出组内标签和分数。', '检查 positive_count 和 negative_count，任一个为零就跳过。', '可以在提交中定义独立 AUC 辅助函数；答案必须在单次提交内完整可运行。'],
        ['把单类组的 AUC 设为 0.5 后纳入平均。', '各组简单平均，漏掉样本量权重。', '有效组分子配上全数据样本数分母。'],
        ['曝光数、点击数、用户等权三种加权方式分别偏向什么群体？', '为什么 GAUC 和全局 AUC 的变化方向可能相反？', '大规模场景如何流式维护分组统计？'],
        '''
        import torch

        def group_auc(labels, scores, group_ids):
            weighted = scores.new_zeros(())
            total = 0
            for group in torch.unique(group_ids):
                selected = group_ids == group
                y, s = labels[selected], scores[selected]
                n_pos = int(y.sum().item())
                n_neg = y.numel() - n_pos
                if not n_pos or not n_neg:
                    continue
                order = torch.argsort(s)
                sy = y[order].to(scores.dtype)
                _, inverse, counts = torch.unique_consecutive(
                    s[order], return_inverse=True, return_counts=True)
                positives = scores.new_zeros(counts.numel())
                positives.scatter_add_(0, inverse, sy)
                negatives = counts.to(scores.dtype) - positives
                wins = (positives * (negatives.cumsum(0) - 0.5 * negatives)).sum()
                auc = wins / (n_pos * n_neg)
                weighted = weighted + auc * y.numel()
                total += y.numel()
            return weighted / total
        ''', sources=auc_source, prerequisites=('binary_auc',), minutes=25, **numeric)

    add('ranking_metrics', 'Recall@K / NDCG@K', '推荐 · 评估指标', '进阶', 'ranking_metrics(scores, relevance, k)',
        '同时衡量 Top-K 找回多少相关物品，以及相关程度更高的物品是否排得更靠前。返回每个 query 的指标，避免混淆 batch 平均与单用户结果。',
        'Recall@K = TopK 中 rel>0 的数量 / 全部 rel>0 的数量；DCG@K = Σᵣ (2^relᵣ−1)/log₂(r+1)；NDCG = DCG / IDCG',
        ['scores：[B,N] 有限浮点；relevance：[B,N] 的 long，等级 0..4；B,N>0，整数 k>=1，实际 K=min(k,N)。',
         '按每行 scores 降序排序；同分时原列索引小的在前，须稳定排序。rank r 从 1 开始。',
         'Recall 把 relevance>0 视为相关；NDCG 使用指数 gain 2^rel−1，IDCG 对同一行 relevance 降序后取前 K 项。',
         '返回 tuple(recall_per_query, ndcg_per_query)，两者 shape [B]、dtype/device 与 scores 一致；无相关 item 的 query 两项均为 0；不检查梯度。',
         '参考资料用于理解 NDCG；本题固定指数 gain 和稳定 tie 顺序，不能直接照搬其他库默认的 tie 平均策略。'],
        '例 1：scores=[[3,2,1]]，relevance=[[0,2,1]]，k=2 → Recall=[0.5]，NDCG≈[0.521296]\n例 2：scores=[[1,1,1]]，relevance=[[0,1,2]]，k=1 → Recall=[0]，NDCG=[0]（同分按原索引）',
        ['torch.argsort(..., descending=True, stable=True) 后用 gather 取 relevance。', '折扣向量是 log2(arange(2,K+2))，不是 log2(arange(1,K+1))。', '把零分母替换为安全值，再显式为无相关 query 返回 0。'],
        ['用 K 作 Recall 分母，实际写成 Precision@K。', 'NDCG 只算 DCG 或对预测排序自身归一化。', '把 rel 当 gain，或把不同 query 一起排序。'],
        ['二值标签下 Recall、Precision 和 NDCG 分别强调什么？', '候选采样会如何改变离线排序指标？', '宏平均与按用户活跃度加权可能产生什么差异？'],
        '''
        import torch

        def ranking_metrics(scores, relevance, k):
            k = min(k, scores.shape[1])
            order = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :k]
            chosen = relevance.gather(1, order)
            relevant_count = (relevance > 0).sum(1).to(scores.dtype)
            recall = (chosen > 0).sum(1).to(scores.dtype) / relevant_count.clamp_min(1)
            discount = torch.arange(2, k + 2, dtype=scores.dtype, device=scores.device).log2()
            gain = torch.pow(2.0, chosen.to(scores.dtype)) - 1
            ideal_rel = torch.sort(relevance, dim=1, descending=True).values[:, :k]
            ideal_gain = torch.pow(2.0, ideal_rel.to(scores.dtype)) - 1
            dcg, idcg = (gain / discount).sum(1), (ideal_gain / discount).sum(1)
            ndcg = torch.where(idcg > 0, dcg / idcg.clamp_min(1), torch.zeros_like(dcg))
            return recall, ndcg
        ''', sources=source('NDCG 定义与不同 tie 约定', 'https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html'),
        prerequisites=('binary_auc',), minutes=30, **numeric)

    add('calibration_error', 'ECE · 点击概率校准', '推荐 · 评估指标', '进阶', 'calibration_error(probabilities, labels, n_bins=10)',
        '评估预测点击概率与实际点击率是否匹配。本题计算二分类正类概率校准误差，不把概率变成分类置信度。',
        'ECE = Σ非空桶b (n_b/N) × |mean(p in b) − mean(y in b)|',
        ['probabilities：[N] float32/float64，取值 [0,1]；labels：[N] long 0/1；N>0，整数 n_bins>=1。',
         '桶编号固定为 min(floor(p×n_bins), n_bins−1)：等宽、左闭右开，最后一桶包含 p=1。边界归属按输入 dtype 下该计算确定。',
         '每桶比较 mean(p) 与 mean(y)，按桶内样本比例加权；空桶贡献 0。返回与 probabilities 同 dtype/device 的标量 Tensor，不检查梯度。',
         '不使用 max(p,1−p)，不把标签改为“预测是否正确”；这是点击概率版本，区别于多分类 top-label confidence ECE。'],
        '例 1：p=[0,0.25,0.5,1]，labels=[0,1,0,1]，n_bins=2 → 0.3125\n例 2：p=[0.25,0.25,0.75,0.75]，labels=[0,1,0,1]，n_bins=1 → 0',
        ['先计算每个样本的桶编号，再逐桶聚合。', '每桶的贡献也等于 abs(sum(p−y in bucket))/N。', '必须跳过空桶或安全处理其分母。'],
        ['对每个样本 abs(p−y) 再平均，失去桶内抵消。', '所有桶等权，忽略样本数。', '把 p=1 丢在有效桶之外。'],
        ['为什么 AUC 很好但校准仍可能很差？', '桶数增多如何影响估计方差？', 'Platt scaling、温度缩放与 isotonic regression 各有什么限制？'],
        '''
        import torch

        def calibration_error(probabilities, labels, n_bins=10):
            bins = (probabilities * n_bins).floor().long().clamp_max(n_bins - 1)
            residual_sum = probabilities.new_zeros(n_bins)
            residual_sum.scatter_add_(0, bins, probabilities - labels.to(probabilities.dtype))
            return residual_sum.abs().sum() / probabilities.numel()
        ''', sources=source('校准背景：On Calibration of Modern Neural Networks', 'https://arxiv.org/abs/1706.04599'),
        prerequisites=('binary_auc',), minutes=20, track='recsys', priority='P2', check_mode='numeric')

    add('embedding_mean_pool', 'Embedding · Masked Mean Pooling', '推荐 · 召回与表示', '基础', 'embedding_mean_pool(table, ids, mask)',
        '将变长行为 ID 序列聚合成一个向量。练习查表、padding、有效长度和重复 ID 的梯度累加。',
        'output[b] = Σₗ mask[b,l]×table[ids[b,l]] / max(Σₗ mask[b,l], 1)',
        ['table：[V,D] 有限浮点，V,D>0；ids：[B,L] long，B>0、L>=0；mask：[B,L] bool，True 表示有效。',
         '所有 ids 都在 [0,V) 内，包括被 mask 的位置；输出 [B,D]，与 table 同 dtype/device。',
         '每行只对有效位置求均值；全 padding 或 L=0 时输出零。只检查 table 的梯度；重复 ID 的梯度必须累计，masked 位置无贡献。',
         '不把某个 ID 自动视为 padding；是否有效完全由 mask 决定，不修改输入。'],
        '例 1：table=[[1,2],[3,4],[5,6]]，ids=[[0,1,2]]，mask=[[True,False,True]] → [[3,4]]\n例 2：同样 ids 全部 mask=False → [[0,0]]',
        ['table[ids] 得到 [B,L,D]，mask.unsqueeze(-1) 用于最后一维广播。', '分母用每行 mask.sum，而非固定序列长度。', '索引和张量求和本身会正确累计重复索引的梯度。'],
        ['除以 L，使短历史的向量被错误缩小。', '忽略 mask，让 padding 参与前向或梯度。', '全 padding 分母为零或使用 torch.tensor 重建结果而断图。'],
        ['sum pooling 与 mean pooling 分别保留了哪些信息？', 'padding_idx 与任意 mask 的语义是否完全相同？', '如何扩展为权重归一化 pooling？'],
        '''
        import torch

        def embedding_mean_pool(table, ids, mask):
            embedded = table[ids]
            valid = mask.unsqueeze(-1).to(table.dtype)
            total = (embedded * valid).sum(dim=1)
            count = mask.sum(dim=1, keepdim=True).clamp_min(1).to(table.dtype)
            return total / count
        ''', sources=source('PyTorch Embedding', 'https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html'), minutes=20, **differentiable)

    add('cosine_score_matrix', '双塔末端 · 余弦评分矩阵', '推荐 · 召回与表示', '进阶', 'cosine_score_matrix(users, items, temperature=1.0, eps=1e-12)',
        '给定两塔已经生成的用户与物品向量，计算所有用户对所有候选的余弦评分。本题练习末端评分，不包含特征塔和训练框架。',
        'û = u / max(||u||₂, ε)，î = i / max(||i||₂, ε)；scores = Û Îᵀ / τ',
        ['users：[B,D]，items：[M,D]，B,M,D>0，有限浮点且同 dtype/device；允许 B≠M 和非连续张量。',
         'temperature>0、eps>0 为固定 Python 数值；逐行沿 D 归一化，分母为 max(L2_norm,eps)，不是 sqrt(sum(x²)+eps)。',
         '返回 [B,M]，保持 dtype/device；零向量归一化后仍为零，检查 users/items 的梯度。',
         '不能只计算对角配对；不修改输入，也不做额外 sigmoid/softmax。'],
        '例 1：users=[[3,4]]，items=[[3,4],[-4,3]]，temperature=0.5 → [[2,0]]\n例 2：users=[[0,0]]，items=[[1,2],[3,4]] → [[0,0]]',
        ['torch.linalg.vector_norm(x, dim=-1, keepdim=True) 可求逐行范数。', '先各自归一化再矩阵乘；用 clamp_min(eps) 处理零向量。'],
        ['沿 batch 维归一化或只归一化一侧。', '把 epsilon 加到范数内部，改变了本题公式。', '漏除 temperature，或输出 [B] 的对应配对分数。'],
        ['点积与余弦召回分别如何利用向量长度？', '怎样缓存 item embedding 以便 ANN 检索？', 'temperature 如何改变训练梯度与 softmax 分布？'],
        '''
        import torch

        def cosine_score_matrix(users, items, temperature=1.0, eps=1e-12):
            u = users / torch.linalg.vector_norm(users, dim=-1, keepdim=True).clamp_min(eps)
            v = items / torch.linalg.vector_norm(items, dim=-1, keepdim=True).clamp_min(eps)
            return (u @ v.transpose(0, 1)) / temperature
        ''', sources=source('TensorFlow Recommenders 双塔背景', 'https://www.tensorflow.org/recommenders/examples/basic_retrieval'),
        prerequisites=('embedding_mean_pool',), minutes=20, **differentiable)

    add('bpr_loss', 'BPR · Pairwise 排序损失', '推荐 · 召回与表示', '进阶', 'bpr_loss(pos_scores, neg_scores)',
        '让同一用户的正样本得分高于负样本。实现成对排序目标，并保证极大分差下的损失与梯度稳定。',
        'L = mean(−log σ(s⁺−s⁻)) = mean(log(1+exp(s⁻−s⁺)))',
        ['pos_scores、neg_scores：[B]，B>0，同 dtype/device 的有限浮点；每个位置构成一对，含 ±1000。',
         '返回标量 mean loss；不额外包含参数正则、样本权重或 sigmoid 后的概率输入。',
         '检查正负两组分数的梯度；相同分数时单项损失 log(2)，对 pos/neg 梯度分别为 −0.5/B、0.5/B。',
         '建议从 logaddexp 等稳定基本算子实现，不直接用现成 BPR loss；禁止原地修改输入。'],
        '例 1：pos=[0,0]，neg=[0,0] → log(2)≈0.693147\n例 2：pos=[1000,-1000]，neg=[0,0] → 约 500',
        ['先写清 delta=neg−pos，正样本更大时损失应更小。', 'torch.logaddexp(zeros_like(delta), delta) 稳定表达 softplus。'],
        ['把分差符号写反，鼓励负例得分更高。', '直接 log(1+exp(delta)) 在大正值溢出。', '先平均分数再算 log，而不是逐对损失后平均。'],
        ['BPR 与逐样本 BCE 分别优化什么关系？', '负样本的难度和采样分布如何影响训练？', '把每个正例扩展到多个负例时应如何归约？'],
        '''
        import torch

        def bpr_loss(pos_scores, neg_scores):
            delta = neg_scores - pos_scores
            return torch.logaddexp(torch.zeros_like(delta), delta).mean()
        ''', sources=source('BPR 原始论文', 'https://arxiv.org/abs/1205.2618'), prerequisites=('sigmoid',), minutes=20, **differentiable)

    add('inbatch_softmax_loss', 'In-batch Softmax · 对角正样本', '推荐 · 召回与表示', '进阶', 'inbatch_softmax_loss(users, items, temperature=0.1)',
        '将同一 batch 的其他物品作为负例，为每个用户训练一个按行的多分类目标。明确对角目标、归约方向和温度。',
        'Sᵢⱼ = uᵢ·vⱼ / τ；L = meanᵢ(logsumexpⱼ Sᵢⱼ − Sᵢᵢ)',
        ['users/items：[B,D]，B,D>0，同 dtype/device 的有限浮点；第 i 行 user 与第 i 行 item 是正例，temperature>0。',
         '不隐式归一化 embedding；构造 [B,B] 点积分数，对每行做 CE，目标为该行对角位置，再沿 B 平均。',
         '只做 user→item 单向目标，不与 item→user loss 平均；返回标量，检查两组向量梯度。',
         'B=1 时损失与梯度为零；本题不提供 item IDs，其他列即使值相同也仍被作为负例，不自动去重。'],
        '例 1：users=items=[[1,0],[0,1]]，temperature=1 → log(1+exp(−1))≈0.313262\n例 2：B=1，任意有限 users/items、正 temperature → 0',
        ['users @ items.T 的第 i 行对应第 i 个 user，沿最后一维计算 logsumexp。', 'Tensor.diagonal() 取正例分数。', '大 logits 时保持在 log 空间，避免先 exp 再求和。'],
        ['沿列计算 CE，改变了召回方向。', '忘记除温度或额外归一化 embedding。', '把双向对比损失作为本题的单向目标。'],
        ['batch 越大为何通常会带来更多负例？', '热门重复 item 会产生什么假负例问题？', '多正例的分子、分母应怎样定义？'],
        '''
        import torch

        def inbatch_softmax_loss(users, items, temperature=0.1):
            scores = (users @ items.T) / temperature
            scores = scores - scores.amax(dim=1, keepdim=True)
            return (torch.logsumexp(scores, dim=1) - scores.diagonal()).mean()
        ''', sources=source('CPC / InfoNCE 背景', 'https://arxiv.org/abs/1807.03748'),
        prerequisites=('cross_entropy', 'cosine_score_matrix'), minutes=25, **differentiable)

    add('sampled_softmax_loss', '采样修正 · LogQ 与假负例', '推荐 · 召回与表示', '综合', 'sampled_softmax_loss(users, items, item_ids, sampling_probs, temperature=0.1)',
        '在批内 softmax 上加入给定 item 边际采样概率的 logQ 修正，并移除与当前正例同 ID 的其他列。练习一个明确定义的教学目标，不把它当作所有 sampled softmax 方法的通用实现。',
        'zᵢⱼ = uᵢ·vⱼ/τ − log Q(itemⱼ)；Aᵢ={j : j=i 或 item_idⱼ≠item_idᵢ}；L = meanᵢ(logsumexpⱼ∈Aᵢ zᵢⱼ − zᵢᵢ)',
        ['users/items：[B,D]，item_ids：[B] long，sampling_probs：[B] 与向量同 dtype/device；B,D>0，temperature>0。',
         '第 i 对 user/item 是正例；batch 中 item 可视为从给定边际分布 Q 产生，sampling_probs[j]=Q(item_ids[j])，0<Q<=1；同 ID 的 Q 相同，不要求当前 batch 的 Q 总和为 1。',
         '先把点积除以温度，再对所有列（包含正例）减 log Q_j；不得把 logQ 也除以温度。不再加 log B 或额外采样系数。',
         '对每行，mask 掉与正例同 ID 的非对角列，保留对角正例。其他 ID 若出现多次仍按列计入分母，不对所有候选去重，也不合并成多正例分子。',
         '返回标量 mean loss；仅检查 users/items 梯度，IDs 和 Q 固定。B=1 或所有 ID 相同，损失和向量梯度为零；不做 embedding 归一化。',
         '这是指定的 logQ 批内目标及 accidental-hit 规则，不声称对任意采样过程都是全库 softmax 的无偏估计。'],
        '例 1：users=items=[[0],[0]]，IDs=[10,20]，Q=[0.25,0.5]，temperature=1 → (log(1.5)+log(3))/2≈0.752039\n例 2：相同向量，IDs=[10,10]，Q=[0.25,0.25] → 0（每行只保留自身正例）',
        ['先广播得到 [B,B] 同 ID 矩阵，再用 eye 保留对角线。', '采样修正是列方向的 sampling_probs.log().unsqueeze(0)。', '用 masked_fill(...,-inf) 后 logsumexp；每行正例始终存在，分母不会全被屏蔽。'],
        ['只给负例减 logQ，漏掉正例修正。', '同 ID mask 顺手删掉对角正例，造成整行 −inf。', '在减 logQ 之后才除温度，或把给定 Q 替换成当前小 batch 的经验频率。'],
        ['若 Q 估计不准，会怎样影响热门与长尾物品的梯度？', '负采样、去重采样和批内采样为何需要不同的修正解释？', '为什么所有 Q 同乘一个常数不改变本题 loss？', '本题剔除假负例与多正例 softmax 在优化目标上有什么差异？'],
        '''
        import torch

        def sampled_softmax_loss(users, items, item_ids, sampling_probs, temperature=0.1):
            scores = (users @ items.T) / temperature - sampling_probs.log().unsqueeze(0)
            same_id = item_ids[:, None] == item_ids[None, :]
            diagonal = torch.eye(users.shape[0], dtype=torch.bool, device=users.device)
            allowed = (~same_id) | diagonal
            masked_scores = scores.masked_fill(~allowed, float('-inf'))
            shifted = masked_scores - masked_scores.amax(dim=1, keepdim=True)
            return (torch.logsumexp(shifted, dim=1) - shifted.diagonal()).mean()
        ''', sources=source('Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations',
                            'https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/'),
        prerequisites=('inbatch_softmax_loss',), minutes=35, track='recsys', priority='P2', check_mode='autograd')
