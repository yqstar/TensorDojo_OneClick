"""推荐手写题：损失、逻辑回归、优化器与逆倾向加权。"""


def register(add):
    def put(pid, title, topic, level, signature, summary, formula, contract,
            example, hints, pitfalls, followups, solution, *, priority='P0',
            check_mode='autograd', prerequisites=(), minutes=20, source='loss'):
        links = {
            'loss': ('PyTorch 损失函数定义', 'https://docs.pytorch.org/docs/stable/nn.html#loss-functions'),
            'focal': ('Focal Loss 原论文', 'https://arxiv.org/abs/1708.02002'),
            'sgd': ('PyTorch SGD', 'https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html'),
            'adam': ('PyTorch Adam', 'https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html'),
            'clip': ('PyTorch clip_grad_norm_', 'https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html'),
            'ips': ('Recommendations as Treatments', 'https://proceedings.mlr.press/v48/schnabel16.html'),
        }
        title_, url = links[source]
        add(pid, title, topic, level, signature, summary, formula, contract,
            example, hints, pitfalls, followups, solution,
            sources=({'title': title_, 'url': url},), prerequisites=prerequisites,
            minutes=minutes, track='recsys', priority=priority, check_mode=check_mode)

    put('mse_loss', 'MSE Loss · 平方误差', '损失函数', '基础',
        'mse_loss(pred, target, reduction="mean")',
        '手写回归任务的均方误差，明确 mean 的分母。可用于推荐中的评分回归，同时练习归约与预测值梯度。',
        'ℓᵢ = (predᵢ − targetᵢ)²；mean = Σᵢℓᵢ / numel(pred)',
        ['pred、target：严格同 shape 的非空 float32/float64 Tensor；允许标量和非连续输入，禁止隐式广播。',
         'reduction 为 "none" / "sum" / "mean"；none 返回同 shape 张量，另外两种返回标量 Tensor。',
         '不额外乘 1/2，不开平方；保持 dtype/device，不修改输入。只检查 pred 的梯度，target 固定。',
         '建议用基础张量运算手写，不直接调用 F.mse_loss；平台检查结果与梯度，不强制限制 API。'],
        '例 1：pred=[1,3]，target=[0,1]，mean → 2.5\n例 2：同一输入，none → [1,4]，sum → 5',
        ['先逐元素相减、平方，暂时不要求平均。', 'mean 要覆盖所有维度，不能只除以 batch size。'],
        ['把 MSE 写成 RMSE 或多乘 1/2。', '用 len(pred) 作分母，遇到二维数据会失败。'],
        ['推导 mean 模式对 pred 的梯度。', '使用 MSE 与 BCE 训练 sigmoid 分类器时，梯度有什么差别？'],
        '''
        import torch

        def mse_loss(pred, target, reduction="mean"):
            losses = (pred - target).square()
            if reduction == "none":
                return losses
            return losses.sum() if reduction == "sum" else losses.mean()
        ''', minutes=10)

    put('bce_with_logits', '稳定 BCE with Logits', '损失函数', '基础',
        'bce_with_logits(logits, targets, reduction="mean")',
        '实现 CTR 二分类常用的交叉熵。直接接收 logits，兼容软标签和 ±1000；还要保证零点梯度正确。',
        'ℓ(z,y) = log(1+exp(z)) − yz = y·softplus(−z) + (1−y)·softplus(z)',
        ['logits、targets：同 shape 非空浮点 Tensor；targets∈[0,1]，可为软标签。',
         '支持 none/sum/mean；mean 除以全部元素数；只检查 logits 梯度。',
         '包含 z=0 与 ±1000；单个元素在零点的导数必须为 0.5−y。',
         '建议避免直接调用 BCE；允许 torch.logaddexp 等稳定基础算子。'],
        '例 1：logits=[0,0]，targets=[0,1]，mean → log(2)≈0.693147\n例 2：logits=[1000,-1000]，targets=[0,1]，none → [1000,1000]',
        ['把 log(1+exp(z)) 看成 logaddexp(0,z)。',
         '也可分别算 softplus(z) 和 softplus(-z)，以标签加权；避免先 sigmoid 再 log。',
         'abs/clamp 拼接的等价前向，可能在零点产生不同的自动求导结果。'],
        ['直接 exp(1000) 溢出，或 log(sigmoid(-1000)) 得到无穷。',
         '用 clamp_min(z,0)−z*y+log1p(exp(−abs(z))) 时忽略零点子梯度。'],
        ['BCE 与多分类 CE 的标签假设有什么不同？', '为什么裁剪概率可能改变训练梯度？'],
        '''
        import torch

        def bce_with_logits(logits, targets, reduction="mean"):
            zero = torch.zeros_like(logits)
            positive = torch.logaddexp(zero, -logits)
            negative = torch.logaddexp(zero, logits)
            losses = targets * positive + (1 - targets) * negative
            if reduction == "none":
                return losses
            return losses.sum() if reduction == "sum" else losses.mean()
        ''', prerequisites=('sigmoid',))

    put('logistic_regression', '逻辑回归 · 前向与概率', '线性模型与优化', '基础',
        'logistic_regression(x, weight, bias)',
        '用显式参数实现 LR 的打分和点击概率。输出 logits 可送入 BCE，概率可用于阈值决策和校准分析。',
        'z = Xw + b；p = sigmoid(z)',
        ['x：[B,D]；weight：[D]；bias：零维标量 Tensor；B,D>0，dtype/device 一致。',
         '返回 (logits, probabilities)，两个 Tensor 均为 [B]；B=1 也不能丢掉 batch 维。',
         '检查 x、weight、bias 的梯度；参数由调用方给定，不在函数中创建随机模型。',
         '允许 torch.sigmoid，需支持 ±1000 的 logits；不返回 0/1 硬分类标签。'],
        '例 1：x=[[1,2],[0,1]]，weight=[1,-1]，bias=1 → logits=[0,0]，p=[0.5,0.5]\n例 2：x=[[2]]，weight=[3]，bias=-1 → logits=[5]，p≈[0.993307]',
        ['矩阵乘向量自然得到 [B]，无需 squeeze。', '给所有样本广播同一个标量 bias，再算 sigmoid。'],
        ['漏掉 bias。', 'squeeze() 把 B=1 的结果变成标量。', '把概率取整会丢失可训练的梯度。'],
        ['训练时应把 logits 还是概率传给 BCEWithLogitsLoss？', '为什么线上阈值不一定为 0.5？'],
        '''
        import torch

        def logistic_regression(x, weight, bias):
            logits = x @ weight + bias
            return logits, torch.sigmoid(logits)
        ''', prerequisites=('sigmoid', 'linear'), minutes=15)

    put('logistic_regression_backward', '逻辑回归 · 手写梯度与 L2', '线性模型与优化', '进阶',
        'logistic_regression_backward(x, weight, bias, targets, l2=0.0)',
        '手推 LR 的平均损失和参数梯度，为完整训练循环准备一步可核验的计算。bias 不参与 L2。',
        'L = mean(BCE(Xw+b,y)) + λ/2·Σw²；dw = Xᵀ(p−y)/B + λw；db = mean(p−y)',
        ['x：[B,D]；weight：[D]；bias：标量 Tensor；targets：[B] 的浮点 0/1 标签；l2≥0。',
         '返回 (loss, grad_weight, grad_bias)，shape 分别为 []、[D]、[]，dtype/device 与 x 一致。',
         '手写梯度表达式，不使用 backward/autograd.grad；平台仅检查数值，不评估二阶导。',
         '损失用 mean，L2 系数为 l2/2；bias 不正则化；输入不可原地更新。'],
        '例 1：x=[[1],[3]]，w=[0]，b=0，y=[0,1]，l2=0 → (log(2),[-0.5],0)\n例 2：x=[[0]]，w=[2]，b=0，y=[1]，l2=0.1 → (log(2)+0.2,[0.2],-0.5)',
        ['先用 sigmoid 得到 p，再计算每条样本的误差 p−y。', 'X.T @ error 后除 B；L2 的梯度是 l2*weight。', '损失的稳定计算可以复用 BCE 思路，但函数必须独立可运行。'],
        ['dw 忘记除 batch size。', '把 l2*weight 的系数写成 2*l2。', '给 bias 也添加正则，偏离本题目标。'],
        ['如何用这三个输出实现固定步数的 full-batch SGD？', 'L1 正则在零点的梯度如何约定？'],
        '''
        import torch

        def logistic_regression_backward(x, weight, bias, targets, l2=0.0):
            logits = x @ weight + bias
            zero = torch.zeros_like(logits)
            losses = (targets * torch.logaddexp(zero, -logits)
                      + (1 - targets) * torch.logaddexp(zero, logits))
            loss = losses.mean() + 0.5 * l2 * weight.square().sum()
            error = torch.sigmoid(logits) - targets
            dw = x.T @ error / x.shape[0] + l2 * weight
            db = error.mean()
            return loss, dw, db
        ''', check_mode='numeric', prerequisites=('logistic_regression', 'bce_with_logits'), minutes=25)

    put('weighted_bce', '加权 BCE · 样本权重与正类权重', '损失函数', '进阶',
        'weighted_bce(logits, targets, sample_weight, pos_weight, reduction="mean")',
        '区分样本权重与正类项权重，处理类别不平衡和不同样本的重要性。此题采用与 PyTorch 一致的元素平均口径。',
        'ℓᵢ = sᵢ·[a·yᵢ·softplus(−zᵢ)+(1−yᵢ)·softplus(zᵢ)]；mean = Σℓᵢ / numel(z)',
        ['logits、targets、sample_weight：相同非空 shape；targets∈[0,1]；sample_weight≥0。',
         'pos_weight：非负标量 Tensor，只乘正类损失项；所有浮点 Tensor 同 dtype/device。',
         '支持 none/sum/mean；mean 不除以 sum(sample_weight)，也不除以正负权重之和。',
         '只对 logits 求导；targets 和两类权重固定；全零 sample_weight 返回保持计算图的零。'],
        '例 1：z=[0,0]，y=[0,1]，s=[1,2]，a=3 → none=[log(2),6log(2)]\n例 2：同一输入 mean=3.5log(2)；s=[0,0] 时 mean=0',
        ['把正类项和负类项分别展开，才能看清 pos_weight 只影响哪一项。', '最后再乘 sample_weight，并按元素数归约。'],
        ['把 pos_weight 乘给全部 BCE。', '自行使用 sum(sample_weight) 作为 mean 分母。', '全零权重时返回 torch.tensor(0.) 导致计算图丢失。'],
        ['若改成除以 sum(sample_weight)，缩放所有权重会发生什么？', '类别重加权会怎样改变概率校准？'],
        '''
        import torch

        def weighted_bce(logits, targets, sample_weight, pos_weight, reduction="mean"):
            zero = torch.zeros_like(logits)
            losses = sample_weight * (
                pos_weight * targets * torch.logaddexp(zero, -logits)
                + (1 - targets) * torch.logaddexp(zero, logits))
            if reduction == "none":
                return losses
            return losses.sum() if reduction == "sum" else losses.mean()
        ''', priority='P1', prerequisites=('bce_with_logits',))

    put('focal_loss', 'Binary Focal Loss', '损失函数', '进阶',
        'focal_loss(logits, targets, alpha=0.25, gamma=2.0)',
        '从二分类交叉熵构建 Focal Loss，让容易分类样本的权重衰减。明确 alpha 的正负类方向，处理极端 logits。',
        'pₜ = y·σ(z)+(1−y)·(1−σ(z))；αₜ=yα+(1−y)(1−α)；L=mean[−αₜ(1−pₜ)^γ log(pₜ)]',
        ['logits、targets：严格同 shape 的非空浮点 Tensor；targets 只能为 0/1 硬标签。',
         'alpha∈[0,1] 为正类权重；负类权重为 1−alpha；gamma≥0，允许小数。',
         '返回标量均值；只检查 logits 梯度。gamma=0 时退化为 alpha 加权的 BCE。',
         '支持 ±1000；避免在 pₜ=1 时对 0 做小于 1 次幂引起反向 NaN。'],
        '例 1：z=[0,0]，y=[0,1]，alpha=0.25，gamma=2 → log(2)/8≈0.0866434\n例 2：相同输入 gamma=0 → log(2)/2≈0.346574',
        ['令 s=2y−1，则 −log(pₜ)=softplus(−s*z)。',
         'log(1−pₜ)=−softplus(s*z)，在 log 域计算调制项 exp(−gamma*softplus(s*z))。'],
        ['所有样本都乘 alpha，漏掉负类的 1−alpha。', '从已舍入为 1 的概率计算 (1−p)^gamma，可能污染梯度。'],
        ['gamma 增大时，哪些样本贡献变化最大？', 'Focal Loss 与单纯正类加权分别解决什么问题？'],
        '''
        import torch

        def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
            signed = (2 * targets - 1) * logits
            zero = torch.zeros_like(logits)
            ce = torch.logaddexp(zero, -signed)
            modulation = torch.exp(-gamma * torch.logaddexp(zero, signed))
            alpha_t = targets * alpha + (1 - targets) * (1 - alpha)
            return (alpha_t * modulation * ce).mean()
        ''', priority='P1', prerequisites=('bce_with_logits', 'weighted_bce'), source='focal')

    put('cross_entropy_extended', '进阶 CE · 软标签、平滑与忽略', '损失函数', '进阶',
        'cross_entropy_extended(logits, targets, label_smoothing=0.0, ignore_index=-100, reduction="mean")',
        '在已有 CE 上补齐两种标签形式。把忽略样本的归约与标签平滑写清楚，避免有效样本数变化造成尺度错误。',
        'q = (1−ε)·target_distribution + ε/C；ℓᵢ = −Σc qᵢc·log_softmax(zᵢ)c',
        ['logits：[N,C] 非空浮点 Tensor。硬标签 targets：[N] int64，取值 0..C−1 或 ignore_index。',
         '软标签 targets：[N,C] 同 dtype/device，每行非负且和为 1；软标签分支不使用 ignore_index，应传默认 -100。',
         'label_smoothing∈[0,1]；把 ε/C 分给所有类别（包括真实类别），两种标签均支持。',
         'none 返回 [N]；sum 返回标量和；mean 硬标签除有效行数，软标签除 N。忽略行在 none 中为零。',
         '硬标签全部忽略时，sum/mean 返回与 logits 相连的零；这是本题对默认 mean 行为的显式扩展。',
         '只检查 logits 梯度；建议不直接调用 F.cross_entropy/log_softmax，使用 logsumexp。'],
        '例 1：z=[[0,0],[2,0]]，y=[0,-100]，mean → log(2)\n例 2：z=[[0,0]]，软标签=[[0.2,0.8]]，ε=0.1 → log(2)',
        ['先用 logits−logsumexp(logits) 得到稳定对数概率。',
         '硬标签先把忽略位置换成合法索引，构造 one-hot 后再 mask。',
         '有效样本数为零时用 clamp_min(1) 作为分母，分子仍保持计算图。'],
        ['先 softmax 再 log 在大幅值下不稳定。', '被忽略行仍进入 mean 分母。', '平滑只分给其他 C−1 类，和本题 ε/C 约定不同。'],
        ['软标签 CE 与 KL 散度相差哪一项？', 'label smoothing 怎样改变最优置信度？'],
        '''
        import torch

        def cross_entropy_extended(logits, targets, label_smoothing=0.0, ignore_index=-100, reduction="mean"):
            log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
            if targets.is_floating_point():
                distribution = targets
                valid = torch.ones(logits.shape[0], dtype=torch.bool, device=logits.device)
            else:
                valid = targets != ignore_index
                safe_targets = torch.where(valid, targets, torch.zeros_like(targets))
                distribution = torch.zeros_like(logits).scatter(1, safe_targets[:, None], 1)
            distribution = (1 - label_smoothing) * distribution + label_smoothing / logits.shape[-1]
            losses = -(distribution * log_probs).sum(dim=-1)
            losses = torch.where(valid, losses, torch.zeros_like(losses))
            if reduction == "none":
                return losses
            total = losses.sum()
            return total if reduction == "sum" else total / valid.sum().clamp_min(1)
        ''', priority='P1', prerequisites=('cross_entropy',), minutes=30)

    put('sgd_momentum', 'SGD + Momentum · 显式状态更新', '线性模型与优化', '基础',
        'sgd_momentum(param, grad, velocity, lr, momentum=0.0, weight_decay=0.0)',
        '实现无原地操作的一步 SGD，接收上一步动量并返回新状态。可将输出接回下一次调用，组成训练循环。',
        'g = grad + wd·param；v_new = μ·velocity + g；p_new = param − lr·v_new',
        ['param、grad、velocity 为相同非空 shape、dtype/device 的浮点 Tensor；首次 velocity 全零。',
         'lr≥0，0≤momentum<1，weight_decay≥0；不含 Nesterov 与 dampening。',
         '返回 (new_param, new_velocity)，两者 shape/dtype/device 与 param 相同；不可原地修改输入。',
         'momentum=0 时仍返回本步 g 作为 new_velocity；lr=0 时参数不变但状态仍更新。',
         '仅检查数值；测试包含由前三步真实更新衔接得到的旧状态。'],
        '例 1：p=[1]，g=[2]，v=[0]，lr=0.1，μ=0.9 → p_new=[0.8]，v_new=[2]\n例 2：继续输入 g=[1] → v_new=[2.8]，p_new=[0.52]',
        ['先加入与旧参数相关的 L2 梯度。', '用新 velocity 更新 param，不能用旧 velocity。'],
        ['每次把 velocity 清零，导致多步退化为普通 SGD。', 'lr=0 时直接返回旧 velocity。', '将 weight_decay 写成独立的 AdamW 式规则。'],
        ['动量为何需要跨 batch 保存？', '带 momentum 的 L2 梯度与解耦权重衰减有什么差别？'],
        '''
        import torch

        def sgd_momentum(param, grad, velocity, lr, momentum=0.0, weight_decay=0.0):
            effective_grad = grad + weight_decay * param
            new_velocity = momentum * velocity + effective_grad
            return param - lr * new_velocity, new_velocity
        ''', check_mode='numeric', prerequisites=('logistic_regression_backward',), minutes=15, source='sgd')

    put('adam_step', 'Adam · 一二阶矩与偏差修正', '线性模型与优化', '进阶',
        'adam_step(param, grad, exp_avg, exp_avg_sq, step, lr, beta1=0.9, beta2=0.999, eps=1e-8)',
        '手写 Adam 当前一步的参数与状态更新，区分一阶矩、二阶矩、偏差修正以及 epsilon 的位置。',
        'm=β₁m_old+(1−β₁)g；v=β₂v_old+(1−β₂)g²；m̂=m/(1−β₁ᵗ)；v̂=v/(1−β₂ᵗ)；p_new=p−lr·m̂/(√v̂+ε)',
        ['四个 Tensor 严格同 shape/dtype/device；exp_avg_sq 非负；step 为当前步的整数且≥1。',
         'lr≥0，beta1/beta2∈[0,1)，eps>0；首次 step=1 且两个旧状态为零。',
         '返回 (new_param, new_exp_avg, new_exp_avg_sq)；状态返回未修正的 m/v。',
         'epsilon 在 sqrt(v_hat) 外；不含 weight decay、AMSGrad；禁止原地修改。',
         '仅数值检查，包含 step=1、连续多步与不同 beta/eps；建议不用 torch.optim 代替手写。'],
        '例 1：p=[1]，g=[2]，旧状态=0，t=1，lr=0.1，β=(0.9,0.999) → m=[0.2]，v=[0.004]，p_new≈[0.9]\n例 2：旧状态=0，g=[0] → 三个输出分别为原 p、0、0',
        ['先更新原始矩，再各自除以 1−beta**step。', '平方的是梯度，不是新一阶矩。', '返回原始新状态，下一步才不会重复修正。'],
        ['遗漏 bias correction；第一步误差尤其明显。', '把 eps 放到 sqrt 内。', '把 m_hat/v_hat 当作需要保存的状态。'],
        ['增加 AdamW 时，参数衰减应放在哪里？', '为什么一阶矩可为负，而二阶矩不能为负？'],
        '''
        import torch

        def adam_step(param, grad, exp_avg, exp_avg_sq, step, lr, beta1=0.9, beta2=0.999, eps=1e-8):
            new_avg = beta1 * exp_avg + (1 - beta1) * grad
            new_sq = beta2 * exp_avg_sq + (1 - beta2) * grad.square()
            corrected_avg = new_avg / (1 - beta1 ** step)
            corrected_sq = new_sq / (1 - beta2 ** step)
            new_param = param - lr * corrected_avg / (corrected_sq.sqrt() + eps)
            return new_param, new_avg, new_sq
        ''', check_mode='numeric', prerequisites=('sgd_momentum',), minutes=25, source='adam')

    put('clip_grad_norm', '全局梯度 L2 范数裁剪', '线性模型与优化', '进阶',
        'clip_grad_norm(grads, max_norm)',
        '在多个参数张量之间统一计算梯度范数，并使用同一个系数缩放所有梯度。返回新张量，便于接入函数式训练循环。',
        'n = √(Σparameters Σelements g²)；scale = min(1, max_norm/(n+10⁻⁶))；g_new = scale·g',
        ['grads：非空 tuple，每项为非空浮点 Tensor，shape 可不同但 dtype/device 相同；允许标量。',
         'max_norm≥0，固定采用 L2 全局范数；epsilon 固定为 1e-6，与 PyTorch 常用实现一致。',
         '返回 (new_grads, total_norm)，new_grads 必须为同长度 tuple，total_norm 为裁剪前标量 Tensor。',
         '输入均有限且平方和不会上溢；全零梯度仍返回零；无原地操作，仅检查数值。'],
        '例 1：grads=([3],[4])，max_norm=2 → total_norm=5，梯度约 ([1.2],[1.6])\n例 2：grads=([0],[0,0]) → total_norm=0，新梯度仍全零',
        ['不能给每个 Tensor 单独定缩放系数，要先聚合平方和。', '把比例限制为不超过 1，避免扩大本来很小的梯度。'],
        ['每个参数单独裁剪，改变了整体梯度方向。', '返回裁剪后的范数。', '忘了 tuple 的输出结构或原地修改了梯度。'],
        ['全局裁剪怎样保留梯度方向？', 'AMP 训练时裁剪应发生在 unscale 前还是后？'],
        '''
        import torch

        def clip_grad_norm(grads, max_norm):
            total_norm = torch.stack([g.square().sum() for g in grads]).sum().sqrt()
            scale = (max_norm / (total_norm + 1e-6)).clamp(max=1.0)
            return tuple(g * scale for g in grads), total_norm
        ''', priority='P1', check_mode='numeric', prerequisites=('sgd_momentum',), source='clip')

    put('ips_snips', 'IPS / SNIPS · 逆倾向加权统计量', '评估与去偏', '进阶',
        'ips_snips(losses, propensity, min_propensity=0.01)',
        '根据给定观测损失与已知倾向概率，计算两种明确归一化口径的逆倾向加权统计量，比较权重截断的影响。',
        'wᵢ = 1/max(pᵢ,p_min)；IPS = Σwᵢℓᵢ/N；SNIPS = Σwᵢℓᵢ/Σwᵢ',
        ['losses、propensity：[N] 同 dtype/device 浮点 Tensor；N>0；losses≥0 且有限，0<propensity≤1。',
         'min_propensity∈(0,1]；等价于把逆倾向权重上限截断为 1/min_propensity；不删除记录。',
         '返回 (ips, snips)，均为标量 Tensor；仅检查数值，不训练 propensity 模型。',
         'N 明确为传入记录数；题目仅实现此加权公式，不补入未观测数据或额外 observation indicator。',
         '统计量能否解释为某个总体风险，取决于采样机制、目标总体及可识别性假设；裁剪一般引入偏差，不宣称无偏。'],
        '例 1：losses=[1,3]，p=[1,0.5]，p_min=0.1 → IPS=3.5，SNIPS=7/3\n例 2：losses=[2,2]，p=[0.001,1]，p_min=0.1 → IPS=11，SNIPS=2',
        ['先截断 propensity 的下界，再取倒数；不要删掉低概率样本。', '两种结果共享加权损失分子，分母分别是 N 和权重和。'],
        ['把 IPS 和 SNIPS 都除以 N 或都除以权重和。', '只给 loss 做 clamp，未限制极大权重。', '将裁剪后的统计量自动解释为无偏估计。'],
        ['截断如何改变偏差和方差？', '若目标总体规模不同于传入样本数，IPS 分母应怎样重新定义？', '已知 propensity 与估计 propensity 分别需要哪些假设？'],
        '''
        import torch

        def ips_snips(losses, propensity, min_propensity=0.01):
            weights = propensity.clamp_min(min_propensity).reciprocal()
            numerator = (weights * losses).sum()
            return numerator / losses.numel(), numerator / weights.sum()
        ''', priority='P2', check_mode='numeric', prerequisites=('weighted_bce',), minutes=25, source='ips')
