"""推荐基础题的独立 oracle、确定性算例与错误实现回归。"""
import torch
import torch.nn.functional as F

IDS = {
    'mse_loss', 'bce_with_logits', 'logistic_regression',
    'logistic_regression_backward', 'weighted_bce', 'focal_loss',
    'cross_entropy_extended', 'sgd_momentum', 'adam_step',
    'clip_grad_norm', 'ips_snips',
}


def reference(pid, *args):
    if pid == 'mse_loss':
        pred, target, reduction = args
        return F.mse_loss(pred, target, reduction=reduction)
    if pid == 'bce_with_logits':
        logits, targets, reduction = args
        return F.binary_cross_entropy_with_logits(logits, targets, reduction=reduction)
    if pid == 'logistic_regression':
        x, weight, bias = args
        logits = F.linear(x, weight.unsqueeze(0), bias.unsqueeze(0)).select(-1, 0)
        return logits, torch.sigmoid(logits)
    if pid == 'logistic_regression_backward':
        x, weight, bias, targets, l2 = args
        # The learner derives gradients; the oracle delegates differentiation.
        with torch.enable_grad():
            w = weight.detach().clone().requires_grad_(True)
            b = bias.detach().clone().requires_grad_(True)
            logits = F.linear(x, w.unsqueeze(0), b.unsqueeze(0)).select(-1, 0)
            loss = F.binary_cross_entropy_with_logits(logits, targets) + l2 * w.square().sum() / 2
            dw, db = torch.autograd.grad(loss, (w, b))
        return loss.detach(), dw.detach(), db.detach()
    if pid == 'weighted_bce':
        logits, targets, sample_weight, pos_weight, reduction = args
        return F.binary_cross_entropy_with_logits(logits, targets, weight=sample_weight,
                                                  pos_weight=pos_weight, reduction=reduction)
    if pid == 'focal_loss':
        logits, targets, alpha, gamma = args
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        signed = torch.where(targets.bool(), logits, -logits)
        modulation = (gamma * F.logsigmoid(-signed)).exp()
        alpha_t = torch.where(targets.bool(), alpha, 1 - alpha)
        return (alpha_t * modulation * ce).mean()
    if pid == 'cross_entropy_extended':
        logits, targets, smoothing, ignore, reduction = args
        if not targets.is_floating_point() and reduction == 'mean' and not (targets != ignore).any():
            return logits.sum() * 0
        return F.cross_entropy(logits, targets, reduction=reduction,
                               label_smoothing=smoothing, ignore_index=ignore)
    if pid == 'sgd_momentum':
        param, grad, velocity, lr, momentum, decay = args
        p = torch.nn.Parameter(param.detach().clone())
        opt = torch.optim.SGD([p], lr=lr, momentum=momentum, weight_decay=decay)
        if momentum:
            opt.state[p]['momentum_buffer'] = velocity.detach().clone()
        p.grad = grad.detach().clone()
        opt.step()
        v = opt.state[p]['momentum_buffer'] if momentum else grad + decay * param
        return p.detach().clone(), v.detach().clone()
    if pid == 'adam_step':
        param, grad, avg, sq, step, lr, beta1, beta2, eps = args
        p = torch.nn.Parameter(param.detach().clone())
        opt = torch.optim.Adam([p], lr=lr, betas=(beta1, beta2), eps=eps)
        opt.state[p].update(step=torch.tensor(float(step - 1)),
                            exp_avg=avg.detach().clone(), exp_avg_sq=sq.detach().clone())
        p.grad = grad.detach().clone()
        opt.step()
        state = opt.state[p]
        return p.detach().clone(), state['exp_avg'].detach().clone(), state['exp_avg_sq'].detach().clone()
    if pid == 'clip_grad_norm':
        grads, max_norm = args
        params = [torch.nn.Parameter(torch.zeros_like(g)) for g in grads]
        for p, g in zip(params, grads):
            p.grad = g.detach().clone()
        total = torch.nn.utils.clip_grad_norm_(params, max_norm, norm_type=2.0, foreach=False)
        return tuple(p.grad.detach().clone() for p in params), total.detach().clone()
    if pid == 'ips_snips':
        losses, propensity, minimum = args
        # Small explicit scalar accumulation differs from the vectorized answer.
        numerator = losses.new_zeros(())
        weight_sum = losses.new_zeros(())
        for loss, p in zip(losses, propensity):
            weight = 1 / torch.maximum(p, p.new_tensor(minimum))
            numerator = numerator + loss * weight
            weight_sum = weight_sum + weight
        return numerator / len(losses), numerator / weight_sum
    raise KeyError(pid)


def make_cases(pid, seed=1729):
    """The first two tests are public; fixed labels/weights never require grad."""
    if pid not in IDS:
        raise KeyError(pid)
    generator = torch.Generator().manual_seed(seed)
    out = []

    def rand(shape, dtype, scale=1.0):
        return torch.randn(shape, generator=generator, dtype=dtype) * scale

    def add(name, *args, grad=True, paths=('input[0]',), require_connected_paths=()):
        case = {'name': name, 'args': args, 'grad': grad}
        if grad:
            case['grad_paths'] = paths
        if require_connected_paths:
            case['require_connected_paths'] = require_connected_paths
        out.append(case)

    for dtype in (torch.float64, torch.float32):
        suffix = ' · ' + str(dtype).split('.')[-1]

        def tensor(data):
            return torch.tensor(data, dtype=dtype)

        if pid == 'mse_loss':
            add('一维平均' + suffix, tensor([1., 3.]), tensor([0., 1.]), 'mean')
            add('多特征平均分母' + suffix, rand((3, 5), dtype), rand((3, 5), dtype), 'mean')
            add('非连续逐元素' + suffix, rand((5, 3), dtype).T, rand((5, 3), dtype).T, 'none')
            add('三维求和' + suffix, rand((2, 3, 4), dtype), rand((2, 3, 4), dtype), 'sum')
            add('标量' + suffix, tensor(-2.), tensor(.5), 'mean')
            same = rand((2, 3), dtype)
            add('预测等于标签' + suffix, same, same.clone(), 'mean')
        elif pid == 'bce_with_logits':
            add('零点梯度与硬标签' + suffix, tensor([0., 0.]), tensor([0., 1.]), 'mean')
            add('软标签与多维' + suffix, rand((2, 3), dtype), tensor([[.1, .5, .9], [0., 1., .25]]), 'mean')
            add('极端错分与正确分类' + suffix, tensor([1000., -1000., 1000., -1000.]), tensor([0., 1., 1., 0.]), 'none')
            add('非连续求和' + suffix, rand((3, 5), dtype).T, torch.sigmoid(rand((3, 5), dtype)).T, 'sum')
            add('标量软标签' + suffix, tensor(0.), tensor(.2), 'mean')
            add('正负大幅值' + suffix, tensor([-80., -10., 10., 80.]), tensor([.2, 0., 1., .8]), 'mean')
        elif pid in ('logistic_regression', 'logistic_regression_backward'):
            examples = [
                ('非方阵与非零偏置', rand((3, 5), dtype), rand((5,), dtype), tensor(.7)),
                ('单样本保留 batch 维', rand((1, 4), dtype), rand((4,), dtype), tensor(-.4)),
                ('单特征', rand((5, 1), dtype), tensor([1.7]), tensor(.2)),
                ('非连续输入', rand((5, 3), dtype).T, rand((5,), dtype), tensor(-.8)),
                ('零初始化', rand((4, 3), dtype), torch.zeros(3, dtype=dtype), tensor(0.)),
                ('极端 logits', tensor([[1000., 0.], [-1000., 0.], [0., 1.]]), tensor([1., 0.]), tensor(0.)),
            ]
            for i, (name, x, w, b) in enumerate(examples):
                if pid == 'logistic_regression':
                    add(name + suffix, x, w, b, paths=('input[0]', 'input[1]', 'input[2]'))
                else:
                    y = (torch.arange(x.shape[0]) % 2).to(dtype)
                    l2 = (0., .3, .2, .5, .1, 0.)[i]
                    add(name + ' / L2=' + str(l2) + suffix, x, w, b, y, l2, grad=False)
        elif pid == 'weighted_bce':
            add('正负类权重方向' + suffix, tensor([0., 0.]), tensor([0., 1.]), tensor([1., 2.]), tensor(3.), 'none')
            add('mean 按元素数' + suffix, tensor([-2., .3, 1.]), tensor([0., 1., 1.]), tensor([0., 2., 5.]), tensor(2.), 'mean')
            add('全零样本权重保留图' + suffix, rand((2, 3), dtype), torch.sigmoid(rand((2, 3), dtype)), torch.zeros((2, 3), dtype=dtype), tensor(4.), 'mean')
            add('pos_weight 为零' + suffix, tensor([-3., 0., 2.]), tensor([0., 1., .3]), tensor([1., 2., .5]), tensor(0.), 'sum')
            add('非连续软标签' + suffix, rand((3, 5), dtype).T, torch.sigmoid(rand((3, 5), dtype)).T, rand((3, 5), dtype).abs().T, tensor(.4), 'mean')
            add('极端 logits' + suffix, tensor([1000., -1000., 0.]), tensor([.2, .8, .5]), tensor([2., 1., 4.]), tensor(2.5), 'sum')
        elif pid == 'focal_loss':
            add('正负类零点' + suffix, tensor([0., 0.]), tensor([0., 1.]), .25, 2.)
            add('gamma=0 退化' + suffix, tensor([-2., .4, 3., -.7]), tensor([1., 0., 0., 1.]), .7, 0.)
            add('极端 logits 与小数 gamma' + suffix, tensor([-1000., 1000., -1000., 1000., 0.]), tensor([0., 1., 1., 0., 1.]), .25, .5)
            add('非连续多维' + suffix, rand((3, 5), dtype).T, (rand((3, 5), dtype) > 0).to(dtype).T, .3, 3.)
            add('alpha=0' + suffix, tensor([-.5, .2]), tensor([0., 1.]), 0., 2.)
            add('alpha=1 与标量' + suffix, tensor(.7), tensor(1.), 1., 1.5)
        elif pid == 'cross_entropy_extended':
            z = tensor([[0., 0., 0.], [1., -2., .4], [-3., .2, 2.]])
            add('硬标签忽略行' + suffix, z, torch.tensor([0, -100, 2]), 0., -100, 'mean')
            soft = tensor([[.1, .3, .6], [.7, .2, .1], [0., 1., 0.]])
            add('软标签平滑' + suffix, z, soft, .2, -100, 'mean')
            add('硬标签逐样本平滑' + suffix, z, torch.tensor([1, -100, 0]), .3, -100, 'none')
            add('硬标签求和' + suffix, z, torch.tensor([2, 1, 0]), .1, -100, 'sum')
            add('全部忽略 mean 约定' + suffix, z, torch.full((3,), -100), .2, -100, 'mean',
                require_connected_paths=('input[0]',))
            add('全部忽略 none' + suffix, z, torch.full((3,), -100), 0., -100, 'none',
                require_connected_paths=('input[0]',))
            add('软标签非连续' + suffix, z.T, soft.T, 0., -100, 'none')
            # Transposed soft distributions must each still sum to one.
            out[-1]['args'] = (z.T, soft.T / soft.T.sum(-1, keepdim=True), 0., -100, 'none')
            add('全平滑软标签求和' + suffix, z, soft, 1., -100, 'sum')
            add('单类别' + suffix, rand((4, 1), dtype), torch.zeros(4, dtype=torch.long), .4, -100, 'mean')
            add('极端 logits' + suffix, tensor([[1000., -1000., 0.], [-1000., 0., 1000.]]), torch.tensor([1, 0]), .1, -100, 'mean')
            add('自定义 ignore_index' + suffix, z, torch.tensor([0, 1, 2]), .2, 0, 'mean')
        elif pid == 'sgd_momentum':
            add('首步动量' + suffix, tensor([1., -2.]), tensor([2., -.5]), tensor([0., 0.]), .1, .9, 0., grad=False)
            add('非零历史与 L2' + suffix, tensor([1., 2.]), tensor([.5, -.3]), tensor([2., -1.]), .05, .8, .2, grad=False)
            add('momentum=0 仍返回梯度状态' + suffix, rand((2, 3), dtype), rand((2, 3), dtype), rand((2, 3), dtype), .1, 0., .3, grad=False)
            add('lr=0 状态仍更新' + suffix, tensor(2.), tensor(0.), tensor(3.), 0., .9, .4, grad=False)
            add('非连续全零梯度' + suffix, rand((3, 5), dtype).T, torch.zeros((3, 5), dtype=dtype).T, rand((3, 5), dtype).T, .01, .95, .1, grad=False)
            p, v = rand((2, 3), dtype), torch.zeros((2, 3), dtype=dtype)
            for step in range(1, 4):
                g = rand((2, 3), dtype)
                args = (p, g, v, .03, .85, .07)
                add(f'连续更新第 {step} 步' + suffix, *args, grad=False)
                p, v = reference(pid, *args)
        elif pid == 'adam_step':
            add('第一步 bias correction' + suffix, tensor([1., -2.]), tensor([2., -.3]), tensor([0., 0.]), tensor([0., 0.]), 1, .1, .9, .999, 1e-8, grad=False)
            add('非零历史状态' + suffix, tensor([1., 2.]), tensor([-.4, .2]), tensor([.3, -.7]), tensor([.4, .2]), 7, .03, .8, .9, 1e-5, grad=False)
            add('全零梯度和状态' + suffix, tensor(1.2), tensor(0.), tensor(0.), tensor(0.), 1, .1, .9, .99, 1e-8, grad=False)
            add('epsilon 在根号外' + suffix, tensor([1., 2.]), tensor([1e-5, -1e-4]), tensor([0., 0.]), tensor([0., 0.]), 1, .1, .5, .7, .01, grad=False)
            add('beta=0 非连续输入' + suffix, rand((3, 5), dtype).T, rand((3, 5), dtype).T, rand((3, 5), dtype).T, rand((3, 5), dtype).square().T, 20, .05, 0., 0., 1e-6, grad=False)
            add('lr=0 状态仍更新' + suffix, tensor([1., 2.]), tensor([0., 0.]), tensor([.3, -.2]), tensor([.2, .4]), 10, 0., .8, .95, 1e-8, grad=False)
            p = rand((2, 3), dtype)
            m, v = torch.zeros_like(p), torch.zeros_like(p)
            for step in range(1, 4):
                g = rand((2, 3), dtype)
                args = (p, g, m, v, step, .02, .85, .97, 1e-7)
                add(f'连续更新第 {step} 步' + suffix, *args, grad=False)
                p, m, v = reference(pid, *args)
        elif pid == 'clip_grad_norm':
            add('跨参数 3-4-5 范数' + suffix, (tensor([3.]), tensor([4.])), 2., grad=False)
            add('小于阈值不放大' + suffix, (tensor([.1, .2]), tensor([.3])), 10., grad=False)
            add('全零梯度' + suffix, (torch.zeros((2, 3), dtype=dtype), tensor(0.)), 1., grad=False)
            add('阈值为零' + suffix, (tensor([2., -3.]), tensor(4.)), 0., grad=False)
            add('非连续与多 shape' + suffix, (rand((3, 5), dtype).T, rand((2, 1, 4), dtype), tensor(-2.)), .5, grad=False)
            add('恰好阈值考虑 epsilon' + suffix, (tensor([3., 4.]),), 5., grad=False)
        elif pid == 'ips_snips':
            add('两种分母' + suffix, tensor([1., 3.]), tensor([1., .5]), .1, grad=False)
            add('低倾向概率截断' + suffix, tensor([2., 2.]), tensor([.001, 1.]), .1, grad=False)
            add('单条记录' + suffix, tensor([3.]), tensor([.2]), .01, grad=False)
            add('全零损失' + suffix, tensor([0., 0., 0.]), tensor([.1, .5, 1.]), .02, grad=False)
            add('倾向概率全为一' + suffix, rand((5,), dtype).abs(), torch.ones(5, dtype=dtype), .01, grad=False)
            add('混合权重与非连续输入' + suffix, rand((5, 2), dtype).abs()[:, 0], tensor([[1e-8, .3], [.001, .4], [.1, .5], [.8, .2], [1., .4]])[:, 0], .001, grad=False)
    return out


def mutations(bank):
    """One targeted misconception per problem, plus CE's all-ignored case."""
    def replace(pid, old, new):
        code = bank[pid]['solution']
        if old not in code:
            raise AssertionError(f'mutation anchor missing: {pid}: {old}')
        return pid, code.replace(old, new, 1)

    return {
        'MSE 错加一半系数': replace('mse_loss', '(pred - target).square()', '0.5 * (pred - target).square()'),
        'BCE 零点子梯度错误': (
            'bce_with_logits',
            'import torch\n\ndef bce_with_logits(logits, targets, reduction="mean"):\n'
            '    loss = logits.clamp_min(0) - targets*logits + torch.log1p(torch.exp(-logits.abs()))\n'
            '    return loss if reduction == "none" else (loss.sum() if reduction == "sum" else loss.mean())\n'),
        'LR 遗漏偏置': replace('logistic_regression', 'x @ weight + bias', 'x @ weight'),
        'LR 梯度遗漏平均': replace('logistic_regression_backward', 'x.T @ error / x.shape[0]', 'x.T @ error'),
        '加权 BCE 正类权重施加给负类': replace('weighted_bce', '+ (1 - targets) * torch.logaddexp(zero, logits)', '+ pos_weight * (1 - targets) * torch.logaddexp(zero, logits)'),
        'Focal 全部使用 alpha': replace('focal_loss', 'targets * alpha + (1 - targets) * (1 - alpha)', 'torch.full_like(targets, alpha)'),
        'CE 忽略行仍计入分母': replace('cross_entropy_extended', 'valid.sum().clamp_min(1)', 'logits.shape[0]'),
        'CE 全忽略返回断开 logits 的零叶子': replace(
            'cross_entropy_extended',
            '    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)',
            '    if not targets.is_floating_point() and not (targets != ignore_index).any():\n'
            '        shape = (logits.shape[0],) if reduction == "none" else ()\n'
            '        return torch.zeros(shape, dtype=logits.dtype, device=logits.device, requires_grad=True)\n'
            '    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)'),
        'SGD 丢弃历史动量': replace('sgd_momentum', 'momentum * velocity + effective_grad', 'effective_grad'),
        'Adam 遗漏偏差修正': replace('adam_step', 'new_avg / (1 - beta1 ** step)', 'new_avg'),
        '梯度各参数独立裁剪': replace('clip_grad_norm', 'tuple(g * scale for g in grads)', 'tuple(g * (max_norm / (g.square().sum().sqrt() + 1e-6)).clamp(max=1.0) for g in grads)'),
        'IPS 错用 SNIPS 分母': replace('ips_snips', 'numerator / losses.numel()', 'numerator / weights.sum()'),
    }
