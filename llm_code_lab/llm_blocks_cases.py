"""Independent CPU references and regression probes for the LLM block bank."""
import torch
import torch.nn.functional as F

IDS = {
    'lora_linear', 'lora_merge', 'sinusoidal_positions', 'alibi_bias',
    'sliding_window_attention', 'rope_scaling', 'cross_attention',
    'rmsnorm_backward', 'moe_topk_routing', 'moe_layer',
    'moe_load_balance', 'kv_cache_update',
}


def _ordered_experts(row, k):
    """Small explicit sort with a secondary index key, independent of argsort."""
    values = row.detach().tolist()
    return sorted(range(len(values)), key=lambda index: (-values[index], index))[:k]


def reference(pid, *args):
    if pid == 'lora_linear':
        x, base, a, b, alpha = args
        # Merge the update for the oracle; the learner uses two low-rank projections.
        return F.linear(x, base + (alpha / a.shape[0]) * torch.matmul(b, a))
    if pid == 'lora_merge':
        base, a, b, alpha = args
        updated = base.clone()
        for rank in range(a.shape[0]):
            updated = updated + torch.outer(b[:, rank], a[rank]) * (alpha / a.shape[0])
        return updated
    if pid == 'sinusoidal_positions':
        positions, dim, base = args
        # One column at a time avoids reusing the paired flatten implementation.
        columns = []
        for feature in range(dim):
            frequency = base ** (2 * (feature // 2) / dim)
            angles = positions / frequency
            columns.append(torch.cos(angles) if feature % 2 else torch.sin(angles))
        return torch.stack(columns, dim=-1)
    if pid == 'alibi_bias':
        query, key, slopes = args
        output = slopes.new_empty((slopes.numel(), query.numel(), key.numel()))
        for head, slope in enumerate(slopes):
            for i, q in enumerate(query):
                for j, k in enumerate(key):
                    output[head, i, j] = slope * (int(k) - int(q))
        return output
    if pid == 'sliding_window_attention':
        q, k, v, query_pos, key_pos, window, key_mask = args
        batches = []
        # Slice each query's visible keys, then run SDPA on that shorter sequence.
        for batch in range(q.shape[0]):
            rows = []
            for index, pos in enumerate(query_pos):
                visible = key_mask[batch] & (key_pos <= pos) & (key_pos > pos - window)
                if visible.any():
                    rows.append(F.scaled_dot_product_attention(
                        q[batch, :, index:index+1], k[batch, :, visible], v[batch, :, visible],
                        dropout_p=0.0).squeeze(-2))
                else:
                    zero = q[batch, :, index].sum(-1, keepdim=True) * 0
                    rows.append(zero.expand(-1, v.shape[-1]) + k[batch].sum() * 0 + v[batch].sum() * 0)
            batches.append(torch.stack(rows, dim=-2))
        return torch.stack(batches)
    if pid == 'rope_scaling':
        x, positions, scale, base = args
        exponent = 2 * torch.arange(x.shape[-1] // 2, dtype=x.dtype, device=x.device) / x.shape[-1]
        angles = positions.to(x.dtype).unsqueeze(-1) / scale / torch.pow(base, exponent)
        # Complex multiplication is an independent realization of the 2D rotation.
        pairs = torch.complex(x[..., 0::2], x[..., 1::2])
        rotated = pairs * torch.polar(torch.ones_like(angles), angles)
        return torch.stack((rotated.real, rotated.imag), dim=-1).flatten(-2)
    if pid == 'cross_attention':
        query, context, wq, wk, wv, wo, heads, key_mask = args
        dim = query.shape[-1]
        projections = [F.linear(query, wq), F.linear(context, wk), F.linear(context, wv)]
        q, k, v = [x.unflatten(-1, (heads, dim // heads)).movedim(-2, 1) for x in projections]
        visible = key_mask[:, None, None, :]
        any_key = visible.any(-1, keepdim=True)
        safe_mask = torch.where(any_key, visible, torch.ones_like(visible))
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=safe_mask, dropout_p=0.0)
        y = torch.where(any_key, y, torch.zeros_like(y))
        return F.linear(y.movedim(1, -2).flatten(-2), wo)
    if pid == 'rmsnorm_backward':
        x, weight, upstream, eps = args
        with torch.enable_grad():
            inputs = x.detach().clone().requires_grad_(True)
            scale = weight.detach().clone().requires_grad_(True)
            normalized = inputs / torch.sqrt(torch.mean(inputs * inputs, dim=-1, keepdim=True) + eps)
            dx, dw = torch.autograd.grad(normalized * scale, (inputs, scale), grad_outputs=upstream)
        return dx.detach(), dw.detach()
    if pid == 'moe_topk_routing':
        logits, k = args
        indices = torch.tensor([_ordered_experts(row, k) for row in logits], dtype=torch.long, device=logits.device)
        probabilities = torch.stack([F.softmax(logits[token, experts], dim=0)
                                     for token, experts in enumerate(indices)])
        return indices, probabilities
    if pid == 'moe_layer':
        x, logits, w1, b1, w2, b2, k = args
        rows = []
        # Per-token expert evaluation differs from the learner's grouped dispatch.
        for token in range(x.shape[0]):
            experts = _ordered_experts(logits[token], k)
            gate = F.softmax(logits[token, experts], dim=0)
            contributions = [F.linear(F.relu(F.linear(x[token], w1[expert], b1[expert])),
                                      w2[expert], b2[expert]) * gate[slot]
                             for slot, expert in enumerate(experts)]
            rows.append(torch.stack(contributions).sum(0))
        return torch.stack(rows)
    if pid == 'moe_load_balance':
        logits, token_mask, coefficient = args
        if not token_mask.any():
            return logits.sum() * 0
        valid = logits[token_mask]
        probabilities = F.softmax(valid, dim=-1)
        winners = [_ordered_experts(row, 1)[0] for row in valid]
        total = logits.new_zeros(())
        for expert in range(logits.shape[1]):
            fraction = winners.count(expert) / len(winners)
            total = total + fraction * probabilities[:, expert].mean()
        return coefficient * logits.shape[1] * total
    if pid == 'kv_cache_update':
        old_k, old_v, new_k, new_v, start = args
        keys, values = old_k.clone(), old_v.clone()
        keys[..., start:start+new_k.shape[-2], :] = new_k
        values[..., start:start+new_v.shape[-2], :] = new_v
        return keys, values
    raise KeyError(pid)


def make_cases(pid, seed=1729):
    if pid not in IDS:
        raise KeyError(pid)
    generator = torch.Generator().manual_seed(seed)
    cases = []

    def rand(shape, dtype, scale=0.6):
        return torch.randn(shape, generator=generator, dtype=dtype) * scale

    def add(name, *args, grad=True, grad_paths=None, connected=()):
        item = {'name': name, 'args': args, 'grad': grad}
        if grad_paths is not None:
            item['grad_paths'] = grad_paths
        if connected:
            item['require_connected_paths'] = connected
        cases.append(item)

    for dtype in (torch.float64, torch.float32):
        suffix = ' · ' + str(dtype).split('.')[-1]

        def tensor(values):
            return torch.tensor(values, dtype=dtype)

        if pid in ('lora_linear', 'lora_merge'):
            configs = [
                ('非方阵 rank=2', (3, 5), 4, 2, 1.0),
                ('三维输入与非单位缩放', (2, 3, 4), 6, 3, 4.0),
                ('向量与 rank=1', (3,), 2, 1, .5),
                ('B 零初始化', (4, 5), 3, 2, 2.0),
                ('alpha=0', (2, 3), 4, 2, 0.0),
                ('非连续输入和参数', (3, 5), 4, 2, 3.5),
            ]
            for index, (name, shape, dout, rank, alpha) in enumerate(configs):
                din = shape[-1]
                x = rand(shape, dtype)
                base, a, b = rand((dout, din), dtype), rand((rank, din), dtype), rand((dout, rank), dtype)
                if index == 3:
                    b = torch.zeros_like(b)
                if index == 5:
                    x, base = rand((din, shape[0]), dtype).T, rand((din, dout), dtype).T
                    a, b = rand((din, rank), dtype).T, rand((rank, dout), dtype).T
                if pid == 'lora_linear':
                    add(name + suffix, x, base, a, b, alpha,
                        grad_paths=('input[0]', 'input[2]', 'input[3]'))
                else:
                    add(name + suffix, base, a, b, alpha, grad=False)
        elif pid == 'sinusoidal_positions':
            add('零位置与标准偶数维' + suffix, tensor([0., 1., 2.]), 6, 10000., grad=False)
            add('偏移位置与小数' + suffix, tensor([3.5, 7., 12.25]), 4, 1000., grad=False)
            add('奇数特征维' + suffix, tensor([0., .7, 5.]), 5, 10000., grad=False)
            add('dim=1' + suffix, tensor([.2, 3.]), 1, 100., grad=False)
            add('非连续位置' + suffix, tensor([2., 90., 7., 90., 11., 90.])[::2], 8, 100., grad=False)
            add('空位置序列' + suffix, torch.empty(0, dtype=dtype), 7, 10000., grad=False)
        elif pid == 'alibi_bias':
            add('基础距离惩罚' + suffix, torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2]), tensor([.5, .125]), grad=False)
            add('缓存 query 位置偏移' + suffix, torch.tensor([4, 5]), torch.arange(6), tensor([.3, .8, .05]), grad=False)
            add('单头零斜率' + suffix, torch.tensor([3]), torch.tensor([0, 2, 3, 4]), tensor([0.]), grad=False)
            add('不连续及乱序位置' + suffix, torch.tensor([8, 3]), torch.tensor([1, 7, 3]), tensor([.4, .1]), grad=False)
            add('非连续张量' + suffix, torch.arange(10)[::2], torch.arange(12)[::2], tensor([.5, .9, .2, .9])[::2], grad=False)
            add('空 query 维' + suffix, torch.empty(0, dtype=torch.long), torch.tensor([1, 2]), tensor([.5, .1]), grad=False)
        elif pid == 'sliding_window_attention':
            configs = [
                ('窗口边界包含当前位', 1, 2, [0, 1, 2, 3], [0, 1, 2, 3], 2, 3, 2),
                ('缓存前缀与不同长度', 2, 2, [4, 5], [0, 1, 2, 3, 4, 5], 3, 4, 3),
                ('window=1 仅同位置', 1, 1, [1, 3, 5], [0, 1, 2, 3, 4, 5], 1, 2, 4),
                ('全局窗口仍保持因果', 1, 3, [1, 3, 6], [0, 2, 4, 6], 100, 2, 2),
                ('全部 padding 保留零梯度', 2, 2, [2, 3], [0, 1, 2, 3], 3, 3, 2),
                ('非连续与局部空窗口', 2, 2, [1, 7, 11], [0, 2, 4, 8], 2, 3, 4),
            ]
            for index, (name, batch, heads, qp, kp, window, dim, value_dim) in enumerate(configs):
                q, k, v = rand((batch, heads, len(qp), dim), dtype), rand((batch, heads, len(kp), dim), dtype), rand((batch, heads, len(kp), value_dim), dtype)
                mask = torch.ones((batch, len(kp)), dtype=torch.bool)
                if index == 1:
                    mask[0, 3] = False
                    mask[1, 4:] = False
                if index == 4:
                    mask[:] = False
                if index == 5:
                    q = rand((batch, heads, dim, len(qp)), dtype).transpose(-1, -2)
                    k = rand((batch, heads, dim, len(kp)), dtype).transpose(-1, -2)
                    v = rand((batch, heads, value_dim, len(kp)), dtype).transpose(-1, -2)
                    mask[1, 0] = False
                paths = ('input[0]', 'input[1]', 'input[2]')
                add(name + suffix, q, k, v, torch.tensor(qp), torch.tensor(kp), window, mask,
                    grad_paths=paths, connected=paths if index == 4 else ())
        elif pid == 'rope_scaling':
            configs = [
                ('scale=1 恢复普通旋转', (1, 2, 4, 6), [0, 1, 2, 3], 1., 10000.),
                ('scale=2 与非零偏移', (2, 2, 3, 8), [9, 10, 11], 2., 10000.),
                ('非整数 scale', (1, 3, 4, 4), [1, 3, 7, 9], 3.5, 1000.),
                ('最小偶数维 D=2', (2, 1, 2, 2), [2, 5], 4., 10000.),
                ('零位置恒等', (1, 2, 3, 6), [0, 0, 0], 8., 10000.),
                ('长位置与非连续输入', (2, 2, 3, 8), [4096, 8192, 16384], 8., 10000.),
            ]
            for index, (name, shape, positions, scale, base) in enumerate(configs):
                x = rand(shape, dtype)
                if index == 5:
                    x = rand((*shape[:-2], shape[-1], shape[-2]), dtype).transpose(-1, -2)
                pos = torch.tensor(positions) if index != 2 else tensor(positions) + .5
                add(name + suffix, x, pos, scale, base, grad_paths=('input[0]',))
        elif pid == 'cross_attention':
            configs = [
                ('跨序列非方形分数', 1, 3, 5, 6, 4, 2),
                ('跨输入维度与多 batch', 2, 2, 4, 8, 5, 4),
                ('单 key 与单 head', 2, 3, 1, 4, 3, 1),
                ('单 query 部分 padding', 2, 1, 5, 6, 7, 3),
                ('全部 key 屏蔽', 2, 3, 4, 4, 5, 2),
                ('非连续与混合有效 batch', 2, 4, 3, 8, 6, 2),
            ]
            for index, (name, batch, tq, tk, dim, ctxdim, heads) in enumerate(configs):
                query, context = rand((batch, tq, dim), dtype), rand((batch, tk, ctxdim), dtype)
                wq, wk, wv, wo = rand((dim, dim), dtype), rand((dim, ctxdim), dtype), rand((dim, ctxdim), dtype), rand((dim, dim), dtype)
                mask = torch.ones((batch, tk), dtype=torch.bool)
                if index == 3:
                    mask[:, 1::2] = False
                if index == 4:
                    mask[:] = False
                if index == 5:
                    query = rand((batch, dim, tq), dtype).transpose(-1, -2)
                    context = rand((batch, ctxdim, tk), dtype).transpose(-1, -2)
                    wk = rand((ctxdim, dim), dtype).T
                    mask[0] = False
                paths = tuple(f'input[{i}]' for i in range(6))
                add(name + suffix, query, context, wq, wk, wv, wo, heads, mask,
                    grad_paths=paths, connected=paths if index == 4 else ())
        elif pid == 'rmsnorm_backward':
            configs = [
                ('二维随机上游梯度', rand((3, 5), dtype), 1e-6),
                ('三维共享 weight 归约', rand((2, 3, 4), dtype), 1e-5),
                ('向量无前导归约', rand((4,), dtype), 1e-3),
                ('D=1 与明显 epsilon', tensor([[.2], [1.5], [-.7]]), .3),
                ('零输入', torch.zeros((2, 4), dtype=dtype), .01),
                ('非连续张量', rand((5, 3), dtype).T, 1e-4),
            ]
            for name, x, eps in configs:
                add(name + suffix, x, rand((x.shape[-1],), dtype), rand(x.shape, dtype), eps, grad=False)
        elif pid == 'moe_topk_routing':
            add('并列分数按专家索引' + suffix, tensor([[1., 3., 3., 0.], [2., 2., 2., 2.]]), 2, grad=False)
            add('k=1 权重恒一' + suffix, rand((4, 5), dtype), 1, grad=False)
            add('k 等于专家数' + suffix, rand((3, 4), dtype), 4, grad=False)
            add('极端 router logits' + suffix, tensor([[1000., 999., -1000.], [-1000., -999., -998.]]), 2, grad=False)
            add('非连续输入' + suffix, rand((5, 3), dtype).T, 3, grad=False)
            add('单 token 单专家' + suffix, tensor([[.7]]), 1, grad=False)
        elif pid == 'moe_layer':
            configs = [
                ('不同 token 不同专家', 4, 3, 4, 5, 2, 2),
                ('k=1 路由概率梯度为零', 3, 4, 3, 5, 2, 1),
                ('所有专家参与', 2, 3, 3, 4, 5, 3),
                ('并列分数固定分支', 3, 2, 4, 3, 2, 2),
                ('单 token 单专家', 1, 3, 1, 4, 2, 1),
                ('非连续参数与未选专家', 4, 3, 5, 4, 2, 2),
            ]
            for index, (name, n, dim, experts, hidden, dout, topk) in enumerate(configs):
                x, logits = rand((n, dim), dtype), rand((n, experts), dtype)
                w1, b1 = rand((experts, hidden, dim), dtype), rand((experts, hidden), dtype)
                w2, b2 = rand((experts, dout, hidden), dtype), rand((experts, dout), dtype)
                if index == 3:
                    logits = torch.zeros_like(logits)
                if index == 5:
                    x = rand((dim, n), dtype).T
                    w1 = rand((experts, dim, hidden), dtype).transpose(-1, -2)
                    w2 = rand((experts, hidden, dout), dtype).transpose(-1, -2)
                    logits = tensor([[3., 2., -1., -2., -3.]]).expand(n, -1).clone()
                add(name + suffix, x, logits, w1, b1, w2, b2, topk)
        elif pid == 'moe_load_balance':
            add('均衡 top-1 分派' + suffix, tensor([[2., 0.], [0., 2.]]), torch.tensor([True, True]), .01, grad_paths=('input[0]',))
            add('不均衡负载保留 router 梯度' + suffix, tensor([[3., 1., 0.], [1., .9, -.4], [-1., 2., 0.]]), torch.ones(3, dtype=torch.bool), .2, grad_paths=('input[0]',))
            add('忽略 padding 的强 logits' + suffix, tensor([[1., 2., 0.], [1000., -1000., 0.], [2., .5, 0.]]), torch.tensor([True, False, True]), .03, grad_paths=('input[0]',))
            add('全 ties 选择第一个专家' + suffix, torch.zeros((4, 3), dtype=dtype), torch.ones(4, dtype=torch.bool), .01, grad_paths=('input[0]',))
            add('全无效 token 保留图' + suffix, rand((3, 4), dtype), torch.zeros(3, dtype=torch.bool), .1, grad_paths=('input[0]',), connected=('input[0]',))
            add('非连续与不同有效数' + suffix, rand((5, 4), dtype).T, torch.tensor([True, False, True, False]), .05, grad_paths=('input[0]',))
        elif pid == 'kv_cache_update':
            configs = [
                ('从零写入部分 cache', 1, 2, 5, 2, 0, 3, 4),
                ('非零起点接续写入', 2, 2, 6, 2, 3, 4, 3),
                ('中间覆盖保留两端', 2, 1, 7, 3, 2, 3, 2),
                ('刚好写到容量末尾', 1, 3, 5, 2, 3, 2, 4),
                ('空更新与末尾起点', 2, 2, 4, 0, 4, 3, 2),
                ('非连续完整覆盖', 2, 2, 5, 5, 0, 3, 4),
            ]
            for index, (name, batch, heads, capacity, length, start, dk, dv) in enumerate(configs):
                kc, vc = rand((batch, heads, capacity, dk), dtype), rand((batch, heads, capacity, dv), dtype)
                kn, vn = rand((batch, heads, length, dk), dtype), rand((batch, heads, length, dv), dtype)
                if index == 5:
                    kc = rand((batch, heads, dk, capacity), dtype).transpose(-1, -2)
                    vc = rand((batch, heads, dv, capacity), dtype).transpose(-1, -2)
                    kn = rand((batch, heads, dk, length), dtype).transpose(-1, -2)
                    vn = rand((batch, heads, dv, length), dtype).transpose(-1, -2)
                add(name + suffix, kc, vc, kn, vn, start, grad=False)
    return cases


def mutations(bank):
    def replace(pid, old, new):
        source = bank[pid]['solution']
        if old not in source:
            raise AssertionError(f'mutation anchor missing for {pid}: {old}')
        return pid, source.replace(old, new, 1)

    return {
        'LoRA 前向遗漏 rank 缩放': replace('lora_linear', '(alpha / rank) * update', 'alpha * update'),
        'LoRA 冻结基础权重误截断输入梯度': replace('lora_linear', 'base = x @ base_weight.T', 'base = (x @ base_weight.T).detach()'),
        'LoRA 合并遗漏 rank 缩放': replace('lora_merge', '(alpha / lora_a.shape[0])', 'alpha'),
        '正弦编码奇偶通道颠倒': replace('sinusoidal_positions', '(angles.sin(), angles.cos())', '(angles.cos(), angles.sin())'),
        'ALiBi 距离符号反向': replace('alibi_bias', 'key_positions[None, :] - query_positions[:, None]', 'query_positions[:, None] - key_positions[None, :]'),
        '滑窗边界多看一个位置': replace('sliding_window_attention', '(distance < window)', '(distance <= window)'),
        'RoPE 插值缩放方向反向': replace('rope_scaling', 'positions.to(x.dtype) / scale', 'positions.to(x.dtype) * scale'),
        'Cross Attention 误加 causal mask': replace('cross_attention', 'visible = key_mask[:, None, None, :]', 'visible = key_mask[:, None, None, :] & (torch.arange(tk, device=query.device)[None, :] <= torch.arange(tq, device=query.device)[:, None])[None, None]'),
        'RMSNorm 反向遗漏统计量梯度': replace('rmsnorm_backward', 'upstream * inv - x * inv.pow(3) * correction', 'upstream * inv'),
        'MoE Top-K 不重新归一化': replace('moe_topk_routing', 'torch.softmax(selected, dim=-1)', 'torch.softmax(router_logits, dim=-1).gather(1, indices)'),
        'MoE 丢失 gate 权重': replace('moe_layer', 'values * gates[tokens, slots, None]', 'values / k'),
        'Switch 负载频率误用平均概率': replace('moe_load_balance', 'fractions = dispatch.sum(0) / count', 'fractions = mean_probabilities'),
        'KV Cache 忽略写入起点': replace('kv_cache_update', '    end = start + new_keys.shape[-2]', '    start = 0\n    end = start + new_keys.shape[-2]'),
    }
