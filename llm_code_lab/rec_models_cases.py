"""Independent small-tensor oracles and structural mutations for R20–R31."""
import torch
import torch.nn.functional as F

IDS = {'fm_second_order', 'dcn_cross_layer', 'deepfm', 'ffm', 'wide_deep',
       'cin_layer', 'din_pool', 'augru_cell', 'sasrec_block', 'mmoe',
       'esmm_loss', 'ple_layer'}


def reference(pid, *args):
    if pid == 'fm_second_order':
        x, factors = args
        out = x.sum(-1) * 0 + factors.sum() * 0
        for i in range(x.shape[1]):
            for j in range(i + 1, x.shape[1]):
                out = out + (factors[i] * factors[j]).sum() * x[:, i] * x[:, j]
        return out
    if pid == 'dcn_cross_layer':
        x0, xl, weight, bias = args
        outer = x0.unsqueeze(-1) * xl.unsqueeze(-2)
        return torch.einsum('bij,j->bi', outer, weight) + bias + xl
    if pid == 'deepfm':
        ids, values, table, linear, bias, w1, b1, w2, b2 = args
        slots = [F.embedding(ids[:, i], table) * values[:, i:i+1] for i in range(ids.shape[1])]
        out = bias + sum(F.embedding(ids[:, i], linear[:, None]).squeeze(-1) * values[:, i]
                         for i in range(ids.shape[1]))
        for i in range(len(slots)):
            for j in range(i + 1, len(slots)):
                out = out + (slots[i] * slots[j]).sum(-1)
        return out + F.linear(F.relu(F.linear(torch.cat(slots, -1), w1, b1)), w2, b2).squeeze(-1)
    if pid == 'ffm':
        x, fields, factors = args
        directed = factors[:, fields, :]
        pair_scores = (directed * directed.transpose(0, 1)).sum(-1).triu(diagonal=1)
        return ((x @ pair_scores) * x).sum(-1)
    if pid == 'wide_deep':
        wide, deep, ww, bias, w1, b1, w2, b2 = args
        return F.linear(wide, ww.unsqueeze(0), bias.reshape(1)).squeeze(-1) + F.linear(F.relu(F.linear(deep, w1, b1)), w2, b2).flatten()
    if pid == 'cin_layer':
        x0, xl, weight, bias = args
        channels = []
        for c in range(weight.shape[0]):
            terms = [x0[:, i] * xl[:, j] * weight[c, i, j]
                     for i in range(x0.shape[1]) for j in range(xl.shape[1])]
            channels.append(torch.stack(terms).sum(0) + bias[c])
        return torch.stack(channels, 1)
    if pid == 'din_pool':
        query, history, mask, w1, b1, w2, b2 = args
        out = query * 0 + history.sum(1) * 0 + (w1.sum() + b1.sum() + w2.sum() + b2.sum()) * 0
        for t in range(history.shape[1]):
            h = history[:, t]
            z = torch.cat([query, h, query - h, query * h], -1)
            score = F.linear(F.relu(F.linear(z, w1, b1)), w2, b2)
            out = out + torch.where(mask[:, t:t+1], score * h, torch.zeros_like(h))
        return out
    if pid == 'augru_cell':
        x, h, a, w = args
        joined = torch.cat((x, h), -1)
        r = torch.sigmoid(F.linear(joined, torch.cat((w['wr'], w['ur']), -1), w['br']))
        z = torch.sigmoid(F.linear(joined, torch.cat((w['wz'], w['uz']), -1), w['bz']))
        candidate = torch.tanh(F.linear(x, w['wn'], w['bn']) + F.linear(r * h, w['un']))
        return h + (candidate - h) * z * a[:, None]
    if pid == 'sasrec_block':
        x, mask, w, heads, eps = args
        b, t, d = x.shape
        x0 = torch.where(mask[..., None], x, torch.zeros_like(x))
        z = F.layer_norm(x0, (d,), w['norm1_w'], w['norm1_b'], eps)
        q, k, v = [F.linear(z, w[name]).unflatten(-1, (heads, d // heads)).movedim(-2, 1)
                   for name in ('wq', 'wk', 'wv')]
        batches = []
        # Oracle selects only each query's visible prefix instead of constructing a causal matrix.
        for bi in range(b):
            tokens = []
            for i in range(t):
                indices = torch.nonzero(mask[bi, :i+1], as_tuple=False).flatten()
                if indices.numel():
                    yi = F.scaled_dot_product_attention(q[bi, :, i:i+1], k[bi, :, indices],
                                                        v[bi, :, indices], dropout_p=0.0)
                else:
                    yi = q[bi, :, i:i+1] * 0 + v[bi, :, :1] * 0
                tokens.append(yi)
            batches.append(torch.cat(tokens, dim=-2))
        attention = torch.stack(batches).movedim(1, -2).flatten(-2)
        h = torch.where(mask[..., None], x0 + F.linear(attention, w['wo']), torch.zeros_like(x0))
        z2 = F.layer_norm(h, (d,), w['norm2_w'], w['norm2_b'], eps)
        ff = F.linear(F.relu(F.linear(z2, w['w1'], w['b1'])), w['w2'], w['b2'])
        return torch.where(mask[..., None], h + ff, torch.zeros_like(h))
    if pid == 'mmoe':
        x, w = args
        experts = [F.relu(F.linear(x, ew, eb)) for ew, eb in zip(w['expert_w'], w['expert_b'])]
        tasks = []
        for gw, gb in zip(w['gate_w'], w['gate_b']):
            gates = F.softmax(F.linear(x, gw, gb), -1)
            tasks.append(sum(e * gates[:, i:i+1] for i, e in enumerate(experts)))
        return torch.stack(tasks, 1)
    if pid == 'esmm_loss':
        ctr, cvr, clicks, conversions = args
        # Odds(pCTR*pCVR) = exp(ctr+cvr)/(1+exp(ctr)+exp(cvr)).
        joint_logits = ctr + cvr - torch.logsumexp(torch.stack((torch.zeros_like(ctr), ctr, cvr)), dim=0)
        return F.binary_cross_entropy_with_logits(ctr, clicks) + F.binary_cross_entropy_with_logits(joint_logits, conversions)
    if pid == 'ple_layer':
        x1, x2, xs, w = args
        groups = [[F.relu(F.linear(x, ew, eb)) for ew, eb in zip(w[p+'_w'], w[p+'_b'])]
                  for x, p in ((x1, 'task1'), (x2, 'task2'), (xs, 'shared'))]
        e1, e2, es = groups
        outputs = []
        for x, pool, p in ((x1, e1+es, 'gate1'), (x2, e2+es, 'gate2'), (xs, e1+e2+es, 'gate_shared')):
            gates = F.softmax(F.linear(x, w[p+'_w'], w[p+'_b']), -1)
            outputs.append(sum(expert * gates[:, i:i+1] for i, expert in enumerate(pool)))
        return tuple(outputs)
    raise KeyError(pid)


def make_cases(pid, seed=1729):
    if pid not in IDS:
        raise KeyError(pid)
    generator = torch.Generator().manual_seed(seed)
    out = []
    for dtype in (torch.float64, torch.float32):
        suffix = str(dtype).split('.')[-1]
        def rand(*shape, scale=0.6):
            return torch.randn(shape, generator=generator, dtype=dtype) * scale
        def add(name, *args, grad_paths=None):
            case = {'name': name+' · '+suffix, 'args': args, 'grad': True}
            if grad_paths is not None:
                case['grad_paths'] = grad_paths
            out.append(case)
        if pid == 'fm_second_order':
            add('两特征手算', torch.tensor([[2., 3.]], dtype=dtype), torch.tensor([[1.], [4.]], dtype=dtype))
            add('负值与非单位值', rand(3, 4), rand(4, 3))
            add('单特征无交互', rand(2, 1), rand(1, 4))
            x = rand(3, 5); x[:, ::2] = 0
            add('稀疏特征', x, rand(5, 2))
            add('全零特征', torch.zeros(2, 4, dtype=dtype), rand(4, 3))
            add('非连续输入与参数', rand(5, 3).T, rand(2, 5).T)
            add('相同因子', rand(2, 3), torch.ones(3, 2, dtype=dtype))
            add('单样本多因子', rand(1, 6), rand(6, 5))
        elif pid == 'dcn_cross_layer':
            add('标量特征手算', torch.tensor([[2.]], dtype=dtype), torch.tensor([[3.]], dtype=dtype),
                torch.tensor([4.], dtype=dtype), torch.tensor([5.], dtype=dtype))
            add('原输入不同于当前层', rand(3, 4), rand(3, 4), rand(4), rand(4))
            add('零权重保留bias残差', rand(2, 3), rand(2, 3), torch.zeros(3, dtype=dtype), rand(3))
            add('零原始输入', torch.zeros(2, 3, dtype=dtype), rand(2, 3), rand(3), rand(3))
            add('零当前层', rand(2, 3), torch.zeros(2, 3, dtype=dtype), rand(3), rand(3))
            add('非连续双输入', rand(5, 3).T, rand(5, 3).T, rand(5), rand(5))
            add('单样本多特征', rand(1, 7), rand(1, 7), rand(7), rand(7))
            add('非连续向量参数', rand(4, 3), rand(4, 3), rand(6)[::2], rand(6)[::2])
        elif pid == 'deepfm':
            settings = [(2,3,4,5), (3,2,3,4), (1,1,2,3), (2,4,2,3),
                        (2,3,3,2), (3,3,2,4), (1,2,1,1), (2,3,2,5)]
            names = ['共享embedding三路', '重复ID累加', '单field与单batch', '负值与稀疏特征',
                     '零values保留bias', '非连续values', '最小hidden', '深分支关闭仍有FM']
            for i, (b, f, k, h) in enumerate(settings):
                vocab = 7
                ids = torch.randint(vocab, (b, f), generator=generator)
                if i == 1: ids[:] = torch.tensor([2, 2])
                values = rand(b, f) if i != 5 else rand(f, b).T
                if i == 3: values[:, ::2] = 0
                if i == 4: values.zero_()
                w2 = rand(1, h)
                if i == 7: w2.zero_()
                add(names[i], ids, values, rand(vocab, k), rand(vocab), rand(), rand(h, f*k), rand(h), w2, rand(1))
        elif pid == 'ffm':
            settings = [(2,3,2,2), (3,4,3,3), (2,1,2,3), (1,3,1,2),
                        (2,5,3,1), (3,4,3,2), (2,3,4,2), (1,2,2,1)]
            names = ['跨field方向', '无序重复field', '单特征无交互', '同field也交互',
                     '稀疏特征', '非连续输入', '零特征', '最小向量']
            for i, (b, f, c, k) in enumerate(settings):
                x = rand(b, f) if i != 5 else rand(f, b).T
                if i == 4: x[:, 1::2] = 0
                if i == 6: x.zero_()
                fields = torch.arange(f).remainder(c).flip(0)
                add(names[i], x, fields, rand(f, c, k))
        elif pid == 'wide_deep':
            for i in range(8):
                b, f, d, h = (1, 1, 1, 1) if i == 2 else (3, 4, 2, 5)
                wide, deep = (rand(b, f), rand(b, d)) if i != 5 else (rand(f, b).T, rand(d, b).T)
                ww, w2 = rand(f), rand(1, h)
                if i == 3: ww.zero_()
                if i == 4: w2.zero_()
                if i == 6: wide.zero_(); deep.zero_()
                add(['双路融合', '不同输入宽度', '最小batch', '只有deep贡献', '只有wide贡献',
                     '非连续双输入', '零输入非零bias', '大正负logits'][i], wide, deep, ww,
                    rand(scale=20 if i == 7 else 0.6), rand(h, d), rand(h), w2, rand(1))
        elif pid == 'cin_layer':
            add('逐坐标手算', torch.tensor([[[2.,3.]]], dtype=dtype), torch.tensor([[[4.,5.]]], dtype=dtype),
                torch.tensor([[[2.]]], dtype=dtype), torch.tensor([1.], dtype=dtype))
            for i, (b,f,h,c,d) in enumerate([(2,3,2,4,5), (1,1,2,3,1), (2,4,3,1,2),
                                           (2,3,2,4,3), (3,2,3,2,4), (1,2,2,3,5), (2,3,4,2,1)]):
                x0 = rand(b,f,d) if i != 3 else rand(b,d,f).transpose(-2,-1)
                xl = rand(b,h,d)
                weight = rand(c,f,h)
                if i == 2: x0.zero_()
                if i == 4: weight.zero_()
                add(['不同field通道数', '单embedding坐标', '零原始输入', '非连续输入',
                     '零权重保留bias', '共享坐标权重', '单坐标多通道'][i], x0, xl, weight, rand(c))
        elif pid == 'din_pool':
            settings = [(2,3,4,3), (2,4,3,5), (2,0,3,4), (2,3,2,3),
                        (1,1,1,1), (3,4,2,3), (1,4,2,3), (2,3,3,2)]
            names = ['候选相关评分', '不同有效长度', '空历史', '全padding',
                     '单历史最小维度', '非连续历史', '复制历史强度', '负评分与零mask梯度']
            for i,(b,l,d,h) in enumerate(settings):
                q = rand(b,d); history = rand(b,l,d) if i != 5 else rand(b,d,l).transpose(-2,-1)
                mask = torch.ones(b,l,dtype=torch.bool)
                if i == 1: mask[0,1:] = False; mask[1,-1] = False
                if i == 3: mask.zero_()
                if i == 6: history = history[:, :2].repeat(1,2,1)
                w1,b1,w2,b2 = rand(h,4*d),rand(h),rand(1,h),rand(1)
                if i == 7: w2.zero_(); b2.fill_(-2); mask[0].zero_()
                add(names[i],q,history,mask,w1,b1,w2,b2)
        elif pid == 'augru_cell':
            for i in range(8):
                b,d,h = (1,1,1) if i == 2 else (3,2,4)
                x,hidden = (rand(b,d),rand(b,h)) if i != 5 else (rand(d,b).T,rand(h,b).T)
                attention = torch.rand(b,generator=generator,dtype=dtype)
                if i == 1: attention.zero_()
                if i == 3: attention.fill_(1)
                weights = {name:rand(h,d) for name in ('wr','wz','wn')}
                weights.update({name:rand(h,h) for name in ('ur','uz','un')})
                weights.update({name:rand(h) for name in ('br','bz','bn')})
                if i == 4:
                    for value in weights.values(): value.zero_()
                if i == 6: hidden.zero_()
                if i == 7: weights['bz'].fill_(12); weights['br'].fill_(-8)
                add(['普通门更新', 'attention零保留状态', '标量特征', 'attention一退化GRU',
                     '零权重插值', '非连续双输入', '零初始状态', '饱和门方向'][i], x,hidden,attention,weights)
        elif pid == 'sasrec_block':
            settings = [(2,4,6,2), (2,5,4,2), (1,1,2,1), (2,3,4,1),
                        (1,4,6,3), (2,4,4,2), (1,3,1,1), (2,3,6,2)]
            names = ['多头因果', '左padding与不同长度', '单token', '全部padding',
                     '间隔padding', '非连续输入', '特征维一', '非默认eps与bias']
            for i,(b,t,d,heads) in enumerate(settings):
                x = rand(b,t,d) if i != 5 else rand(b,d,t).transpose(-2,-1)
                mask = torch.ones(b,t,dtype=torch.bool)
                if i == 1: mask[0,:2]=False; mask[1,:1]=False
                if i == 3: mask.zero_()
                if i == 4: mask[:,1::2]=False
                weights = {n:rand(d,d,scale=0.3) for n in ('wq','wk','wv','wo')}
                weights.update({n:rand(d) for n in ('norm1_w','norm1_b','norm2_w','norm2_b')})
                weights.update(w1=rand(d+2,d),b1=rand(d+2),w2=rand(d,d+2),b2=rand(d))
                add(names[i],x,mask,weights,heads,1e-3 if i == 7 else 1e-5)
        elif pid == 'mmoe':
            settings = [(2,3,4,2,5),(3,2,3,4,2),(1,1,1,1,1),(2,3,1,3,4),
                        (2,4,3,1,2),(3,2,4,3,2),(2,3,3,2,4),(2,2,2,3,2)]
            names = ['共享专家独立gate','任务数不同于专家数','最小结构','单expert gate退化',
                     '单任务','非连续输入','均匀gate','极端gate logits']
            for i,(b,d,e,t,h) in enumerate(settings):
                x = rand(b,d) if i != 5 else rand(d,b).T
                w = dict(expert_w=rand(e,h,d),expert_b=rand(e,h),gate_w=rand(t,e,d),gate_b=rand(t,e))
                if i == 6: w['gate_w'].zero_(); w['gate_b'].zero_()
                if i == 7: w['gate_b'] = torch.tensor([[1000.,-1000.],[-1000.,1000.],[0.,0.]],dtype=dtype)
                add(names[i],x,w)
        elif pid == 'esmm_loss':
            datasets = [([0.],[0.],[1.],[1.]), ([0.],[0.],[0.],[0.]),
                        ([1.,-2.,3.,0.],[-1.,2.,0.,1.],[1.,0.,1.,1.],[0.,0.,1.,0.]),
                        ([1000.,-1000.,1000.,-1000.],[1000.,1000.,-1000.,-1000.],[0.,1.,1.,1.],[0.,1.,1.,0.]),
                        ([4.,-3.,0.],[2.,1.,-2.],[0.,0.,0.],[0.,0.,0.]),
                        ([3.,-1.,2.],[-3.,1.,0.],[1.,1.,1.],[1.,0.,1.])]
            for i, arrays in enumerate(datasets):
                add(['零logit正标签','零logit负标签','点击未转化与混合标签','正负1000稳定性',
                     '全未点击','全点击仍用全batch'][i],*[torch.tensor(a,dtype=dtype) for a in arrays],
                    grad_paths=('input[0]','input[1]'))
            add('非连续logits',rand(8)[::2],rand(8)[::2],torch.tensor([1.,0.,1.,0.],dtype=dtype),
                torch.tensor([1.,0.,0.,0.],dtype=dtype),grad_paths=('input[0]','input[1]'))
            add('两tower非对称',rand(5,scale=8),rand(5,scale=0.1),torch.ones(5,dtype=dtype),
                torch.tensor([0.,1.,0.,1.,0.],dtype=dtype),grad_paths=('input[0]','input[1]'))
        elif pid == 'ple_layer':
            settings = [(2,3,2,3,1,4),(3,2,1,2,3,2),(1,1,1,1,1,1),(2,3,2,1,2,3),
                        (2,2,1,1,2,3),(3,4,2,3,2,2),(2,3,2,2,1,3),(1,2,3,1,1,2)]
            names = ['三个流不同输入','不等专家数','最小三流','任务gate隔离','均匀gate',
                     '非连续输入','共享流零输入','gate偏置选择专家']
            for i,(b,d,e1,e2,es,h) in enumerate(settings):
                xs = [rand(b,d) if i != 5 else rand(d,b).T for _ in range(3)]
                if i == 6: xs[2].zero_()
                w = {}
                for name,e in (('task1',e1),('task2',e2),('shared',es)):
                    w[name+'_w'],w[name+'_b'] = rand(e,h,d),rand(e,h)
                for name,e in (('gate1',e1+es),('gate2',e2+es),('gate_shared',e1+e2+es)):
                    w[name+'_w'],w[name+'_b'] = rand(e,d),rand(e)
                    if i == 4: w[name+'_w'].zero_(); w[name+'_b'].zero_()
                    if i == 7: w[name+'_b'] = torch.arange(e,dtype=dtype)*4
                add(names[i],*xs,w)
    return out


def mutations(bank):
    """Each mutation changes one structural rule; reject stale replacement anchors."""
    specifications = {
        'fm_double_count': ('fm_second_order', 'return 0.5 *', 'return 1.0 *'),
        'dcn_bias_inside_cross': ('dcn_cross_layer', 'x0 * (xl @ weight).unsqueeze(-1) + bias + xl',
                                 'x0 * ((xl @ weight).unsqueeze(-1) + bias) + xl'),
        'deepfm_probability_fusion': ('deepfm', 'return wide + fm + deep',
                                      'return torch.sigmoid(wide) + torch.sigmoid(fm) + torch.sigmoid(deep)'),
        'ffm_own_field': ('ffm', 'factors[i, fields[j]] * factors[j, fields[i]]',
                         'factors[i, fields[i]] * factors[j, fields[j]]'),
        'wide_deep_probability_fusion': ('wide_deep', 'return wide + deep',
                                         'return torch.sigmoid(wide) + torch.sigmoid(deep)'),
        'cin_early_embedding_sum': ('cin_layer', 'interaction, weight)',
                                    'interaction.sum(-1, keepdim=True).expand_as(interaction), weight)'),
        'din_softmax_normalization': ('din_pool', 'scores = scores * mask.to(scores.dtype)',
                                      'scores = torch.softmax(scores, dim=1) * mask.to(scores.dtype)'),
        'augru_reversed_update': ('augru_cell', '(1 - update) * hidden + update * n',
                                  'update * hidden + (1 - update) * n'),
        'sasrec_future_leak': ('sasrec_block', "device=x.device).tril()", "device=x.device)"),
        'mmoe_task_softmax': ('mmoe', "+ w['gate_b'], dim=-1)", "+ w['gate_b'], dim=1)"),
        'esmm_direct_cvr_supervision': ('esmm_loss', 'return ctr.mean() + joint.mean()',
                                        'return ctr.mean() + F.binary_cross_entropy_with_logits(v, conversions)'),
        'ple_task2_experts_in_task1': ('ple_layer', 'torch.cat((e1, es), 1)',
                                      'torch.cat((e1, es + e2.mean(1, keepdim=True)), 1)'),
    }
    result = {}
    for name, (pid, before, after) in specifications.items():
        code = bank[pid]['solution']
        if before not in code:
            raise AssertionError('Mutation anchor missing: '+name)
        result[name] = (pid, code.replace(before, after))
    return result
