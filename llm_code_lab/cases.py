"""Small CPU test tensors and independent reference implementations.
Local practice tests are inspectable; they are NOT secret contest tests.
"""
import math
import torch
import torch.nn.functional as F
from functools import lru_cache


@lru_cache(maxsize=1)
def recommendation_modules():
    import importlib
    return tuple(importlib.import_module(name) for name in
        ('rec_basics_cases', 'rec_retrieval_cases', 'rec_models_cases'))


@lru_cache(maxsize=1)
def case_modules():
    import importlib
    return recommendation_modules() + tuple(importlib.import_module(name) for name in
        ('llm_blocks_cases', 'llm_training_cases', 'rl_cases'))


def reference(pid, *a):
    if pid == 'relu': return F.relu(a[0])
    if pid == 'relu_backward': return torch.where(a[0] > 0, a[1], 0.)
    if pid == 'sigmoid': return torch.sigmoid(a[0])
    if pid == 'silu': return F.silu(a[0])
    if pid == 'gelu': return F.gelu(a[0], approximate='tanh')
    if pid == 'softmax': return F.softmax(a[0], dim=a[1])
    if pid == 'linear': return F.linear(*a)
    if pid == 'layernorm':
        x, w, b, eps = a
        return F.layer_norm(x, (x.shape[-1],), w, b, eps)
    if pid == 'rmsnorm':
        x, w, eps = a
        return x / torch.sqrt(torch.mean(torch.pow(x, 2), dim=-1, keepdim=True) + eps) * w
    if pid == 'swiglu':
        x, gate, up, down = a
        return F.linear(F.silu(F.linear(x, gate)) * F.linear(x, up), down)
    if pid == 'attention':
        return F.scaled_dot_product_attention(*a, dropout_p=0.0)
    if pid == 'masked_attention':
        q, k, v, mask = a
        # Explicit all-masked convention also works on older SDPA versions.
        has_key = mask.any(dim=-1, keepdim=True)
        safe_mask = torch.where(has_key, mask, torch.ones_like(mask))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=safe_mask, dropout_p=0.0)
        return torch.where(has_key, out, torch.zeros_like(out))
    if pid == 'multihead':
        x, wq, wk, wv, wo, heads = a
        b, t, d = x.shape
        # Split with unflatten/movedim rather than the student's reshape path.
        q, k, v = [F.linear(x, w).unflatten(-1, (heads, d // heads)).movedim(-2, 1)
                   for w in (wq, wk, wv)]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        return F.linear(y.movedim(1, -2).flatten(-2), wo)
    if pid == 'rope':
        x, pos, base = a
        d = x.shape[-1]
        # Explicit small 2x2 rotations form an independent layout oracle.
        pairs = x.unflatten(-1, (d // 2, 2))
        exponent = torch.arange(d // 2, dtype=x.dtype, device=x.device) * (2.0 / d)
        angles = pos.to(x.dtype).unsqueeze(-1) / torch.pow(base, exponent)
        c, s = torch.cos(angles), torch.sin(angles)
        matrix = torch.stack((c, -s, s, c), -1).reshape(*angles.shape, 2, 2)
        return torch.matmul(matrix, pairs.unsqueeze(-1)).squeeze(-1).flatten(-2)
    if pid == 'gqa':
        q, k, v = a
        group = q.shape[1] // k.shape[1]
        # Per-head oracle avoids relying on repeat_interleave for the mapping.
        heads = []
        for h in range(q.shape[1]):
            kv = h // group
            heads.append(F.scaled_dot_product_attention(q[:, h:h+1], k[:, kv:kv+1],
                                                        v[:, kv:kv+1], dropout_p=0.0))
        return torch.cat(heads, 1)
    if pid == 'kv_cache':
        q, kn, vn, kc, vc = a
        k, v = torch.cat([kc, kn], -2), torch.cat([vc, vn], -2)
        prefix = kc.shape[-2]
        # Each query attends a separately truncated prefix; no causal mask oracle reuse.
        ys = [F.scaled_dot_product_attention(q[..., i:i+1, :], k[..., :prefix+i+1, :],
                                             v[..., :prefix+i+1, :], dropout_p=0.0)
              for i in range(q.shape[-2])]
        return torch.cat(ys, -2), k, v
    if pid == 'cross_entropy': return F.cross_entropy(*a, reduction='mean')
    if pid == 'decoder':
        x, w, heads, eps = a
        def norm(z, weight):
            return z / torch.sqrt((z*z).mean(-1, keepdim=True) + eps) * weight
        h = x + reference('multihead', norm(x, w['norm1']), w['wq'], w['wk'], w['wv'], w['wo'], heads)
        return h + reference('swiglu', norm(h, w['norm2']), w['w_gate'], w['w_up'], w['w_down'])
    for module in case_modules():
        if pid in module.IDS: return module.reference(pid, *a)
    raise KeyError(pid)


def make_cases(pid, seed=1729):
    """The first two cases are the public run set; submit evaluates all cases."""
    g = torch.Generator().manual_seed(seed)
    out = []
    def rand(shape, dtype=torch.float64, scale=1.0):
        return torch.randn(shape, generator=g, dtype=dtype) * scale
    def add(name, *args, grad=True):
        out.append({'name':name, 'args':args, 'grad':grad and pid != 'relu_backward'})
    if pid in ('relu','sigmoid','silu','gelu','relu_backward'):
        for dtype in (torch.float64, torch.float32):
            suffix = str(dtype).split('.')[-1]
            data = [
                ('正负值与零点', torch.tensor([-2., -.5, 0., 1., 3.], dtype=dtype)),
                ('多维随机', rand((2, 3, 5), dtype)),
                ('标量', torch.tensor(-.7, dtype=dtype)),
                ('非连续输入', rand((3, 7), dtype).T),
                ('极端幅值', torch.tensor([-1000., -100., 0., 100., 1000.], dtype=dtype)),
                ('全零', torch.zeros(4, dtype=dtype)),
            ]
            for name, x in data:
                a = (x, rand(x.shape, dtype)) if pid == 'relu_backward' else (x,)
                add(f'{name} · {suffix}', *a)
    elif pid == 'softmax':
        for dtype in (torch.float64, torch.float32):
            s = str(dtype).split('.')[-1]
            add('大 logits · '+s, torch.tensor([[1000.,1001.,1002.],[-1000.,-1000.,-1000.]],dtype=dtype), -1)
            add('多维随机 · '+s, rand((2,3,5),dtype), -1)
            add('dim=0 · '+s, rand((3,5),dtype), 0)
            add('dim=1 · '+s, rand((2,3,5),dtype), 1)
            add('非连续 · '+s, rand((5,3),dtype).T, -1)
            add('单元素维度 · '+s, rand((3,1),dtype), -1)
    elif pid in ('linear','layernorm','rmsnorm','swiglu'):
        for dtype in (torch.float64, torch.float32):
            s = str(dtype).split('.')[-1]
            samples = [('多维随机',rand((2,3,5),dtype)), ('二维非方阵',rand((3,7),dtype)),
                       ('特征维=1',rand((2,1),dtype)), ('常量输入',torch.ones((2,4),dtype=dtype)),
                       ('全零输入',torch.zeros((2,3),dtype=dtype)), ('非连续',rand((7,3),dtype).T)]
            for i,(name,x) in enumerate(samples):
                d = x.shape[-1]
                if pid == 'linear': a = (x, rand((d+2,d),dtype,.3), rand((d+2,),dtype))
                elif pid == 'layernorm': a = (x, rand((d,),dtype), rand((d,),dtype), 1e-5 if i%2 else 1e-3)
                elif pid == 'rmsnorm': a = (x,rand((d,),dtype),1e-6 if i%2 else 1e-3)
                else:
                    f=d+3
                    a = (x,rand((f,d),dtype,.3),rand((f,d),dtype,.3),rand((d,f),dtype,.3))
                add(name+' · '+s,*a)
    elif pid in ('attention','masked_attention','gqa'):
        for dtype in (torch.float64,torch.float32):
            s = str(dtype).split('.')[-1]
            for i,(b,h,l,klen,d,dv) in enumerate([(1,2,3,3,4,4),(2,4,3,5,6,3),(1,1,1,4,3,2),(2,2,4,1,5,7),(1,4,5,5,3,2),(1,2,2,6,4,3)]):
                hkv = max(1,h//2) if pid=='gqa' else h
                q,k,v = rand((b,h,l,d),dtype),rand((b,hkv,klen,d),dtype),rand((b,hkv,klen,dv),dtype)
                if i==5:
                    q = rand((b,h,d,l),dtype).transpose(-2,-1)
                if pid == 'masked_attention':
                    if i==0: mask=torch.ones(l,klen,dtype=torch.bool).tril(); name='因果 mask'
                    elif i==1:
                        mask=torch.ones(b,1,1,klen,dtype=torch.bool); mask[...,-2:]=False; name='padding 广播'
                    elif i==2: mask=torch.zeros(l,klen,dtype=torch.bool); name='全屏蔽输出与梯度'
                    elif i==3: mask=torch.ones(l,klen,dtype=torch.bool); name='单 key'
                    elif i==4:
                        mask=torch.ones(l,klen,dtype=torch.bool).tril(); mask[2,:]=False; name='部分行全屏蔽'
                    else: mask=torch.rand((b,h,l,klen),generator=g)>.4; name='非连续与随机 mask'
                    add(name+' · '+s,q,k,v,mask)
                else:
                    name=['常规注意力','L≠S 且 Dk≠Dv','单 query / KV 头','单 key','多 token','非连续 Q'][i]
                    add(name+' · '+s,q,k,v)
    elif pid in ('multihead','decoder'):
        for dtype in (torch.float64,torch.float32):
            s=str(dtype).split('.')[-1]
            for i,(b,t,d,h) in enumerate([(2,3,8,2),(1,5,12,3),(1,1,4,1),(2,4,8,4),(1,3,6,2)]):
                x = rand((b,t,d),dtype) if i!=4 else rand((b,d,t),dtype).transpose(-2,-1)
                w={name:rand((d,d),dtype,.2) for name in ('wq','wk','wv','wo')}
                if pid=='multihead': a=(x,w['wq'],w['wk'],w['wv'],w['wo'],h)
                else:
                    f=d+5
                    w.update(norm1=rand((d,),dtype),norm2=rand((d,),dtype),w_gate=rand((f,d),dtype,.2),
                             w_up=rand((f,d),dtype,.2),w_down=rand((d,f),dtype,.2))
                    a=(x,w,h,1e-6)
                add(['标准结构','不同头数','单 token','多头因果','非连续输入'][i]+' · '+s,*a)
    elif pid=='rope':
        for dtype in (torch.float64,torch.float32):
            s=str(dtype).split('.')[-1]
            for i,(b,h,t,d,base) in enumerate([(1,2,4,8,10000.),(2,3,3,6,10000.),(1,1,1,2,100.),(1,2,4,10,500.),(1,2,3,4,10000.)]):
                x=rand((b,h,t,d),dtype) if i!=4 else rand((b,h,d,t),dtype).transpose(-2,-1)
                pos=torch.zeros(t,dtype=torch.long) if i==0 else torch.arange(t,dtype=torch.long)*(i+1)+7
                add(['位置全零','非零 offset','最小旋转对','非标准 base','非连续输入'][i]+' · '+s,x,pos,base)
    elif pid=='kv_cache':
        for dtype in (torch.float64,torch.float32):
            s=str(dtype).split('.')[-1]
            for i,(p,n) in enumerate([(3,1),(3,2),(0,3),(0,1),(5,3)]):
                b,h,d,dv=2,2,4,3
                add(['单 token 解码','chunk 因果偏移','空缓存 prefill','空缓存单 token','长前缀 chunk'][i]+' · '+s,
                    rand((b,h,n,d),dtype),rand((b,h,n,d),dtype),rand((b,h,n,dv),dtype),
                    rand((b,h,p,d),dtype),rand((b,h,p,dv),dtype))
    elif pid=='cross_entropy':
        for dtype in (torch.float64,torch.float32):
            s=str(dtype).split('.')[-1]
            add('均匀 logits · '+s,torch.zeros(2,2,dtype=dtype),torch.tensor([0,1]))
            add('随机分类 · '+s,rand((4,7),dtype),torch.tensor([1,3,0,6]))
            add('极端 logits · '+s,torch.tensor([[1000.,-1000.,0.],[-1000.,1000.,0.]],dtype=dtype),torch.tensor([1,0]))
            add('单类别 · '+s,rand((3,1),dtype),torch.zeros(3,dtype=torch.long))
            add('非连续 · '+s,rand((7,4),dtype).T,torch.tensor([0,1,2,3]))
    else:
        for module in case_modules():
            if pid in module.IDS: return module.make_cases(pid, seed)
        raise KeyError(pid)
    return out
