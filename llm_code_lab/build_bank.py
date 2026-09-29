"""Authoring source: regenerates problems.json and the embedded HTML bank."""
from pathlib import Path
import json
import textwrap

ROOT = Path(__file__).resolve().parent
P = []
SOURCES = {
    'torch': ('PyTorch 官方文档', 'https://docs.pytorch.org/docs/stable/nn.functional.html'),
    'attn': ('Attention Is All You Need', 'https://arxiv.org/abs/1706.03762'),
    'sdpa': ('PyTorch SDPA / Mask 约定', 'https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html'),
    'glu': ('GLU Variants Improve Transformer', 'https://arxiv.org/abs/2002.05202'),
    'rope': ('RoFormer', 'https://arxiv.org/abs/2104.09864'),
    'rms': ('Root Mean Square Layer Normalization', 'https://arxiv.org/abs/1910.07467'),
}

def add(id, title, topic, level, signature, summary, formula, contract, example, hints, pitfalls, followups, solution, sources=('torch',), prerequisites=(), minutes=15, track='llm', priority=None, check_mode=None):
    check_mode = check_mode or ('numeric' if id == 'relu_backward' else 'autograd')
    instruction = ('保留计算图，不要 detach 或转成 NumPy。' if check_mode == 'autograd'
                   else '手写题目要求的计算，不要修改输入。')
    P.append(dict(id=id, number=len(P)+1, title=title, topic=topic, level=level, minutes=minutes,
                  track=track, priority=priority, check_mode=check_mode,
                  summary=summary, formula=formula, contract=contract, example=example,
                  hints=hints, pitfalls=pitfalls, followups=followups, prerequisites=list(prerequisites),
                  starter='import torch\n\n\ndef '+signature+':\n    # 请使用 PyTorch 张量运算实现；保留函数名和参数。\n    # '+instruction+'\n    raise NotImplementedError("在这里开始实现")\n',
                  solution=textwrap.dedent(solution).strip()+'\n',
                  sources=[dict(s) if isinstance(s,dict) else {'title':SOURCES[s][0],'url':SOURCES[s][1]} for s in sources]))

add('relu','ReLU 前向','激活与数值','基础','relu(x)',
    '实现逐元素 ReLU，让负数归零。重点不只是输出正确，还要在反向传播时保留计算图。',
    'y = max(x, 0)；本题约定 x = 0 时导数为 0',
    ['x：任意维度的有限浮点 Tensor；包括标量、非连续张量。','输出与 x 的 shape、dtype、device 相同；禁止原地修改。','建议不直接调用 torch.relu / F.relu；当前版本不强制检查 API 使用。'],
    '输入：[-2, -0.5, 0, 1, 3]\n输出：[ 0,    0, 0, 1, 3]',
    ['先用 x > 0 构造布尔条件。','torch.where 可以选择原值或同形状的零。'],
    ['在 x=0 使用 >= 会改变本题约定的导数。','torch.tensor(existing_tensor) 会断开计算图。'],
    ['为什么 ReLU 在零点不可微，却仍能训练？','torch.clamp 的零点导数是否符合本题约定？'],
    '''
    import torch

    def relu(x):
        # 严格 > 0：明确零点处的反向传播约定。
        return torch.where(x > 0, x, torch.zeros_like(x))
    ''')

add('relu_backward','ReLU 手写反向','激活与数值','基础','relu_backward(x, grad_output)',
    '给定前向输入与上游梯度，计算 ReLU 对输入的梯度。这里训练的是链式法则，不是调用 backward。',
    'grad_x = grad_output × 1[x > 0]',
    ['x 与 grad_output 同 shape、dtype、device。','输出 grad_x；零点使用 0 次梯度。','本题不做二阶梯度评估，仅检查手写反向的数值输出。'],
    'x = [-2, 0, 3]，grad_output = [4, 5, 6]\n期望 grad_x = [0, 0, 6]',
    ['局部导数是一个 0/1 mask。','局部导数还需要乘以上游梯度，不能只返回 mask。'],
    ['漏乘 grad_output。','误将局部导数当作最终梯度。'],
    ['如果上游梯度全为 1，会漏掉哪些实现错误？','如何把它封装成 torch.autograd.Function？'],
    '''
    import torch

    def relu_backward(x, grad_output):
        return grad_output * (x > 0).to(x.dtype)
    ''', prerequisites=('relu',))

add('sigmoid','稳定 Sigmoid','激活与数值','基础','sigmoid(x)',
    '从指数函数实现 Sigmoid，同时避免直接 exp(-x) 在大负值下溢出为无穷大并污染梯度。',
    'σ(x) = 1 / (1 + exp(-x))',
    ['x：有限 float32/float64 Tensor，包含 ±1000。','保持 shape/dtype/device 和计算图；输出范围 [0, 1]。','建议不用 torch.sigmoid，采用数值稳定的等价实现。'],
    '输入：[-1000, 0, 1000]\n输出：[0, 0.5, 1]',
    ['torch.where 两侧表达式都会被计算，不能只在文字上分支。','可以在正支把 x clamp_min(0)，负支把 x clamp_max(0)，确保指数不溢出。'],
    ['先计算 exp(-x) 再 where，反向仍可能出现 NaN。','稳定前向并不自动保证稳定梯度。'],
    ['为什么 exp(-1000) 下溢到 0 通常可以接受？','为什么不能简单把 x 裁剪到 [-10, 10]？'],
    '''
    import torch

    def sigmoid(x):
        pos = 1 / (1 + torch.exp(-x.clamp_min(0)))
        e = torch.exp(x.clamp_max(0))
        neg = e / (1 + e)
        return torch.where(x >= 0, pos, neg)
    ''' )

add('silu','SiLU / Swish','激活与数值','基础','silu(x)',
    '实现 SwiGLU 中使用的 SiLU 激活。允许使用 torch.sigmoid，但不要调用现成的 SiLU。',
    'SiLU(x) = x × σ(x)',
    ['x：任意 shape 的有限浮点 Tensor。','保持输入 dtype/device；支持自动求导。','不是 ReLU；负半轴的输出不必为 0。'],
    '输入：[0, 1]\n输出：[0, 0.7310586]',
    ['先算门值 sigmoid，再与原输入逐元素相乘。','使用乘法运算保留到 x 的两条梯度路径。'],
    ['只返回 sigmoid(x)，漏掉与 x 相乘。','错误使用矩阵乘法 @。'],
    ['口头推导 SiLU 的一阶导数。','SiLU 与 GLU 的结构层级有什么不同？'],
    '''
    import torch

    def silu(x):
        return x * torch.sigmoid(x)
    ''', sources=('torch','glu'), prerequisites=('sigmoid',))

add('gelu','GELU · tanh 近似','激活与数值','进阶','gelu(x)',
    '实现 GELU 的 tanh 近似版。本题明确指定近似公式，不与 erf 精确版混用。',
    'GELU(x) ≈ 0.5x × [1 + tanh(√(2/π) × (x + 0.044715x³))]',
    ['参考标准：F.gelu(x, approximate="tanh")。','输入包含大幅值、零和非连续张量。','允许 torch.tanh；建议不直接使用 F.gelu。'],
    '输入：[0, 1]\n输出：[0, 约 0.841192]',
    ['先核对三次项的系数与括号。','使用 math.sqrt(2 / math.pi) 得到常数。'],
    ['用 erf 版与 tanh 版判题结果可能不同。','把 x³ 写成 3x。'],
    ['近似激活函数在性能与误差上如何取舍？','为什么 ReLU 与 GELU 的负值行为不同？'],
    '''
    import math
    import torch

    def gelu(x):
        z = math.sqrt(2 / math.pi) * (x + 0.044715 * x.pow(3))
        return 0.5 * x * (1 + torch.tanh(z))
    ''', prerequisites=('silu',))

add('softmax','数值稳定 Softmax','激活与数值','进阶','softmax(x, dim=-1)',
    '实现指定维度上的 Softmax。测试不仅检查归一化，还会对随机上游梯度检查反向传播。',
    'pᵢ = exp(xᵢ − max(x)) / Σⱼ exp(xⱼ − max(x))',
    ['归一化维度 dim 可能为 0、1 或 -1。','输入均为有限数，不包含整行 -inf；全屏蔽场景见 Masked Attention。','支持大 logits、非连续 Tensor、单元素维度。'],
    '输入：[1000, 1001, 1002]，dim=-1\n输出：[0.090031, 0.244728, 0.665241]',
    ['沿 dim 求最大值，keepdim=True。','分子分母都沿相同维度，防止错误广播。'],
    ['直接 exp(x) 容易上溢。','只检验 sum(softmax) 的梯度无法检验真实 Jacobian。'],
    ['推导 ∂pᵢ/∂xⱼ。','减去最大值为什么不改变数学结果？'],
    '''
    import torch

    def softmax(x, dim=-1):
        z = x - x.amax(dim=dim, keepdim=True)
        e = z.exp()
        return e / e.sum(dim=dim, keepdim=True)
    ''' )

add('linear','Linear · 最后一维投影','归一化与前馈','基础','linear(x, weight, bias)',
    '实现作用在最后一维的线性层。固定采用 PyTorch 权重布局，为后面的注意力投影打基础。',
    'y = x @ weightᵀ + bias',
    ['x：[..., Din]；weight：[Dout, Din]；bias：[Dout]。','输出：[..., Dout]，保留任意前导 batch/token 维度。','对 x、weight、bias 都检查梯度；不要在函数里重新初始化权重。'],
    'x：[2, 3, 4]，weight：[6, 4]，bias：[6]\n输出：[2, 3, 6]',
    ['权重的最后两维转置，而不是对 x 转置。','@ 自带前导维度广播。'],
    ['把 weight 布局当作 [Din, Dout]。','把 batch 和 token 维误拼进特征维。'],
    ['参数量和时间复杂度分别是多少？','为什么用非方阵测试更容易定位维度错误？'],
    '''
    import torch

    def linear(x, weight, bias):
        return x @ weight.transpose(-1, -2) + bias
    ''')

add('layernorm','LayerNorm','归一化与前馈','进阶','layernorm(x, weight, bias, eps=1e-5)',
    '在最后一个特征维做 LayerNorm。特意加入 D=1、常量输入，检验方差口径和 eps 的位置。',
    'y = (x − mean(x)) / √(mean((x−mean(x))²) + ε) × γ + β',
    ['x：[..., D]；weight、bias：[D]。','只归一化最后一维；方差分母为 D，不是 D−1。','eps 在平方根内；必须保持计算图。'],
    'x：[2, 5, 8]，weight/bias：[8]\n输出：[2, 5, 8]',
    ['mean(..., dim=-1, keepdim=True)。','手写平方偏差均值，或使用 var(unbiased=False)。'],
    ['torch.var 默认估计口径容易与 LayerNorm 不同。','在 batch 维归一化变成了另一种运算。'],
    ['Pre-Norm 与 Post-Norm 改变了哪条梯度路径？','为什么 D=1 是一个有效边界测试？'],
    '''
    import torch

    def layernorm(x, weight, bias, eps=1e-5):
        centered = x - x.mean(dim=-1, keepdim=True)
        var = centered.square().mean(dim=-1, keepdim=True)
        return centered * torch.rsqrt(var + eps) * weight + bias
    ''')

add('rmsnorm','RMSNorm','归一化与前馈','进阶','rmsnorm(x, weight, eps=1e-6)',
    '实现只按均方根缩放的 RMSNorm。与 LayerNorm 对照，明确它没有减均值操作。',
    'y = x / √(mean(x²) + ε) × γ',
    ['x：[..., D]；weight：[D]；无 bias。','只沿最后一维计算均方值，不做中心化。','本版测试 float32/float64；混合精度策略留作扩展。'],
    'x：[1, 1]，weight：[1, 1]，eps=1e-6\n输出：约 [1, 1]，而不是 [0, 0]',
    ['先平方，再沿最后一维求均值。','torch.rsqrt 可以表达 1 / sqrt。'],
    ['不小心减去均值，写成了 LayerNorm。','把 eps 加在 sqrt 外。'],
    ['RMSNorm 对输入缩放具有怎样的近似性质？','低精度训练时为什么会考虑升到 FP32 计算统计量？'],
    '''
    import torch

    def rmsnorm(x, weight, eps=1e-6):
        inv_rms = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)
        return x * inv_rms * weight
    ''', sources=('torch','rms'), prerequisites=('layernorm',))

add('swiglu','SwiGLU FFN','归一化与前馈','进阶','swiglu(x, w_gate, w_up, w_down)',
    '实现完整的 SwiGLU 前馈子层：两路上投影、门控相乘、再下投影。它不只是换一个激活函数。',
    'y = [SiLU(xW_gateᵀ) ⊙ (xW_upᵀ)] W_downᵀ',
    ['x：[B, T, D] 或 [..., D]。','w_gate/w_up：[H, D]；w_down：[D, H]。','不含 bias/dropout/residual/norm；输出与 x 同 shape。','检查 x 与全部三组权重的梯度；权重由参数传入。'],
    'x：[2, 3, 8]，w_gate/w_up：[12, 8]，w_down：[8, 12]\n中间两路：[2, 3, 12]；输出：[2, 3, 8]',
    ['先分别计算 gate 和 up，不能复用同一个权重。','门控乘法是逐元素 *，不是 @。','最后由 H 投影回 D。'],
    ['忘记 w_down，下游 residual 就无法相加。','把 SiLU 作用在两路乘积上，改变了定义。'],
    ['无 bias 时参数量为何是 3DH？','与双线性普通 FFN 等参数比较，应如何选择 H？'],
    '''
    import torch

    def swiglu(x, w_gate, w_up, w_down):
        gate = x @ w_gate.T                    # [..., H]
        up = x @ w_up.T                        # [..., H]
        hidden = (gate * torch.sigmoid(gate)) * up
        return hidden @ w_down.T               # [..., D]
    ''', sources=('glu','torch'), prerequisites=('linear','silu'), minutes=20)

add('attention','Scaled Dot-Product Attention','注意力与位置','进阶','attention(q, k, v)',
    '从 Q、K、V 手写缩放点积注意力。不带 mask，允许 query 与 key 的序列长度不同。',
    'Attention(Q,K,V) = softmax(QKᵀ / √dₖ) V',
    ['q：[B,H,L,Dk]；k：[B,H,S,Dk]；v：[B,H,S,Dv]。','输出：[B,H,L,Dv]；允许 L≠S、Dk≠Dv。','不使用 dropout；建议不直接调用 F.scaled_dot_product_attention。'],
    'q：[2, 3, 4, 8]，k：[2, 3, 6, 8]，v：[2, 3, 6, 5]\nscore：[2, 3, 4, 6]；输出：[2, 3, 4, 5]',
    ['只转置 k 的最后两个维度。','缩放因子来自 q.shape[-1]，不是序列长度。','softmax 归一化 key 的 S 维。'],
    ['在 L 维 softmax。','没有除以 sqrt(Dk)。'],
    ['推导 dense attention 的时间、空间复杂度。','为何 attention 通常不能直接交换 Q 和 K？'],
    '''
    import math
    import torch

    def attention(q, k, v):
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
        scores = scores - scores.amax(dim=-1, keepdim=True)
        p = scores.exp()
        p = p / p.sum(dim=-1, keepdim=True)
        return p @ v
    ''', sources=('attn','sdpa'), prerequisites=('softmax',), minutes=25)

add('masked_attention','Masked Attention','注意力与位置','进阶','masked_attention(q, k, v, mask)',
    '给 Attention 增加可广播的布尔 mask。先把语义定义清楚：本题 True 表示允许关注，整行屏蔽时输出零。',
    'scoreᵢⱼ = QᵢKⱼᵀ / √dₖ；mask=False 的位置不参与归一化',
    ['Q/K/V 布局与 Attention 一致。','mask：bool，广播到 [B,H,L,S]；True=可见。','每一行都被屏蔽时，输出零且梯度有限；本题明确采用此约定。','测试包含 causal 下三角、padding、全屏蔽与广播 mask。'],
    '3×3 causal mask：\n[[True, False, False],\n [True, True,  False],\n [True, True,  True ]]',
    ['把不可见分数替换成 -inf。','全部为 -inf 的行直接 softmax 会出现问题；先检测是否有可见元素。','在 softmax 之前为全屏蔽行准备安全输入，之后再把概率归零。'],
    ['先 softmax 再将被屏蔽概率置零，剩余概率未重归一化。','直接 nan_to_num 可能掩盖反向传播里的 NaN。'],
    ['为什么 SDPA 与 MultiheadAttention 的布尔 mask 语义不能想当然？','padding mask 和 causal mask 如何组合？'],
    '''
    import math
    import torch

    def masked_attention(q, k, v, mask):
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
        scores = scores.masked_fill(~mask, float('-inf'))
        has_key = mask.any(dim=-1, keepdim=True)
        safe_scores = torch.where(has_key, scores, torch.zeros_like(scores))
        p = torch.softmax(safe_scores, dim=-1)
        p = torch.where(mask, p, torch.zeros_like(p))
        return p @ v
    ''', sources=('sdpa','attn'), prerequisites=('attention',), minutes=25)

add('multihead','Causal Multi-Head Attention','注意力与位置','挑战','multihead(x, wq, wk, wv, wo, n_heads)',
    '完整实现因果多头自注意力：投影、拆头、注意力、拼头与输出投影。不包含残差和归一化。',
    'MHA(x) = Concat(head₁, …, headₕ) Wₒᵀ',
    ['x：[B,T,D]；四个权重均为 [D,D]，按 out×in 布局。','D 能被 n_heads 整除；始终使用 causal mask，含对角线。','无 bias/dropout/residual；输出：[B,T,D]。','输入可能非连续；建议用 reshape，而不要假定 view 总能工作。'],
    '[B,T,D] → [B,T,H,d] → [B,H,T,d]\nattention → [B,H,T,d] → [B,T,H,d] → [B,T,D]',
    ['q/k/v 都先投影，再 reshape 并 transpose(1,2)。','注意力缩放使用每头维度 d=D/H。','拼头前先把 T 和 H 换回来。'],
    ['reshape 与 transpose 顺序不对，会混合 token/head。','忘记 causal mask 或输出 wo 投影。'],
    ['为什么增加头数不一定增加投影参数量？','如何测试模型没有偷看未来 token？'],
    '''
    import math
    import torch

    def multihead(x, wq, wk, wv, wo, n_heads):
        b, t, d = x.shape
        dh = d // n_heads
        def split(w):
            return (x @ w.T).reshape(b, t, n_heads, dh).transpose(1, 2)
        q, k, v = split(wq), split(wk), split(wv)
        scores = q @ k.transpose(-2, -1) / math.sqrt(dh)
        visible = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
        scores = scores.masked_fill(~visible, float('-inf'))
        out = torch.softmax(scores, dim=-1) @ v
        out = out.transpose(1, 2).reshape(b, t, d)
        return out @ wo.T
    ''', sources=('attn','sdpa'), prerequisites=('linear','masked_attention'), minutes=35)

add('rope','RoPE · 相邻维旋转','注意力与位置','挑战','rope(x, positions, base=10000.0)',
    '为张量应用旋转位置编码。本题固定使用相邻维配对，不使用前后半维配对，支持非零 position offset。',
    'θₚ,ⱼ = positions[p] / base^(2j/D)\n(x₂ⱼ, x₂ⱼ₊₁) → (x₂ⱼ cosθ − x₂ⱼ₊₁ sinθ, x₂ⱼ sinθ + x₂ⱼ₊₁ cosθ)',
    ['x：[B,H,T,D]；D 为偶数；positions：长度 T 的整数 Tensor。','仅对 x 做旋转并返回同 shape 输出；base>1。','不添加位置向量；位置不一定从零开始，也不一定连续。'],
    'positions 全为 0 时，输出必须等于输入。\n任意 position 下，每对维度的平方和应保持不变。',
    ['构造 arange(0,D,2) / D，计算逆频率。','positions[:,None] 与 inv_freq[None,:] 相乘。','分别取 0::2 与 1::2，旋转后交错合并。'],
    ['使用前后半维配对，与题目约定不一致。','忽略 positions 参数导致 KV cache 偏移错误。'],
    ['为什么旋转能把相对位置带入 QK 点积？','换一种配对布局是否一定数学上错误？'],
    '''
    import torch

    def rope(x, positions, base=10000.0):
        d = x.shape[-1]
        idx = torch.arange(0, d, 2, dtype=x.dtype, device=x.device)
        inv_freq = base ** (-idx / d)
        angle = positions.to(dtype=x.dtype, device=x.device)[:, None] * inv_freq[None, :]
        c, s = angle.cos(), angle.sin()
        even, odd = x[..., 0::2], x[..., 1::2]
        out_even = even * c - odd * s
        out_odd = even * s + odd * c
        return torch.stack((out_even, out_odd), dim=-1).flatten(-2)
    ''', sources=('rope',), prerequisites=('multihead',), minutes=30)

add('gqa','Grouped-Query Attention','注意力与位置','挑战','gqa(q, k, v)',
    '实现 Q 头数多于 KV 头数的分组查询注意力。先验证分组映射，再讨论如何避免物理复制 KV。',
    'group_size = Hq / Hkv；query head h 使用 KV head floor(h / group_size)',
    ['q：[B,Hq,L,Dk]；k：[B,Hkv,S,Dk]；v：[B,Hkv,S,Dv]。','Hq 是 Hkv 的正整数倍；本题无 mask、无 dropout。','输出：[B,Hq,L,Dv]；允许 Hq=Hkv 或 Hkv=1。'],
    'Hq=4，Hkv=2：\nquery heads [0,1] → KV head 0\nquery heads [2,3] → KV head 1',
    ['正确复制顺序是 repeat_interleave，不是 repeat。','复制后可以复用普通注意力的数学逻辑。','本题以正确性为目标，分组 einsum 版本可作为优化。'],
    ['repeat 得到 0,1,0,1；需要的是 0,0,1,1。','使用 Hkv 而不是 Dk 做缩放。'],
    ['GQA 主要节省哪些缓存或访存开销？','如何不复制 KV 实现同样的计算？'],
    '''
    import math
    import torch

    def gqa(q, k, v):
        group = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
        return torch.softmax(scores, dim=-1) @ v
    ''', sources=('sdpa',), prerequisites=('attention','multihead'), minutes=30)

add('kv_cache','KV Cache · 增量注意力','注意力与位置','挑战','kv_cache(q, k_new, v_new, k_cache, v_cache)',
    '在已有 KV 前缀上追加新 token，返回新 token 的因果注意力输出与更新后的缓存。包括多 token chunk，而不只单 token。',
    'K_all = concat(K_cache, K_new)\n新 query i 只能关注 key j ≤ prefix_len + i',
    ['q/k_new：[B,H,N,Dk]；v_new：[B,H,N,Dv]。','k_cache：[B,H,P,Dk]；v_cache：[B,H,P,Dv]；P 可以为 0。','返回 tuple：(out, new_k_cache, new_v_cache)。','out：[B,H,N,Dv]；禁止修改原缓存；本题保留梯度，不 detach。','不做 QKV 投影、RoPE 或预分配；重点是缓存拼接与位置偏移。'],
    'P=3, N=2：\nquery 0 可见 key 0..3\nquery 1 可见 key 0..4',
    ['在 dim=-2 上拼接缓存与新增值。','构造 key_positions <= P + query_index 的 mask。','普通 N×(P+N) 左上对齐 tril 会漏掉合法前缀。'],
    ['直接 is_causal=True，忽略非方阵 mask 的对齐问题。','误认为一个 chunk 内新增 token 可以互相全部可见。'],
    ['为什么 KV cache 不缓存所有历史 Q？','如何比较 full-sequence 与 token-by-token 的输出一致性？'],
    '''
    import math
    import torch

    def kv_cache(q, k_new, v_new, k_cache, v_cache):
        prefix, n = k_cache.shape[-2], q.shape[-2]
        k_all = torch.cat((k_cache, k_new), dim=-2)
        v_all = torch.cat((v_cache, v_new), dim=-2)
        qp = torch.arange(n, device=q.device)[:, None] + prefix
        kp = torch.arange(prefix + n, device=q.device)[None, :]
        scores = q @ k_all.transpose(-2, -1) / math.sqrt(q.shape[-1])
        scores = scores.masked_fill(kp > qp, float('-inf'))
        out = torch.softmax(scores, dim=-1) @ v_all
        return out, k_all, v_all
    ''', sources=('sdpa',), prerequisites=('masked_attention',), minutes=35)

add('cross_entropy','Cross-Entropy from Logits','训练与综合','进阶','cross_entropy(logits, targets)',
    '从 logits 计算多分类交叉熵，返回 batch 平均损失。不要先算 softmax 再取 log，以避免极小概率下溢。',
    'loss = mean(logsumexp(logits, -1) − logits[batch, target])',
    ['logits：[N,C] 有限浮点 Tensor；targets：[N] 的 int64 类别索引。','0≤targets<C；N、C≥1；返回零维 Tensor。','无 ignore_index、label smoothing 或 class weight；只对 logits 检查梯度。'],
    'logits = [[0,0],[0,0]]，targets = [0,1]\nloss = ln(2) ≈ 0.693147',
    ['用 logsumexp 构造稳定的归一化项。','gather 提取每一行真实类别的 logit。'],
    ['sum 和 mean 的 reduction 口径不同。','先 softmax 再 log，可能得到 -inf。'],
    ['推导对 logits 的梯度为何等于 (p−onehot)/N。','如何扩展为 next-token loss，并处理 padding？'],
    '''
    import torch

    def cross_entropy(logits, targets):
        log_z = torch.logsumexp(logits, dim=-1)
        selected = logits.gather(1, targets[:, None]).squeeze(1)
        return (log_z - selected).mean()
    ''', prerequisites=('softmax',), minutes=20)

add('decoder','Pre-Norm Decoder Block','训练与综合','挑战','decoder(x, weights, n_heads, eps=1e-6)',
    '把前面的模块组成一个可反向传播的 Decoder Block：RMSNorm → causal MHA → residual → RMSNorm → SwiGLU → residual。',
    'h = x + MHA(RMSNorm(x))\ny = h + SwiGLU(RMSNorm(h))',
    ['x：[B,T,D]；D 能被 n_heads 整除。','weights 字典：norm1/norm2：[D]；wq/wk/wv/wo：[D,D]。','w_gate/w_up：[F,D]；w_down：[D,F]；所有矩阵采用 out×in。','无 RoPE、dropout、bias、embedding、LM head；这是一个 block，不是完整语言模型。','只提交本题代码；可在同一编辑区定义辅助函数；其他题目的函数不会自动导入。'],
    '输入：[2,4,8]；H=2；F=12\n两次残差连接后输出仍为 [2,4,8]',
    ['先实现辅助函数 norm、attention 和 ffn。','第一次残差后得到 h，第二个 norm 必须用 h。','输出投影 wo 和 w_down 都必须放在对应残差之前。'],
    ['写成 Post-Norm，数值与目标结构不同。','第二个 residual 误加到 x，而不是 h。','函数内新建层导致每次调用重新采样权重。'],
    ['如何从 block 扩展为完整自回归语言模型？','做消融时如何区分结构变化与参数量变化？','迁移到推荐场景时哪些输入、mask 和目标需要改变？'],
    '''
    import math
    import torch

    def decoder(x, weights, n_heads, eps=1e-6):
        def norm(a, scale):
            return a * torch.rsqrt(a.square().mean(-1, keepdim=True) + eps) * scale
        z = norm(x, weights['norm1'])
        b, t, d = z.shape
        dh = d // n_heads
        def project(name):
            return (z @ weights[name].T).reshape(b, t, n_heads, dh).transpose(1, 2)
        q, k, v = project('wq'), project('wk'), project('wv')
        scores = q @ k.transpose(-2, -1) / math.sqrt(dh)
        mask = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
        scores = scores.masked_fill(~mask, float('-inf'))
        attn = (torch.softmax(scores, -1) @ v).transpose(1, 2).reshape(b, t, d)
        h = x + attn @ weights['wo'].T
        z = norm(h, weights['norm2'])
        gate = z @ weights['w_gate'].T
        up = z @ weights['w_up'].T
        ffn = ((gate * torch.sigmoid(gate)) * up) @ weights['w_down'].T
        return h + ffn
    ''', sources=('attn','glu','rms'), prerequisites=('rmsnorm','swiglu','multihead'), minutes=45)

RECOMMENDATION_ORDER = (
    'mse_loss', 'bce_with_logits', 'logistic_regression', 'logistic_regression_backward',
    'weighted_bce', 'focal_loss', 'cross_entropy_extended', 'sgd_momentum', 'adam_step',
    'clip_grad_norm', 'binary_auc', 'group_auc', 'ranking_metrics', 'calibration_error',
    'embedding_mean_pool', 'cosine_score_matrix', 'bpr_loss', 'inbatch_softmax_loss',
    'sampled_softmax_loss', 'fm_second_order', 'dcn_cross_layer', 'deepfm', 'ffm',
    'wide_deep', 'cin_layer', 'din_pool', 'augru_cell', 'sasrec_block', 'mmoe',
    'esmm_loss', 'ple_layer', 'ips_snips',
)


LLM_BLOCK_ORDER = (
    'lora_linear', 'lora_merge', 'sinusoidal_positions', 'alibi_bias',
    'sliding_window_attention', 'rope_scaling', 'cross_attention', 'rmsnorm_backward',
    'moe_topk_routing', 'moe_layer', 'moe_load_balance', 'kv_cache_update',
)

LLM_TRAINING_ORDER = (
    'temperature_softmax', 'topk_sampling_probs', 'topp_sampling_probs',
    'repetition_penalty', 'causal_lm_loss', 'sequence_log_probs',
    'kl_distillation_loss', 'dpo_loss', 'reward_pairwise_loss', 'adamw_step',
    'gradient_accumulation', 'token_perplexity',
)

REINFORCEMENT_ORDER = (
    'discounted_returns', 'td0_target', 'gae_advantages', 'normalize_advantages',
    'reinforce_loss', 'categorical_entropy', 'importance_sampling_ratio',
    'ppo_clipped_loss', 'ppo_value_loss', 'dqn_loss', 'double_dqn_target',
    'polyak_update', 'value_iteration_step', 'c51_projection',
    'grpo_advantages', 'grpo_loss',
)


def register_group(module_names, expected_order):
    import importlib
    original_count = len(P)
    for name in module_names:
        importlib.import_module(name).register(add)
    order = {pid:i for i,pid in enumerate(expected_order)}
    added = P[original_count:]
    if len(added) != len(order) or {p['id'] for p in added} != set(order):
        raise ValueError(f'{module_names} 的题库与规划题号不一致')
    P[original_count:] = sorted(added, key=lambda p:order[p['id']])


def validate_catalog():
    ids = {p['id'] for p in P}
    if len(ids) != len(P): raise ValueError('题目 ID 重复')
    for number,p in enumerate(P,1):
        p['number'] = number
        if not set(p['prerequisites']) <= ids:
            raise ValueError(f"{p['id']} 的前置题不存在")
        if p['track'] not in ('llm', 'recsys', 'rl'):
            raise ValueError(f"{p['id']} 的学习方向无效")
        if p['check_mode'] not in ('autograd', 'numeric'):
            raise ValueError(f"{p['id']} 的判题方式无效")

    by_id = {p['id']:p for p in P}
    visited, visiting = set(), set()
    def visit(pid):
        if pid in visiting: raise ValueError(f'{pid} 的前置题存在循环')
        if pid in visited: return
        visiting.add(pid)
        for prerequisite in by_id[pid]['prerequisites']: visit(prerequisite)
        visiting.remove(pid)
        visited.add(pid)
    for pid in by_id: visit(pid)


register_group(('rec_basics_bank', 'rec_retrieval_bank', 'rec_models_bank'), RECOMMENDATION_ORDER)
register_group(('llm_blocks_bank',), LLM_BLOCK_ORDER)
register_group(('llm_training_bank',), LLM_TRAINING_ORDER)
register_group(('rl_bank',), REINFORCEMENT_ORDER)
validate_catalog()


if __name__ == '__main__':
    raw = json.dumps(P, ensure_ascii=False, indent=2)
    (ROOT / 'problems.json').write_text(raw, encoding='utf-8')
    template = ROOT / 'ui.template.html'
    if template.exists():
        (ROOT / 'index.html').write_text(template.read_text(encoding='utf-8').replace('__PROBLEMS_JSON__', raw.replace('</', '<\\/')), encoding='utf-8')
    print(f'Built {len(P)} problems')
