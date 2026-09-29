"""LLM implementation exercises: adapters, positions, attention and sparse experts."""


def register(add):
    sources = {
        'lora': ('LoRA 原论文 §4.1', 'https://arxiv.org/html/2106.09685v2#S4.SS1'),
        'transformer': ('Attention Is All You Need', 'https://arxiv.org/abs/1706.03762'),
        'alibi': ('ALiBi 原论文 §3', 'https://arxiv.org/html/2108.12409v2#S3'),
        'pi': ('Position Interpolation 原论文 §2.3', 'https://arxiv.org/html/2306.15595v2#S2.SS3'),
        'sdpa': ('PyTorch SDPA 官方定义', 'https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html'),
        'rms': ('Root Mean Square Layer Normalization', 'https://arxiv.org/abs/1910.07467'),
        'moe': ('Sparsely-Gated Mixture-of-Experts 原论文', 'https://arxiv.org/abs/1701.06538'),
        'switch': ('Switch Transformers 辅助损失式 (4)–(6)', 'https://arxiv.org/html/2101.03961v3#S2.SS2'),
        'cache': ('Hugging Face KV cache 概念文档', 'https://huggingface.co/docs/transformers/cache_explanation'),
    }

    def put(pid, title, topic, level, signature, summary, formula, contract,
            example, hints, pitfalls, followups, solution, *, source,
            priority='P1', check_mode='autograd', prerequisites=(), minutes=25):
        title_, url = sources[source]
        add(pid, title, topic, level, signature, summary, formula, contract,
            example, hints, pitfalls, followups, solution,
            sources=({'title': title_, 'url': url},), track='llm', priority=priority,
            check_mode=check_mode, prerequisites=prerequisites, minutes=minutes)

    put('lora_linear', 'LoRA · 低秩分支前向', '参数高效微调', '进阶',
        'lora_linear(x, base_weight, lora_a, lora_b, alpha=1.0)',
        '在冻结的线性投影上加入可训练的低秩分支，明确矩阵方向、rank 缩放与输入梯度。',
        'y = xW₀ᵀ + (α/r)·(xAᵀ)Bᵀ；ΔW = (α/r)BA',
        ['x：[...,Din]，至少一维；base_weight：[Dout,Din]；lora_a：[r,Din]；lora_b：[Dout,r]；维度均≥1。',
         '所有 Tensor 同 dtype/device，float32/float64；alpha≥0；输出 [...,Dout]，不含 bias 或 dropout。',
         'base_weight 是固定参数；只检查 x、lora_a、lora_b 的梯度。冻结 W₀ 不意味着可以切断 x 经过基础分支的梯度。',
         '支持 B 全零的常见初始化和 alpha=0；不能原地改权重，不在函数内随机初始化。',
         '建议直接实现两次低秩投影；判题验证数值和梯度，不证明实际计算复杂度。'],
        '例 1：x=[[2,3]]，W₀=[[1,0]]，A=[[0,1]]，B=[[2]]，α=1 → [[8]]。\n例 2：B=0 时输出等于 xW₀ᵀ，但 B 一般仍有非零梯度。',
        ['先按 PyTorch 布局写出基础分支 x @ base_weight.T。', '低秩支路先 Din→r，再 r→Dout，最后乘 alpha/r。', '不要对基础支路输出 detach：它还需要对 x 求导。'],
        ['漏掉除以 r。', '把 BA 的乘法方向反过来。', '冻结权重时误把整个基础分支切断计算图。'],
        ['为什么常见初始化把 B 设为零而 A 随机？', '相较全参数微调，训练参数量减少到多少？'],
        '''
        import torch

        def lora_linear(x, base_weight, lora_a, lora_b, alpha=1.0):
            rank = lora_a.shape[0]
            base = x @ base_weight.T
            update = (x @ lora_a.T) @ lora_b.T
            return base + (alpha / rank) * update
        ''', source='lora', priority='P0', prerequisites=('linear',))

    put('lora_merge', 'LoRA · 推理权重合并', '参数高效微调', '基础',
        'lora_merge(base_weight, lora_a, lora_b, alpha=1.0)',
        '把低秩增量合并为一张推理权重，验证合并后线性投影与双分支前向等价。',
        'W_merged = W₀ + (α/r)BA',
        ['base_weight：[Dout,Din]；lora_a：[r,Din]；lora_b：[Dout,r]；所有尺寸≥1，alpha≥0。',
         '返回新的 [Dout,Din] 浮点 Tensor，dtype/device 与 base_weight 一致；不能原地修改任一输入。',
         '仅做数值检查；本题不含 bias、量化、dropout 或反合并，不改变存储中的原权重。'],
        '例 1：W₀=[[1,2]]，A=[[3,4]]，B=[[2]]，α=0.5 → W_merged=[[4,6]]。\n例 2：alpha=0 或 B=0 时，新权重的数值与 W₀ 相同。',
        ['先检查 B @ A 的 shape 是否与 W₀ 相同。', 'rank 从 A.shape[0] 读取，不是从输入维度读取。'],
        ['用 W₀ += update 修改原权重，反复调用后累积增量。', '误把 alpha/r 写成 alpha 或 alpha/sqrt(r)。'],
        ['合并后的计算量与普通 Linear 有何关系？', '多任务切换与量化权重会带来哪些合并问题？'],
        '''
        import torch

        def lora_merge(base_weight, lora_a, lora_b, alpha=1.0):
            return base_weight + (alpha / lora_a.shape[0]) * (lora_b @ lora_a)
        ''', source='lora', check_mode='numeric', prerequisites=('linear',), minutes=15)

    put('sinusoidal_positions', '正弦位置编码 · 奇偶通道', '位置编码与长上下文', '基础',
        'sinusoidal_positions(positions, dim, base=10000.0)',
        '从给定位置生成 Transformer 正弦位置编码，明确频率配对、奇数维度和 dtype。',
        'PE(p,2i)=sin(p/base^(2i/dim))；PE(p,2i+1)=cos(p/base^(2i/dim))',
        ['positions：[T] 非负有限浮点 Tensor，可含非整数位置；dim 为正整数，允许奇数；base>1。',
         '输出 [T,dim]，dtype/device 与 positions 一致；T=0 时返回 [0,dim]。',
         '偶数通道用 sin、奇数通道用 cos，同一对使用同一频率；奇数 dim 的最后一维为 sin。',
         '仅检查数值；生成位置编码而不把它加到 token embedding 上。'],
        '例 1：positions=[0]，dim=5 → [[0,1,0,1,0]]。\n例 2：positions=[1]，dim=2 → [[sin(1),cos(1)]]。',
        ['频率索引使用 0,2,4,...，再除以 dim。', '把 sin/cos 沿最后一维交错排列，奇数 dim 时截掉最后多出的 cos。'],
        ['sin 与 cos 通道排列颠倒。', '奇数 dim 时错误广播或多返回一维。', '默认创建 float32 张量，破坏输入精度。'],
        ['为什么同一对 sin/cos 需要相同频率？', '绝对位置编码和 RoPE 注入位置信息的位置有什么不同？'],
        '''
        import torch

        def sinusoidal_positions(positions, dim, base=10000.0):
            exponent = torch.arange(0, dim, 2, dtype=positions.dtype, device=positions.device) / dim
            angles = positions[:, None] / torch.pow(base, exponent)
            pairs = torch.stack((angles.sin(), angles.cos()), dim=-1)
            return pairs.flatten(-2)[:, :dim]
        ''', source='transformer', priority='P0', check_mode='numeric', minutes=20)

    put('alibi_bias', 'ALiBi · 每头线性相对偏置', '位置编码与长上下文', '进阶',
        'alibi_bias(query_positions, key_positions, slopes)',
        '生成加入 attention logits 的线性位置偏置，支持缓存解码的绝对位置偏移。斜率由调用方给定。',
        'bias[h,i,j] = slopes[h] · (key_positions[j] − query_positions[i])',
        ['query_positions：[Tq]、key_positions：[Tk]，非负 int64；允许不连续位置、Tq≠Tk；slopes：[H] 非负浮点 Tensor，H≥1。',
         '返回 [H,Tq,Tk]，dtype/device 与 slopes 一致；所有输入同 device；允许某一位置序列为空。',
         '这里只输出有限的有符号偏置，不含 causal -inf。key≤query 的可见区域对应原论文距离惩罚；未来位置必须由调用方另外屏蔽。',
         '偏置应加在已经按 sqrt(D) 缩放的 QK 分数上，本函数不再除以 sqrt(D)；不生成或训练 slopes。',
         '数值检查，不评估梯度；这不是双向 attention 的绝对距离变体。'],
        '例 1：qpos=[2]，kpos=[0,1,2]，slopes=[0.5] → [[[-1,-0.5,0]]]。\n例 2：qpos=[3]，kpos=[4]，slopes=[0.25] → [[[0.25]]]；此未来位置仍需另加 causal mask。',
        ['先构造 [Tq,Tk] 的 key−query 差值。', '在前面插入 head 维，与 slopes[:,None,None] 广播相乘。'],
        ['把 query−key 写成惩罚，导致过去越远分数越高。', '缓存解码时使用从零重置的 query 索引。', '在此返回 -inf，混淆有限偏置与可见性 mask。'],
        ['为什么 ALiBi 不需要加到 V 上？', '若改为双向注意力，绝对距离变体要怎样明确题面？'],
        '''
        import torch

        def alibi_bias(query_positions, key_positions, slopes):
            relative = key_positions[None, :] - query_positions[:, None]
            return slopes[:, None, None] * relative.to(slopes.dtype)[None, :, :]
        ''', source='alibi', check_mode='numeric', prerequisites=('attention',))

    put('sliding_window_attention', '滑动窗口因果 Attention', '注意力与缓存', '进阶',
        'sliding_window_attention(q, k, v, query_positions, key_positions, window, key_mask)',
        '把因果约束、局部窗口和 padding 组合为同一个可见性条件，覆盖有缓存偏移和全屏蔽行。',
        'visible[b,i,j] = key_mask[b,j] AND 0 ≤ qpos[i]−kpos[j] < window\ny = softmax(QKᵀ/√D + mask) V；没有可见 key 的行输出 0',
        ['q：[B,H,Tq,D]；k：[B,H,Tk,D]；v：[B,H,Tk,Dv]；所有尺寸≥1；返回 [B,H,Tq,Dv]。',
         'query_positions：[Tq]、key_positions：[Tk] int64 非负绝对位置；window 为正整数，按位置差计数。',
         'key_mask：[B,Tk] bool，True 可用；可见区间是 qpos−window < kpos ≤ qpos，包括当前位置。',
         '全屏蔽行返回有限零并保留 q/k/v 的零梯度连接；只检查 q/k/v 梯度。',
         '不含 dropout、投影或残差。教学答案构造完整 mask，不宣称具有稀疏 kernel 的复杂度。'],
        '例 1：qpos=[3]，kpos=[0,1,2,3]，window=2 → 只看位置 2、3。\n例 2：window=1 且不存在相同位置的有效 key → 该 query 输出零。',
        ['不要用矩阵索引 i−j 替代绝对位置差，缓存偏移时会错。', '先合并三种条件，再屏蔽 attention 分数。', '全屏蔽行在 softmax 前使用安全临时值，归一化后再把无效概率清零。'],
        ['使用 ≤window，多看了一个位置。', '遗漏因果条件，泄露未来 token。', '对全 -inf 行直接 softmax 得到 NaN。'],
        ['如何实现不显式构造 Tq×Tk 分数矩阵的窗口 kernel？', '跨层堆叠局部 attention 如何扩大有效感受野？'],
        '''
        import math
        import torch

        def sliding_window_attention(q, k, v, query_positions, key_positions, window, key_mask):
            distance = query_positions[:, None] - key_positions[None, :]
            local = (distance >= 0) & (distance < window)
            visible = local[None, None, :, :] & key_mask[:, None, None, :]
            scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
            has_key = visible.any(-1, keepdim=True)
            safe = torch.where(has_key, scores.masked_fill(~visible, -torch.inf), torch.zeros_like(scores))
            probabilities = torch.softmax(safe, dim=-1)
            probabilities = torch.where(visible, probabilities, torch.zeros_like(probabilities))
            return probabilities @ v
        ''', source='sdpa', prerequisites=('masked_attention', 'kv_cache'), minutes=35)

    put('rope_scaling', 'RoPE · 位置线性插值', '位置编码与长上下文', '进阶',
        'rope_scaling(x, positions, scale=1.0, base=10000.0)',
        '在相邻两维配对的 RoPE 上实现 Position Interpolation：缩放位置索引，再进行旋转。',
        'θᵢ = (position/scale) · base^(−2i/D)；[a,b] → [a cosθ−b sinθ, a sinθ+b cosθ]',
        ['x：[B,H,T,D]，B/H/T≥1，D 为正偶数；positions：[T] 同 device 的非负有限 int64 或浮点 Tensor。',
         'scale≥1，base>1；输出与 x 同 shape/dtype/device；只检查 x 梯度，positions 固定。',
         '相邻维度 (0,1)、(2,3)... 配对；先转成 x.dtype 再做 positions/scale，不能使用整数除法。',
         'scale=1 恢复普通 RoPE；这是 PI 的线性位置插值，不是修改 base 的 NTK scaling 或 YaRN。',
         '本题只实现位置变换；不能凭通过此题断言任意模型无须训练即可可靠扩展上下文。'],
        '例 1：x=[[[[1,0]]]]，positions=[2]，scale=2 → [[[[cos(1),sin(1)]]]]。\n例 2：positions 全为 0 时输出等于 x，任意合法 scale 均成立。',
        ['频率仍是 base^(−2i/D)，只把位置除以 scale。', '取 x[...,0::2] 和 x[...,1::2]，计算后交错拼回。'],
        ['把 position 乘以 scale，反而扩大相位。', '对 positions 整除，丢掉插值所需的小数。', '将相邻配对与前后半维配对混用。'],
        ['PI 如何改变可见的最大相对位置？', '修改 base 与统一缩放位置对不同频率的影响是否相同？'],
        '''
        import torch

        def rope_scaling(x, positions, scale=1.0, base=10000.0):
            dim = x.shape[-1]
            exponent = torch.arange(0, dim, 2, dtype=x.dtype, device=x.device) / dim
            angles = (positions.to(x.dtype) / scale)[:, None] / torch.pow(base, exponent)
            c, s = angles.cos(), angles.sin()
            even, odd = x[..., 0::2], x[..., 1::2]
            return torch.stack((even * c - odd * s, even * s + odd * c), dim=-1).flatten(-2)
        ''', source='pi', prerequisites=('rope',))

    put('cross_attention', 'Cross Attention · 跨序列多头投影', '注意力与缓存', '进阶',
        'cross_attention(query, context, wq, wk, wv, wo, num_heads, key_mask)',
        '实现 decoder 查询另一段上下文的多头注意力，区分 Q 与 K/V 的来源、序列长度和输入维度。',
        'Q=query·Wqᵀ；K=context·Wkᵀ；V=context·Wvᵀ\ny=Concat(heads(softmax(QKᵀ/√d) V))·Woᵀ',
        ['query：[B,Tq,D]；context：[B,Tk,Dctx]；wq、wo：[D,D]；wk、wv：[D,Dctx]。',
         'num_heads≥1 且整除 D；key_mask：[B,Tk] bool，True 可见；全部尺寸≥1，Tq 与 Tk 可不同。',
         '输出 [B,Tq,D]，全部浮点输入检查梯度；无 bias/dropout/残差，不加 causal mask。',
         '某个 batch 的全部 key 无效时，其全部 query 输出零并保留计算图；mask 在每个 head 共享。'],
        '例 1：query [2,3,8]、context [2,5,6]、heads=2 → 每头分数 [2,2,3,5]，输出 [2,3,8]。\n例 2：Tk=1 且唯一 key 有效时，每个 query 都取得同一个 V 投影后再过 Wo。',
        ['只有 Q 来自 query，K 和 V 都来自 context。', '拆 head 时保留各自 Tq/Tk，合并 head 后再乘 wo.T。', '全 mask 情况需要显式安全 softmax。'],
        ['复用 self-attention 代码时把 K/V 仍从 query 计算。', '错误加入下三角 mask。', '把 Dctx 当作每头维度来缩放。'],
        ['编码器输出固定时，哪些投影可以在生成过程中缓存？', 'Cross attention 与 self attention 的 KV cache 生命周期有何不同？'],
        '''
        import math
        import torch

        def cross_attention(query, context, wq, wk, wv, wo, num_heads, key_mask):
            batch, tq, dim = query.shape
            tk = context.shape[1]
            head_dim = dim // num_heads
            q = (query @ wq.T).reshape(batch, tq, num_heads, head_dim).transpose(1, 2)
            k = (context @ wk.T).reshape(batch, tk, num_heads, head_dim).transpose(1, 2)
            v = (context @ wv.T).reshape(batch, tk, num_heads, head_dim).transpose(1, 2)
            visible = key_mask[:, None, None, :]
            scores = q @ k.transpose(-1, -2) / math.sqrt(head_dim)
            safe = torch.where(visible.any(-1, keepdim=True),
                               scores.masked_fill(~visible, -torch.inf), torch.zeros_like(scores))
            probabilities = torch.where(visible, torch.softmax(safe, -1), torch.zeros_like(scores))
            heads = probabilities @ v
            joined = heads.transpose(1, 2).reshape(batch, tq, dim)
            return joined @ wo.T
        ''', source='transformer', priority='P0', prerequisites=('multihead', 'masked_attention'), minutes=35)

    put('rmsnorm_backward', 'RMSNorm · 手写反向', '归一化与前馈', '挑战',
        'rmsnorm_backward(x, weight, grad_output, eps=1e-6)',
        '从上游梯度推导 RMSNorm 对输入与缩放参数的梯度，特别注意共享 weight 的维度归约。',
        'r=(mean(x²)+ε)^(-1/2)，u=grad_output·weight\ndx=u·r−x·r³·mean(u·x)；dweight=Σ前导维度 grad_output·x·r',
        ['x、grad_output：相同 [...,D] shape，至少一维；weight：[D]；所有尺寸≥1，eps>0。',
         '对应前向 y=x*rsqrt(mean(x²,dim=-1)+eps)*weight；不减均值，不含 bias。',
         '返回 (grad_x, grad_weight)，shape 分别与 x、weight 相同；dtype/device 一致；禁止修改输入。',
         '手写梯度表达式，不调用 autograd/backward 代替推导；只检查数值，不检查二阶导。',
         'x 为一维时 weight 没有共享前导维度，grad_weight 不再求和。'],
        '例 1：x=0 时 dx=grad_output*weight/sqrt(eps)，dweight=0。\n例 2：x [2,3,4] 时，dweight [4] 必须沿 batch 和 token 两个维度求和。',
        ['先令 u=grad_output*weight，处理 x*r 的直接路径。', 'r 依赖全部 D 个输入；它的反向产生第二个减项。', 'dweight 沿所有前导维度相加，不能只沿 batch 维。'],
        ['遗漏 r 对 x 的梯度。', '误减均值，推导成 LayerNorm。', 'weight 梯度取平均而不是求和。'],
        ['为什么 D=1 时 dx 在 eps>0 下不一定为零？', '如何用随机上游梯度验证手推式？'],
        '''
        import torch

        def rmsnorm_backward(x, weight, grad_output, eps=1e-6):
            inv = torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
            upstream = grad_output * weight
            correction = (upstream * x).mean(-1, keepdim=True)
            grad_x = upstream * inv - x * inv.pow(3) * correction
            contribution = grad_output * x * inv
            axes = tuple(range(x.ndim - 1))
            grad_weight = contribution.sum(dim=axes) if axes else contribution
            return grad_x, grad_weight
        ''', source='rms', check_mode='numeric', prerequisites=('rmsnorm', 'relu_backward'), minutes=35)

    put('moe_topk_routing', 'MoE · 确定性 Top-K 路由', '稀疏专家', '进阶',
        'moe_topk_routing(router_logits, k)',
        '从 router logits 选择每个 token 的专家，并仅在入选专家之间重新归一化门控权重。先独立练习路由契约。',
        'indices = stable_descending_argsort(logits)[:k]；weights = softmax(selected_logits)',
        ['router_logits：[N,E] 有限浮点 Tensor，N/E≥1；k 为整数，1≤k≤E。',
         '返回 (indices, weights)，均为 [N,k]；indices 为同 device 的 int64，weights 同 logits dtype/device。',
         '先按 logit 降序，相同值按专家索引升序；每行 weights 和为 1，k=1 时权重恒为 1。',
         '没有噪声、capacity、丢 token 或额外系数；只检查数值，离散索引不评估自动求导。',
         '这是明确归一化的确定性教学路由，不覆盖所有 MoE 模型的 gate 约定。'],
        '例 1：logits=[[1,3,3,0]]，k=2 → indices=[[1,2]]，weights=[[0.5,0.5]]。\n例 2：logits=[[0,1]]，k=1 → indices=[[1]]，weights=[[1]]。',
        ['使用 stable=True 的降序 argsort，使同分时保持原专家次序。', 'gather 入选 logits 后沿 k 维 softmax，避免对极端分数直接 exp。'],
        ['使用不保证稳定 ties 的 topk，导致专家顺序不确定。', '全专家 softmax 后直接截取，未重新归一化。', '沿 token 维而非 expert 维归一化。'],
        ['路由索引的不可导性如何影响专家训练？', 'capacity factor 与负载均衡损失分别处理什么问题？'],
        '''
        import torch

        def moe_topk_routing(router_logits, k):
            indices = torch.argsort(router_logits, dim=-1, descending=True, stable=True)[:, :k]
            selected = router_logits.gather(1, indices)
            weights = torch.softmax(selected, dim=-1)
            return indices, weights
        ''', source='moe', priority='P0', check_mode='numeric', prerequisites=('softmax',))

    put('moe_layer', 'MoE · Top-K 分派与专家聚合', '稀疏专家', '挑战',
        'moe_layer(x, router_logits, w1, b1, w2, b2, k)',
        '实现一层确定性稀疏专家网络：每个 token 只选 k 个两层 ReLU 专家，用归一化 gate 聚合输出。',
        'expert_e(x)=W2_e·ReLU(W1_e·x+b1_e)+b2_e\ny_n=Σ入选e softmax(selected_logits_n)_e · expert_e(x_n)',
        ['x：[N,D]；router_logits：[N,E]；w1：[E,H,D]；b1：[E,H]；w2：[E,Dout,H]；b2：[E,Dout]。',
         '维度均≥1，1≤k≤E；返回 [N,Dout]；全部浮点输入检查梯度；不含 residual、dropout、容量裁剪。',
         '按 logits 降序选择 k 个专家，同分按专家索引升序；只在入选 logits 之间 softmax。',
         '离散选中集合视为固定分支，梯度沿 gate 和选中专家运算传播，不对索引本身求导；同分采用上述固定分支。',
         '本题的专家固定为 Linear-ReLU-Linear，并非所有模型都使用此结构；不要在函数内随机初始化。'],
        '例 1：两个被选专家对某 token 输出 [2] 和 [6]，gate=[0.25,0.75] → [5]。\n例 2：k=1 时 gate=1，输出等于唯一入选专家；该 token 的 router 梯度为零。',
        ['先得到 [N,k] 的 indices 与 weights。', '逐专家找出路由到它的 token 和 slot，只计算这些 token 的 MLP。', '把 weighted expert outputs 按 token 索引累加，重复 token 的 k 路贡献都要保留。'],
        ['漏掉入选 gate 的重新归一化。', '用赋值覆盖同一个 token 的前一位专家结果。', '专家输出先激活或做 softmax，改变了两层 MLP 定义。'],
        ['如何把逐专家循环改成设备上的 grouped GEMM？', '为什么 k=1 的归一化门控需要额外负载均衡信号？'],
        '''
        import torch

        def moe_layer(x, router_logits, w1, b1, w2, b2, k):
            indices = torch.argsort(router_logits, dim=-1, descending=True, stable=True)[:, :k]
            gates = torch.softmax(router_logits.gather(1, indices), dim=-1)
            output = x.new_zeros((x.shape[0], w2.shape[1]))
            for expert in range(w1.shape[0]):
                tokens, slots = torch.where(indices == expert)
                hidden = torch.relu(x[tokens] @ w1[expert].T + b1[expert])
                values = hidden @ w2[expert].T + b2[expert]
                weighted = values * gates[tokens, slots, None]
                output = output.index_add(0, tokens, weighted)
            return output
        ''', source='moe', prerequisites=('linear', 'softmax'), minutes=40)

    put('moe_load_balance', 'MoE · Switch 负载均衡辅助损失', '稀疏专家', '进阶',
        'moe_load_balance(router_logits, token_mask, coefficient=0.01)',
        '实现 Switch 的 top-1 辅助损失，区分不可导的硬分派频率与可导的平均路由概率。',
        'p=softmax(logits)；f_e=有效token分派到e的比例；P_e=有效token的p_e均值\nL_aux=coefficient·E·Σ_e f_e P_e',
        ['router_logits：[N,E] 有限浮点 Tensor，N/E≥1；token_mask：[N] bool，True 有效；coefficient≥0。',
         'top-1 使用 argmax，同分选最小专家索引；每个有效 token 只计入一个专家频率。',
         'f 不参与求导，P 必须保留计算图；仅对 router_logits 求导；输出同 dtype/device 的标量 Tensor。',
         '分母为有效 token 数；无有效 token 时返回与 logits 相连的零。',
         '采用 Switch 的 top-1 口径，与其他题的 top-k gate 归一化不同；不附加 z-loss，不把 f 替换成 P。'],
        '例 1：所有 token 的 logits 都相等时，P_e=1/E，因此输出 coefficient，即使 ties 都分派到专家 0。\n例 2：token_mask 全 False → loss=0，router 梯度全零且计算图保留。',
        ['softmax 沿专家维；argmax 给出每条 token 的硬专家。', '用 one-hot 或 scatter 统计有效 token 的专家频率。', '分别求 f 和 P 后做点积，再乘专家数 E 和 coefficient。'],
        ['用 P.square().sum() 替代 f·P，这变成另一种正则。', '把 padding token 计入分母。', '把概率 detach，辅助损失失去训练 router 的作用。'],
        ['为什么 f 不可导，整个辅助损失仍能训练 router？', '均匀概率与均匀实际负载是同一件事吗？'],
        '''
        import torch

        def moe_load_balance(router_logits, token_mask, coefficient=0.01):
            probabilities = torch.softmax(router_logits, dim=-1)
            valid = token_mask.to(router_logits.dtype)[:, None]
            count = valid.sum().clamp_min(1)
            winners = router_logits.argmax(-1, keepdim=True)
            dispatch = torch.zeros_like(router_logits).scatter(1, winners, 1) * valid
            mean_probabilities = (probabilities * valid).sum(0) / count
            fractions = dispatch.sum(0) / count
            return coefficient * router_logits.shape[1] * (fractions * mean_probabilities).sum()
        ''', source='switch', prerequisites=('softmax',), minutes=30)

    put('kv_cache_update', 'KV Cache · 固定容量片段更新', '注意力与缓存', '基础',
        'kv_cache_update(key_cache, value_cache, new_keys, new_values, start)',
        '实现固定容量 KV cache 的指定位置写入，保留未覆盖区域，并返回新缓存供下一步解码使用。',
        'updated[...,start:start+T,:] = new；其余位置保持原值',
        ['key_cache：[B,H,C,Dk]；value_cache：[B,H,C,Dv]；new_keys：[B,H,T,Dk]；new_values：[B,H,T,Dv]。',
         'B/H/C/Dk/Dv≥1，T≥0；start 为整数，0≤start≤C 且 start+T≤C；全部 Tensor 同 dtype/device。',
         '返回 (updated_keys, updated_values)，shape 与各自 cache 完全相同；数值检查，不评估梯度。',
         '禁止原地修改输入；不扩大容量、不执行 ring buffer、不返回被截短的有效前缀，也不改变片段外的旧值。',
         'T=0 时结果与旧 cache 数值相同；有效长度与 attention mask 由调用方维护。'],
        '例 1：旧 key=[10,20,30,40]，new=[7,8]，start=1 → [10,7,8,40]。\n例 2：容量 C=4，T=0，start=4 → 原 cache 数值保持不变。',
        ['确定 end=start+T，再拆出旧 cache 的前缀和后缀。', '沿序列维 -2 拼接前缀、新片段和后缀，K/V 分别处理。'],
        ['直接对输入 cache 做切片赋值。', '忽略 start 总写入开头。', '错误拼接在 head 维，或把缓存长度从 C 变成 C+T。'],
        ['静态 cache 与动态 cat cache 的显存分配方式有什么差别？', '如果支持每个 batch 不同写入位置，需要怎样修改接口？'],
        '''
        import torch

        def kv_cache_update(key_cache, value_cache, new_keys, new_values, start):
            end = start + new_keys.shape[-2]
            keys = torch.cat((key_cache[..., :start, :], new_keys, key_cache[..., end:, :]), dim=-2)
            values = torch.cat((value_cache[..., :start, :], new_values, value_cache[..., end:, :]), dim=-2)
            return keys, values
        ''', source='cache', priority='P0', check_mode='numeric', prerequisites=('kv_cache',), minutes=20)
