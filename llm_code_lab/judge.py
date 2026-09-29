"""Local trusted-code evaluator. A process boundary is NOT a security sandbox."""
import contextlib
import io
import json
import math
from pathlib import Path
import sys
import time
import traceback
import torch
from cases import make_cases, reference

ROOT=Path(__file__).resolve().parent
BANK={p['id']:p for p in json.loads((ROOT/'problems.json').read_text(encoding='utf-8'))}
torch.set_num_threads(1)

class CappedText(io.TextIOBase):
    def __init__(self, limit=12000): self.limit=limit; self.text=''; self.truncated=False
    def write(self, text):
        text=str(text); remaining=max(0,self.limit-len(self.text))
        self.text+=text[:remaining]
        if len(text)>remaining: self.truncated=True
        return len(text)
    def flush(self): pass
    def getvalue(self): return self.text+ ('\n[输出已截断]' if self.truncated else '')


def copy_tree(obj, grad=False, grad_paths=None, path='input'):
    if isinstance(obj,torch.Tensor):
        # Keep strides to actually test transposed/non-contiguous inputs.
        v=torch.empty_strided(obj.size(),obj.stride(),dtype=obj.dtype,device=obj.device)
        v.copy_(obj.detach())
        selected = grad_paths is None or any(
            path == p or path.startswith(p + '.') or path.startswith(p + '[')
            for p in grad_paths)
        if grad and selected and v.is_floating_point(): v.requires_grad_(True)
        return v
    if isinstance(obj,tuple): return tuple(copy_tree(v,grad,grad_paths,f'{path}[{i}]') for i,v in enumerate(obj))
    if isinstance(obj,list): return [copy_tree(v,grad,grad_paths,f'{path}[{i}]') for i,v in enumerate(obj)]
    if isinstance(obj,dict): return {k:copy_tree(v,grad,grad_paths,f'{path}.{k}') for k,v in obj.items()}
    return obj


def leaves(obj, path='input'):
    if isinstance(obj,torch.Tensor): return [(path,obj)]
    if isinstance(obj,(tuple,list)):
        return [v for i,x in enumerate(obj) for v in leaves(x,f'{path}[{i}]')]
    if isinstance(obj,dict):
        return [v for k,x in obj.items() for v in leaves(x,f'{path}.{k}')]
    return []


def assert_input_unchanged(actual, original, flags, path='input'):
    """Check the complete input tree, including grad flags captured before execution."""
    if isinstance(original, torch.Tensor):
        if not isinstance(actual, torch.Tensor):
            raise AssertionError(f'{path} 的 Tensor 被替换')
        identity, requires_grad = flags[path]
        if id(actual) != identity:
            raise AssertionError(f'{path} 的输入 Tensor 被替换')
        if (actual.shape != original.shape or actual.dtype != original.dtype
                or actual.device != original.device or actual.stride() != original.stride()):
            raise AssertionError(f'{path} 的 shape / dtype / device / stride 被修改')
        if actual.requires_grad != requires_grad:
            raise AssertionError(f'{path} 的 requires_grad 被修改；不能原地 detach 或关闭梯度')
        if not torch.equal(actual.detach(), original):
            raise AssertionError(f'{path} 被原地修改；请创建新输出。')
        return
    if type(actual) is not type(original):
        raise AssertionError(f'{path} 的输入类型被修改')
    if isinstance(original, dict):
        if list(actual) != list(original):
            raise AssertionError(f'{path} 的字典键或顺序被修改')
        for key in original:
            assert_input_unchanged(actual[key], original[key], flags, f'{path}.{key}')
    elif isinstance(original, (tuple, list)):
        if len(actual) != len(original):
            raise AssertionError(f'{path} 的输入长度被修改')
        for i, (a, b) in enumerate(zip(actual, original)):
            assert_input_unchanged(a, b, flags, f'{path}[{i}]')
    elif actual != original:
        raise AssertionError(f'{path} 的输入值被修改')


def describe(obj):
    if isinstance(obj,torch.Tensor):
        x=obj.detach().cpu()
        values=x.reshape(-1)[:8].tolist()
        values=[v if not isinstance(v,float) or math.isfinite(v) else str(v) for v in values]
        return {'shape':list(x.shape),'dtype':str(x.dtype),'device':str(obj.device),
                'values':values,'numel':x.numel(),'contiguous':obj.is_contiguous()}
    if isinstance(obj,(tuple,list)): return [describe(x) for x in obj]
    if isinstance(obj,dict): return {k:describe(v) for k,v in obj.items()}
    if isinstance(obj,float) and not math.isfinite(obj): return str(obj)
    if obj is None or isinstance(obj,(str,bool,int,float)): return obj
    return repr(obj)[:500]


def tol(dtype, grad=False):
    r,a = (2e-4,2e-5) if dtype==torch.float32 else (1e-7,1e-8)
    return (r*5,a*5) if grad else (r,a)


def assert_contract(actual,expected):
    if isinstance(expected,torch.Tensor):
        if not isinstance(actual,torch.Tensor): raise AssertionError(f'必须返回 Tensor，实际为 {type(actual).__name__}')
        if actual.shape!=expected.shape: raise AssertionError(f'shape 不符：得到 {list(actual.shape)}，期望 {list(expected.shape)}')
        if actual.dtype!=expected.dtype: raise AssertionError(f'dtype 不符：得到 {actual.dtype}，期望 {expected.dtype}')
        if actual.device!=expected.device: raise AssertionError(f'device 不符：得到 {actual.device}，期望 {expected.device}')
        return
    if isinstance(expected,tuple):
        if not isinstance(actual,tuple) or len(actual)!=len(expected): raise AssertionError('必须返回题目指定的 tuple，且元素个数一致')
        for x,y in zip(actual,expected): assert_contract(x,y)


def ensure_finite(obj):
    for path,x in leaves(obj,'output'):
        if not torch.isfinite(x).all().item(): raise AssertionError(f'{path} 出现 NaN 或 Inf；检查指数溢出、归一化分母和 mask。')


def assert_values(actual,expected,grad=False):
    if isinstance(expected,torch.Tensor):
        r,a=tol(expected.dtype,grad)
        torch.testing.assert_close(actual,expected,rtol=r,atol=a,equal_nan=False,check_dtype=True,check_device=True)
    else:
        for x,y in zip(actual,expected): assert_values(x,y,grad)


def check_grad(actual, expected, inputs_a, inputs_b, seed, require_connected_paths=()):
    outputs_a=[x for _,x in leaves(actual,'output')]
    outputs_b=[x for _,x in leaves(expected,'output')]
    if not inputs_a:
        raise AssertionError('判题算例未指定可求导输入')
    if any(y.requires_grad and not x.requires_grad for x,y in zip(outputs_a,outputs_b)):
        raise AssertionError('输出没有完整计算图。请检查 detach()、.item()、NumPy 转换或 torch.tensor(existing_tensor)。')
    gen=torch.Generator().manual_seed(seed)
    for direction in range(2):
        vectors=[torch.randn(x.shape,generator=gen,dtype=x.dtype,device=x.device) for x in outputs_b]
        def vjp(outputs, inputs):
            active=[(x,v) for x,v in zip(outputs,vectors) if x.requires_grad]
            if not active: return (None,) * len(inputs)
            return torch.autograd.grad([x for x,_ in active], inputs,
                grad_outputs=[v for _,v in active], allow_unused=True, retain_graph=direction==0)
        ga=vjp(outputs_a,[x for _,x in inputs_a])
        gb=vjp(outputs_b,inputs_b)
        for (path,x),u,v in zip(inputs_a,ga,gb):
            must_connect = any(path == p or path.startswith(p + '.') or path.startswith(p + '[')
                               for p in require_connected_paths)
            if must_connect and u is None and v is not None:
                raise AssertionError(f'{path} 与输出的计算图断开；本算例要求保留零梯度连接。')
            u=torch.zeros_like(x) if u is None else u
            v=torch.zeros_like(x) if v is None else v
            if not torch.isfinite(u).all(): raise AssertionError(f'{path} 的梯度出现 NaN/Inf；前向稳定不等于反向稳定。')
            try: assert_values(u,v,grad=True)
            except AssertionError as e: raise AssertionError(f'{path} 梯度不一致（随机 VJP 方向 {direction+1}）：\n{e}') from e


def evaluate(pid,code,mode='submit',seed=1729):
    if pid not in BANK: raise ValueError('未知题号')
    if not isinstance(code,str) or len(code)>60000: raise ValueError('代码须为字符串且不超过 60000 字符')
    if mode not in ('run','submit'): raise ValueError('mode 必须为 run 或 submit')
    tests=make_cases(pid,seed)
    if not tests: raise ValueError('题目没有配置测试用例')
    if mode=='run': tests=tests[:2]
    log=CappedText()
    namespace={'__name__':'__submission__','torch':torch}
    try:
        with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            exec(compile(code,'submission.py','exec'),namespace)
        fn=namespace.get(pid)
        if not callable(fn): raise ValueError(f'找不到函数 {pid}；请保留题目要求的函数名。')
    except BaseException as e:
        return {'status':'error','message':f'{type(e).__name__}: {e}','traceback':traceback.format_exc(limit=5)[-5000:],
                'passed':0,'total':len(tests),'cases':[],'stdout':log.getvalue(),'mode':mode,'seed':seed}
    results=[]
    for idx,c in enumerate(tests):
        stage='执行'; actual=expected=None
        item={'name':c['name'],'passed':False,'checks':{},'input':describe(c['args'])}
        started=time.perf_counter()
        try:
            a,b=(copy_tree(c['args'],c['grad'],c.get('grad_paths')) for _ in range(2))
            flags={p:(id(x),x.requires_grad) for p,x in leaves(a)}
            inputs_a=[(p,x) for p,x in leaves(a) if x.requires_grad]
            inputs_b=[x for _,x in leaves(b) if x.requires_grad]
            with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                begin=time.perf_counter()
                actual=fn(*a)
                item['forward_ms']=round((time.perf_counter()-begin)*1000,3)
            stage='输入不变性'
            assert_input_unchanged(a,c['args'],flags)
            item['checks']['input']=True
            expected=reference(pid,*b)
            stage='形状 / 类型'
            assert_contract(actual,expected); item['checks']['contract']=True
            stage='数值稳定性'
            ensure_finite(actual); item['checks']['finite']=True
            stage='前向数值'
            assert_values(actual,expected); item['checks']['forward']=True
            stage='反向梯度'
            if c['grad']:
                check_grad(actual,expected,inputs_a,inputs_b,seed+idx+500,
                           c.get('require_connected_paths', ()))
                item['checks']['gradient']=True
            else: item['checks']['gradient']=None
            item['passed']=True
            item['message']='前向 / 计算图 / 随机 VJP 梯度一致' if c['grad'] else '数值 / 返回契约 / 输入不变性一致（本题不检查自动求导）'
        except BaseException as e:
            item['stage']=stage
            item['message']=f'{type(e).__name__}: {e}'[:5000]
            item['traceback']=traceback.format_exc(limit=4)[-3500:]
            if actual is not None:
                try: item['actual']=describe(actual)
                except BaseException: pass
            if expected is not None: item['expected']=describe(expected)
        item['elapsed_ms']=round((time.perf_counter()-started)*1000,2)
        results.append(item)
    passed=sum(x['passed'] for x in results)
    return {'status':'accepted' if passed==len(results) else 'failed','passed':passed,'total':len(results),
            'cases':results,'stdout':log.getvalue(),'mode':mode,'seed':seed,'torch_version':torch.__version__,
            'message':'全部测试通过' if passed==len(results) else '存在未通过的测试',
            'check_mode':BANK[pid].get('check_mode','autograd'),
            'notes':'CPU float32/float64；可微题检查 2 个随机 VJP 方向，其余题检查数值与输入契约；耗时不作为性能排名。'}

if __name__=='__main__':
    # stdout/stderr of the OS process may be discarded by start.py. JSON uses an explicit file.
    target=Path(sys.argv[1])
    try:
        payload=json.loads(sys.stdin.read(250000))
        result=evaluate(payload['id'],payload['code'],payload.get('mode','submit'))
    except BaseException as e:
        result={'status':'error','message':f'{type(e).__name__}: {e}','cases':[],'passed':0,'total':0}
    target.write_text(json.dumps(result,ensure_ascii=False,allow_nan=False),encoding='utf-8')
