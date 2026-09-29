"""Reference answer regression + deliberately incorrect submission regression."""
import json
from pathlib import Path
from judge import BANK,evaluate
from cases import case_modules

def main():
    report=[]
    for p in BANK.values():
        r=evaluate(p['id'],p['solution'])
        print(f"{p['number']:02d} {p['id']:<20} {r['passed']}/{r['total']} {r['status']}")
        report.append({'id':p['id'],'passed':r['passed'],'total':r['total'],'status':r['status']})
        if r['status']!='accepted':
            for c in r['cases']:
                if not c['passed']: print(c['name'],c.get('stage'),c['message'])
            raise SystemExit(1)
    mutations={
        'constant_relu':('relu','def relu(x):\n    return torch.zeros_like(x)'),
        'detached_relu':('relu','def relu(x):\n    return torch.relu(x).detach()'),
        'zero_point_gradient':('relu','def relu(x):\n    return torch.where(x >= 0, x, torch.zeros_like(x))'),
        'unstable_softmax':('softmax','def softmax(x, dim=-1):\n    e=x.exp()\n    return e/e.sum(dim=dim,keepdim=True)'),
        'wrong_dtype':('relu','def relu(x):\n    return torch.relu(x).float()'),
        'all_masked_nan':('masked_attention','def masked_attention(q,k,v,mask):\n    s=(q@k.transpose(-2,-1))/q.shape[-1]**0.5\n    return torch.softmax(s.masked_fill(~mask,float("-inf")),-1)@v'),
        'wrong_gqa_repeat':('gqa',BANK['gqa']['solution'].replace('repeat_interleave(group, dim=1)','repeat(1, group, 1, 1)')),
        'cache_offset':('kv_cache',BANK['kv_cache']['solution'].replace('[:, None] + prefix','[:, None]')),
        'compile_error':('relu','def relu(:'),
        'inplace_detach_relu':('relu','def relu(x):\n    x.detach_()\n    return torch.relu(x)'),
        'deleted_decoder_weight':('decoder', BANK['decoder']['solution'].replace(
            '    return h + ffn', "    weights.pop('w_down')\n    return h + ffn")),
    }
    for module in case_modules():
        added=module.mutations(BANK)
        if mutations.keys() & added.keys(): raise AssertionError('错误反例名称重复')
        if {pid for pid,_ in added.values()} != module.IDS:
            raise AssertionError(f'{module.__name__} 并未覆盖每道新题的错误反例')
        mutations.update(added)
    for name,(pid,code) in mutations.items():
        r=evaluate(pid,code)
        assert r['status']!='accepted',f'Mutation was not detected: {name}'
        print('Caught:',name)
    import platform
    result={'python_version':platform.python_version(),'platform':platform.platform(),
            'torch_version':__import__('torch').__version__,'reference_problems':len(report),
            'reference_cases':sum(x['total'] for x in report),'mutations_caught':len(mutations),'results':report}
    Path(__file__).with_name('test_report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('All reference and negative tests passed.')

if __name__=='__main__': main()
