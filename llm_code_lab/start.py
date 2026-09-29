#!/usr/bin/env python3
"""Serve the offline UI and evaluate trusted personal Python code on loopback only.
No package is downloaded at runtime. This is NOT a hostile-code sandbox.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import tempfile
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

ROOT=Path(__file__).resolve().parent
TOKEN=secrets.token_urlsafe(32)
SLOT=threading.BoundedSemaphore(1)
MAX_BYTES=240000
TIMEOUT_SECONDS=20


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--no-browser',action='store_true')
    args=parser.parse_args()
    if not 1<=args.port<=65535: parser.error('port must be 1..65535')
    origin=f'http://127.0.0.1:{args.port}'
    torch_version=None
    import_error=''
    try:
        import torch
        torch_version=torch.__version__
        has_torch=True
    except Exception as e:
        has_torch=False
        import_error=f'{type(e).__name__}: {e}'
    bank={p['id'] for p in json.loads((ROOT/'problems.json').read_text(encoding='utf-8'))}
    html=(ROOT/'index.html').read_text(encoding='utf-8').replace('__LOCAL_TOKEN__',TOKEN).encode('utf-8')

    class Handler(BaseHTTPRequestHandler):
        server_version='TensorDojo/0.1'
        def log_message(self,fmt,*args): pass
        def setup(self):
            super().setup()
            self.connection.settimeout(30)
        def respond(self,status,data,typ='application/json; charset=utf-8'):
            raw=json.dumps(data,ensure_ascii=False).encode('utf-8') if not isinstance(data,bytes) else data
            self.send_response(status)
            self.send_header('Content-Type',typ)
            self.send_header('Content-Length',str(len(raw)))
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('X-Frame-Options','DENY')
            self.send_header('Referrer-Policy','no-referrer')
            self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; object-src 'none'; base-uri 'none'")
            self.end_headers()
            try: self.wfile.write(raw)
            except (BrokenPipeError,ConnectionResetError): pass
        def host_ok(self):
            if self.headers.get('Host')!=f'127.0.0.1:{args.port}':
                self.respond(403,{'message':'Host 不匹配；请使用启动时显示的 127.0.0.1 地址。'})
                return False
            return True
        def do_GET(self):
            if not self.host_ok(): return
            path=urlsplit(self.path).path
            if path in ('/','/index.html'):
                self.respond(200,html,'text/html; charset=utf-8')
            elif path=='/api/health':
                self.respond(200,{'ready':has_torch,'python':sys.version.split()[0],
                                  'project_id':hashlib.sha256(str(ROOT.resolve()).encode()).hexdigest()[:16],
                                  'engine':'Local PyTorch CPU','torch_version':torch_version,'timeout_seconds':TIMEOUT_SECONDS,
                                  'message':'就绪' if has_torch else '无法导入 torch：'+import_error+'；请按 README 修复环境并重启。'})
            elif path=='/favicon.ico': self.respond(204,b'','image/x-icon')
            else: self.respond(404,{'message':'Not found'})
        def do_OPTIONS(self): self.respond(403,{'message':'跨域请求不被允许。'})
        def do_POST(self):
            if not self.host_ok(): return
            if urlsplit(self.path).path!='/api/run':
                self.respond(404,{'message':'Not found'}); return
            if self.headers.get('Origin')!=origin or self.headers.get('X-Local-Token')!=TOKEN:
                self.respond(403,{'message':'来源校验失败；请在本服务页面运行，刷新页面后重试。'}); return
            if self.headers.get('Content-Type','').split(';')[0]!='application/json':
                self.respond(415,{'message':'Expected application/json'}); return
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=MAX_BYTES: raise ValueError('请求体为空或过大')
                payload=json.loads(self.rfile.read(length))
                if not isinstance(payload,dict): raise ValueError('请求格式不正确')
                if payload.get('id') not in bank: raise ValueError('未知题号')
                if not isinstance(payload.get('code'),str) or len(payload['code'])>60000: raise ValueError('代码太长或类型不正确')
                if payload.get('mode') not in ('run','submit'): raise ValueError('未知运行模式')
            except (ValueError,TypeError) as e:
                self.respond(400,{'message':str(e)}); return
            if not has_torch:
                self.respond(503,{'message':'未安装 PyTorch。请使用 README 中的安装步骤，之后重启。'}); return
            if not SLOT.acquire(blocking=False):
                self.respond(429,{'message':'已有题目正在运行，请在其结束后重试。'}); return
            try:
                with tempfile.TemporaryDirectory(prefix='tensor-dojo-') as tmp:
                    target=Path(tmp)/'result.json'
                    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1')
                    process=subprocess.Popen([sys.executable,str(ROOT/'judge.py'),str(target)],
                        stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                        cwd=tmp,env=env,start_new_session=os.name!='nt')
                    try:
                        process.communicate(json.dumps(payload,ensure_ascii=False).encode('utf-8'),timeout=TIMEOUT_SECONDS)
                    except subprocess.TimeoutExpired:
                        if os.name!='nt':
                            try: os.killpg(process.pid,signal.SIGKILL)
                            except ProcessLookupError: pass
                        else: process.kill()
                        process.wait()
                        self.respond(200,{'status':'timeout','message':f'超过 {TIMEOUT_SECONDS} 秒，已停止本次判题进程。检查死循环或环境初始化耗时。',
                                          'cases':[],'passed':0,'total':0,'mode':payload['mode']}); return
                    if not target.exists():
                        self.respond(200,{'status':'error','message':'判题进程未能返回结果。检查代码是否退出进程、资源占用，或在终端执行 python selftest.py。',
                                          'cases':[],'passed':0,'total':0,'mode':payload['mode']}); return
                    if target.stat().st_size>2_000_000:
                        self.respond(500,{'message':'判题结果过大'}); return
                    self.respond(200,json.loads(target.read_text(encoding='utf-8')))
            except Exception as e:
                self.respond(500,{'message':f'运行服务错误：{type(e).__name__}: {e}'})
            finally: SLOT.release()

    try: server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    except OSError as e:
        raise SystemExit(f'无法启动端口 {args.port}：{e}\n可使用 python start.py --port 8766') from e
    print(f'\n  TensorDojo · 大模型算法训练场\n  {origin}\n')
    print('  仅运行你信任的个人代码。本工具不是恶意代码安全沙箱。')
    print('  无 CDN、无 API 调用；题目和代码不主动上传。Ctrl+C 退出。')
    if not has_torch: print('  注意：尚未安装 torch；当前只能浏览、编辑与保存。')
    if not args.no_browser: webbrowser.open(origin)
    try: server.serve_forever()
    except KeyboardInterrupt: print('\n已停止。')
    finally: server.server_close()

if __name__=='__main__': main()
