#!/usr/bin/env python3
"""TensorDojo local launcher. Needs 64-bit Python; never installs system packages.
A browser HTML file cannot launch this script on its own. Use Start.command/bat.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import struct
import subprocess
import sys
import urllib.error
import urllib.request
import webbrowser

ROOT = Path(__file__).resolve().parent
APP = ROOT / 'llm_code_lab'
VENV = ROOT / '.tensor-dojo-venv'
PYPI = 'https://pypi.org/simple'
CPU_INDEX = 'https://download.pytorch.org/whl/cpu'
PROBE = """
import json, sys, platform, torch
x = torch.tensor([-1., 1.], requires_grad=True)
torch.relu(x).sum().backward()
assert x.grad.tolist() == [0., 1.]
assert hasattr(torch.nn.functional, 'scaled_dot_product_attention')
print(json.dumps({'python': platform.python_version(), 'torch': torch.__version__,
                  'executable': sys.executable}))
"""


def environment():
    return dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1',
                PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                PIP_DISABLE_PIP_VERSION_CHECK='1')


def run(command, **kwargs):
    return subprocess.run([str(s) for s in command], env=environment(), **kwargs)


def probe(python):
    try:
        p = run([python, '-c', PROBE], capture_output=True, text=True,
                encoding='utf-8', errors='replace', timeout=45)
        if p.returncode == 0:
            return json.loads(p.stdout.strip().splitlines()[-1])
    except (OSError, ValueError, subprocess.TimeoutExpired, IndexError):
        pass
    return None


def install_plan(system=None, machine=None, version=None):
    """Conservative, pinned fallback; a working existing torch is reused first."""
    system = system or platform.system()
    machine = (machine or platform.machine()).lower()
    version = tuple(version or sys.version_info[:2])
    if system == 'Darwin' and machine in ('x86_64', 'amd64'):
        if not (3, 10) <= version <= (3, 12):
            raise RuntimeError('Intel Mac 的兼容安装分支需要 Python 3.10–3.12，建议 3.11。'
                               '不会用 Python 3.13+ 强装旧版 torch。')
        return ['torch==2.2.2', 'numpy==1.26.4'], PYPI
    if not (3, 10) <= version <= (3, 13):
        raise RuntimeError('自动安装分支支持 Python 3.10–3.13；建议使用 64 位 Python 3.11。'
                           '已有可用 torch 环境不受此安装分支限制。')
    if system == 'Darwin' and machine == 'arm64':
        return ['torch==2.10.0'], PYPI
    if system in ('Windows', 'Linux') and machine in ('amd64', 'x86_64'):
        return ['torch==2.10.0'], CPU_INDEX
    raise RuntimeError('当前系统/架构没有预设的自动安装分支。请先准备可用 torch 环境，再用该环境启动。')


def fingerprint():
    return f'{platform.system()}-{platform.machine().lower()}-cp{sys.version_info.major}{sys.version_info.minor}'


def wheel_directory():
    return ROOT / 'torch_wheels' / fingerprint()


def pip_command(python, verb):
    return [str(python), '-m', 'pip', '--isolated', '--disable-pip-version-check', verb]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def download_wheels():
    packages, index = install_plan()
    dest = wheel_directory()
    dest.mkdir(parents=True, exist_ok=True)
    print('下载当前平台对应的 wheel 及依赖：', ', '.join(packages), flush=True)
    run(pip_command(sys.executable, 'download') + ['--only-binary=:all:', '--index-url', index,
        '--dest', str(dest)] + packages, check=True)
    # Hashes detect damaged files; this local manifest is not a signed trust mechanism.
    manifest = {'platform': fingerprint(), 'packages': packages, 'index': index,
        'wheels': {f.name: sha256_file(f) for f in dest.glob('*.whl')}}
    (dest / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'已准备 wheel：{dest}\n尚未安装 Python，也没有创建独立桌面应用。', flush=True)


def offline_source(packages):
    dest = wheel_directory()
    manifest_path = dest / 'manifest.json'
    if not manifest_path.exists():
        raise RuntimeError('没有本机匹配的离线 wheel 缓存。联网时先运行 Download_Torch 启动器。')
    m = json.loads(manifest_path.read_text(encoding='utf-8'))
    if m.get('platform') != fingerprint() or m.get('packages') != packages or not m.get('wheels'):
        raise RuntimeError('离线缓存的平台或安装方案不匹配，请重新下载。')
    for name, digest in m['wheels'].items():
        if Path(name).name != name or not name.endswith('.whl'):
            raise RuntimeError('离线缓存文件名无效。')
        f = dest / name
        if not f.is_file() or sha256_file(f) != digest:
            raise RuntimeError(f'离线缓存缺失或损坏：{name}')
    return ['--no-index', '--find-links', str(dest)]


def prepare_python(offline=False):
    vpython = VENV / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    candidates = [vpython, Path(sys.executable)] if vpython.exists() else [Path(sys.executable)]
    for candidate in candidates:
        info = probe(candidate)
        if info:
            print(f"复用环境：{info['executable']}\nPython {info['python']} / torch {info['torch']}", flush=True)
            return candidate, info
    packages, index = install_plan()
    use_cache = offline or (wheel_directory() / 'manifest.json').exists()
    source = offline_source(packages) if use_cache else ['--index-url', index]
    if VENV.exists() and not vpython.exists():
        raise RuntimeError(f'{VENV} 环境不完整，请移走这个环境目录后重试。不会自动删除已有文件。')
    if not VENV.exists():
        print('创建项目专用环境；不会修改系统 Python。', flush=True)
        run([sys.executable, '-m', 'venv', str(VENV)], check=True)
    # Reject a venv created using a different interpreter version/platform.
    p = run([vpython, '-c', 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'],
            capture_output=True, text=True, timeout=15, check=True)
    if p.stdout.strip() != f'{sys.version_info.major}.{sys.version_info.minor}':
        raise RuntimeError('已有项目环境的 Python 版本不同且 torch 不可用。请用原版本 Python 启动，或移走该环境后重建。')
    print('首次准备 torch 和必要依赖：' + ', '.join(packages), flush=True)
    print('此步骤需要联网。' if not use_cache else '使用本机离线缓存，不访问软件包索引。', flush=True)
    run(pip_command(vpython, 'install') + ['--only-binary=:all:'] + source + packages, check=True)
    info = probe(vpython)
    if not info:
        raise RuntimeError('安装后 torch 仍无法正常计算，请检查上方安装错误、系统版本和运行库。')
    return vpython, info


def validate_once(python, info):
    sig = hashlib.sha256(json.dumps(info, sort_keys=True).encode())
    # Include modular question banks/oracles so any evaluator change reruns validation.
    for path in sorted(APP.glob('*.py')) + [APP / 'problems.json']:
        sig.update(path.name.encode() + b'\0')
        sig.update(path.read_bytes())
    expected = sig.hexdigest()
    stamp = ROOT / '.validation.json'
    try:
        if json.loads(stamp.read_text(encoding='utf-8')).get('signature') == expected:
            return
    except (OSError, ValueError):
        pass
    print('首次使用此环境：运行题库自检，成功后才开启判题。', flush=True)
    run([python, str(APP / 'selftest.py')], cwd=APP, check=True, timeout=120)
    stamp.write_text(json.dumps({'signature': expected, 'runtime': info}, indent=2), encoding='utf-8')


def running_here(port):
    url = f'http://127.0.0.1:{port}'
    project = hashlib.sha256(str(APP.resolve()).encode()).hexdigest()[:16]
    # Never send loopback health requests through a configured network proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url + '/api/health', timeout=2) as r:
            body = json.loads(r.read(16384))
        if body.get('project_id') == project and body.get('ready'):
            return True
    except (OSError, ValueError, urllib.error.URLError):
        pass
    # Do not silently change port, as browser localStorage is scoped to the origin.
    with socket.socket() as s:
        s.settimeout(1)
        if s.connect_ex(('127.0.0.1', port)) == 0:
            raise RuntimeError(f'端口 {port} 已被其他服务或旧版占用。先关闭旧服务，或用 --port 8766 启动；'
                               '换端口后原进度需手动导出/导入。')
    return False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--offline', action='store_true', help='缺环境时只从预下载 wheel 安装')
    parser.add_argument('--download-only', action='store_true', help='仅下载本机匹配的 torch wheel 与依赖')
    parser.add_argument('--check', action='store_true', help='准备并自检环境，不启动服务')
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error('port must be 1..65535')
    if sys.version_info < (3, 10) or struct.calcsize('P') != 8:
        raise RuntimeError('请使用 64 位 Python 3.10 或更高版本，推荐 Python 3.11。')
    print('\nTensorDojo · 本地双击启动版\n仅运行可信个人代码；不是恶意代码沙箱。', flush=True)
    if args.download_only:
        download_wheels()
        return 0
    if not args.check and running_here(args.port):
        print('本项目已在运行，直接打开已有页面。', flush=True)
        if not args.no_browser:
            webbrowser.open(f'http://127.0.0.1:{args.port}')
        return 0
    python, info = prepare_python(args.offline)
    validate_once(python, info)
    if args.check:
        print('环境自检通过。', flush=True)
        return 0
    cmd = [python, str(APP / 'start.py'), '--port', str(args.port)]
    if args.no_browser:
        cmd.append('--no-browser')
    print('服务即将打开界面。练习期间保留终端；Ctrl+C 停止。', flush=True)
    try:
        return run(cmd, cwd=APP).returncode
    except KeyboardInterrupt:
        return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\n已取消。')
        raise SystemExit(130)
    except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f'\n启动失败：{exc}\n请查看 README_双击启动.md。', file=sys.stderr)
        raise SystemExit(1)
