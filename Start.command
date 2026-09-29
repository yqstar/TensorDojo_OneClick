#!/bin/bash
# macOS: double-click in Finder. Linux: bash Start.command.
cd "$(dirname "$0")" || exit 1
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
export PATH="/opt/homebrew/bin:/usr/local/bin:/Library/Frameworks/Python.framework/Versions/3.11/bin:$PATH"
PY=""
if [ -n "${TENSORDOJO_PYTHON:-}" ]; then
  PY="$TENSORDOJO_PYTHON"
elif [ -x ".tensor-dojo-venv/bin/python" ]; then
  PY="$PWD/.tensor-dojo-venv/bin/python"
else
  for candidate in python3.11 python3.12 python3.13 python3.10 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys,struct; assert sys.version_info >= (3,10) and struct.calcsize("P")==8' >/dev/null 2>&1; then
      PY="$(command -v "$candidate")"; break
    fi
  done
fi
if [ -z "$PY" ]; then
  echo '未找到 64 位 Python 3.10+。请先安装 Python 3.11；本包没有内置 Python。'
  echo '官方安装来源：https://www.python.org/downloads/'
  read -r -p '按 Enter 关闭...'; exit 1
fi
"$PY" -u launch.py "$@"
CODE=$?
if [ "$CODE" -ne 0 ]; then read -r -p '启动失败。按 Enter 关闭...'; fi
exit "$CODE"
