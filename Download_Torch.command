#!/bin/bash
cd "$(dirname "$0")" || exit 1
bash ./Start.command --download-only
CODE=$?
if [ "$CODE" -eq 0 ]; then read -r -p '下载结束。按 Enter 关闭...'; fi
exit "$CODE"
