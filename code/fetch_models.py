"""把 TabPFN 权重下载到工作区内的 .cache/models/。

为什么不用 hf_hub_download：本机的 huggingface.co 不可达，需要走 hf-mirror.com；
而镜像下载在本机沙箱下会产生**空的快照符号链接**（真实内容在 blobs 里），
因此这里直接走 HTTP 拉取，避免符号链接问题。

用法：
  python code/fetch_models.py --versions v2 v2_5
"""

from __future__ import annotations

import argparse
import os
import ssl
import sys
import urllib.request
from http.client import HTTPException

import config

MIRROR = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
TIMEOUT = 900


def _http_get(url: str, dest: str) -> bool:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    tmp = dest + ".part"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "prl-fetch/1.0"})
        with urllib.request.urlopen(req, timeout=TIMEOUT, context=ctx) as resp, open(tmp, "wb") as fh:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
        size = os.path.getsize(tmp)
        if size < 1024:
            print(f"  下载内容异常（{size} 字节），丢弃")
            os.remove(tmp)
            return False
        os.replace(tmp, dest)
        return True
    except (HTTPException, OSError, TimeoutError) as exc:
        print(f"  HTTP 失败: {exc!r}")
        if os.path.exists(tmp):
            os.remove(tmp)
        return False


def fetch(version: str) -> str | None:
    repo, filename = config.MODEL_SOURCES[version]
    dest = os.path.join(config.MODELS_DIR, filename)
    if os.path.exists(dest) and os.path.getsize(dest) > 1024:
        print(f"[skip] {version}: {dest} ({os.path.getsize(dest) / 1e6:.1f} MB)")
        return dest

    print(f"[get ] {version} <- {repo}/{filename}")
    url = f"{MIRROR}/{repo}/resolve/main/{filename}"
    if _http_get(url, dest):
        print(f"[ok  ] {version}: {dest} ({os.path.getsize(dest) / 1e6:.1f} MB)")
        return dest
    print(f"[FAIL] {version}")
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--versions", nargs="*", default=["v2", "v2_5"])
    args = ap.parse_args()
    for v in args.versions:
        fetch(v)
    ready = config.available_versions()
    print("已就绪版本:", ready)
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
