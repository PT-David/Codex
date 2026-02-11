# batch_convert_from_urls_v2.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", required=True, help="extract_sidebar_toc_v2.py 输出的 urls.txt")
    ap.add_argument("--outdir", required=True, help="Docling 输出目录（md/html/json...）")

    # Docling 环境与脚本
    ap.add_argument("--docling-python", required=True, help=r"Docling venv 的 python.exe，例如 D:\Pro:contentReference[oaicite:13]{index=13}e")
    ap.add_argument("--converter", required=True, help=r"convert_docling_url_auto.py 的完整路径")

    ap.add_argument("--to", default="md", help="输出格式（默认 md，对齐 auto 脚本 --to）")

    # 透传给 convert_docling_url_auto.py 的额外参数：用 --converter-args -- ... 的形式
    ap.add_argument("--converter-args", nargs=argparse.REMAINDER, default=[],
                    help="透传参数：写法如 --converter-args -- --ocr --ocr-engine rapidocr --image-export-mode referenced")

    ap.add_argument("--start", type=int, default=1, help="从第几个 URL 开始（断点续跑用，1-based）")
    ap.add_argument("--limit", type=int, default=0, help="最多跑多少个（0=不限制）")

    args = ap.parse_args()

    urls = [x.strip() for x in Path(args.urls).read_text(encoding="utf-8").splitlines() if x.strip()]

    start_idx = max(args.start, 1) - 1
    urls = urls[start_idx:]

    if args.limit and args.limit > 0:
        urls = urls[:args.limit]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 如果用户写了 "--converter-args -- ..."，argparse 会把 "--" 也放进列表里，这里去掉
    passthrough = args.converter_args[:]
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    total = len(urls)
    for i, url in enumerate(urls, 1):
        cmd = [
            args.docling_python,
            args.converter,
            "--outdir", str(outdir),
            "--to", args.to,
        ] + passthrough + [url]

        print(f"[{i}/{total}] {url}")
        # 不 check=True：遇到单页失败继续跑，便于全站批处理
        subprocess.run(cmd, check=False)

    print("DONE.")


if __name__ == "__main__":
    main()
