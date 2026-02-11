# extract_nav_matrix_v2.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright

from extract_toc_universal_v5 import (
    _safe_name_from_url,
    _render_md,
    _dedupe_preserve_order,
    _flatten_urls,
    _filter_tree,
    _extract_sidebar_tree,
    _extract_links_flat,
    _expand_all_smart,
    _scroll_container,
)


def _first_level_only(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for n in nodes:
        out.append({"title": n.get("title"), "url": n.get("url"), "children": []})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--name", default="")
    ap.add_argument("--project-root", default="")

    ap.add_argument("--scope-origin", default="")
    ap.add_argument("--scope-prefix", default="")

    ap.add_argument("--sidebar-selector", default="", help="可选：强制 sidebar 容器")
    ap.add_argument("--prefer-visible", action="store_true")

    ap.add_argument("--headful", action="store_true")
    ap.add_argument("--timeout-ms", type=int, default=60000)
    ap.add_argument("--user-data-dir", default="")
    ap.add_argument("--channel", default="")

    ap.add_argument("--expand-rounds", type=int, default=35)
    ap.add_argument("--scroll-steps", type=int, default=14)
    ap.add_argument("--pause", type=float, default=0.2)

    args = ap.parse_args()

    project_root = Path(args.project_root) if args.project_root else Path(__file__).resolve().parents[1]
    runtime_dir = project_root / "runtime"
    output_dir = project_root / "output"

    if not args.user_data_dir:
        user_data_dir = runtime_dir / "user_data" / ("profile_" + _safe_name_from_url(args.url)[:30])
    else:
        user_data_dir = Path(args.user_data_dir)

    name = args.name.strip() or _safe_name_from_url(args.url)
    out_dir = output_dir / "toc" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix: Dict[str, Any] = {"site": args.url, "parts": []}

    with sync_playwright() as p:
        browser_type = p.chromium
        launch_kwargs = {"headless": (not args.headful)}
        if args.channel:
            launch_kwargs["channel"] = args.channel

        context = browser_type.launch_persistent_context(str(user_data_dir), **launch_kwargs)
        page = context.new_page()
        page.goto(args.url, wait_until="domcontentloaded", timeout=args.timeout_ms)

        current_url = page.url
        scope_origin = args.scope_origin.strip() or f"{urlparse(current_url).scheme}://{urlparse(current_url).netloc}"
        scope_prefix = args.scope_prefix.strip()

        # 1) parts：只取 sidebar 的一级 section 链接（不展开）
        if args.sidebar_selector:
            root = page.locator("css=" + args.sidebar_selector).first
        else:
            root = page.locator("body").first

        parts = _extract_sidebar_tree(root, prefer_visible=args.prefer_visible) or _extract_links_flat(root, prefer_visible=args.prefer_visible)
        parts = _filter_tree(parts, current_url, scope_origin, scope_prefix, keep_fragments=False)
        parts = _first_level_only(parts)

        # 2) 对每个 part：打开 -> 展开 -> 抽 sidebar 树
        for part in parts:
            part_title = part.get("title") or part.get("url")
            part_url = part.get("url")
            if not part_url:
                continue

            page.goto(part_url, wait_until="domcontentloaded", timeout=args.timeout_ms)

            if args.sidebar_selector:
                pr = page.locator("css=" + args.sidebar_selector).first
            else:
                pr = page.locator("body").first

            _expand_all_smart(pr, rounds=args.expand_rounds, pause=args.pause)
            _scroll_container(pr, steps=args.scroll_steps, pause=args.pause)
            _expand_all_smart(pr, rounds=args.expand_rounds, pause=args.pause)

            toc_tree = _extract_sidebar_tree(pr, prefer_visible=args.prefer_visible) or []
            toc_tree = _filter_tree(toc_tree, page.url, scope_origin, scope_prefix, keep_fragments=False)

            matrix["parts"].append({"title": part_title, "url": part_url, "toc": toc_tree})

        context.close()

    (out_dir / "matrix.json").write_text(json.dumps(matrix, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines: List[str] = []
    all_urls: List[str] = []
    for part in matrix["parts"]:
        md_lines.append(f"# {part['title']}")
        if part.get("url"):
            md_lines.append(f"- 入口：{part['url']}")
        toc = part.get("toc") or []
        if toc:
            md_lines.extend(_render_md(toc))
            all_urls.extend(_flatten_urls(toc))
        md_lines.append("")

    (out_dir / "matrix.md").write_text("\n".join(md_lines).strip() + "\n", encoding="utf-8")
    all_urls = _dedupe_preserve_order([u for u in all_urls if u and u.strip()])
    (out_dir / "all_urls.txt").write_text("\n".join(all_urls) + "\n", encoding="utf-8")

    print(f"OK: {out_dir}")


if __name__ == "__main__":
    main()
