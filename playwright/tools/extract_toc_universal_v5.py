# extract_toc_universal_v5.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

from playwright.sync_api import sync_playwright

Node = Dict[str, Any]


def _safe_name_from_url(url: str) -> str:
    u = urlparse(url)
    host = re.sub(r"[^0-9A-Za-z._-]+", "_", u.netloc or "site")
    path = re.sub(r"[^0-9A-Za-z._-]+", "_", (u.path.strip("/") or "root"))
    return (host + "_" + path)[:80]


def _normalize_abs(href: str, base: str) -> str:
    try:
        return urljoin(base, href)
    except Exception:
        return href


def _is_fragment_only(url: str, current_url: str) -> bool:
    try:
        u = urlparse(url)
        cur = urlparse(current_url)
        if not u.fragment:
            return False
        return (u.netloc == cur.netloc) and (u.path == cur.path) and (u.query == cur.query)
    except Exception:
        return False


def _render_md(nodes: List[Node], depth: int = 0) -> List[str]:
    lines: List[str] = []
    indent = "  " * depth
    for n in nodes:
        title = (n.get("title") or "").strip() or "(untitled)"
        url = n.get("url")
        if url:
            lines.append(f"{indent}- [{title}]({url})")
        else:
            lines.append(f"{indent}- {title}")
        ch = n.get("children") or []
        if ch:
            lines.extend(_render_md(ch, depth + 1))
    return lines


def _flatten_urls(nodes: List[Node], out: Optional[List[str]] = None) -> List[str]:
    if out is None:
        out = []
    for n in nodes:
        url = (n.get("url") or "").strip()
        if url:
            out.append(url)
        for c in n.get("children") or []:
            _flatten_urls([c], out)
    return out


def _dedupe_preserve_order(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if not u or u in seen:
            continue
        seen.add(u)
        out.append(u)
    return out


def _filter_tree(tree: List[Node], current_url: str, scope_origin: str, scope_prefix: str, keep_fragments: bool) -> List[Node]:
    origin_netloc = urlparse(scope_origin or current_url).netloc
    prefix = (scope_prefix or "").rstrip("/")

    def keep_url(u: str) -> bool:
        if not u:
            return False
        uu = urlparse(u)
        if uu.scheme not in ("http", "https"):
            return False
        if origin_netloc and uu.netloc != origin_netloc:
            return False
        if prefix and not u.startswith(prefix):
            return False
        if (not keep_fragments) and _is_fragment_only(u, current_url):
            return False
        return True

    def walk(nodes: List[Node]) -> List[Node]:
        out: List[Node] = []
        for n in nodes:
            url = (n.get("url") or "").strip() or None
            children = n.get("children") or []
            children = walk(children) if children else []
            if url and keep_url(url):
                out.append({"title": n.get("title"), "url": url, "children": children})
            elif (not url) and children:
                out.append({"title": n.get("title"), "url": None, "children": children})
        return out

    return walk(tree)


def _css_escape(s: str) -> str:
    # 简易 CSS escape（足够处理 __nav:1 之类）
    return re.sub(r"([^0-9A-Za-z_-])", lambda m: "\\" + m.group(1), s)


def _expand_all_smart(root, rounds: int, pause: float):
    """
    通用 + 兼容 MkDocs Material：
    - button[aria-expanded=false]
    - details/summary
    - label[for] -> checkbox/radio 未选中（典型：Material for MkDocs）
    """
    for _ in range(rounds):
        clicked = 0

        # 1) aria-expanded toggles
        toggles = root.locator("button[aria-expanded='false'], [role='button'][aria-expanded='false']")
        try:
            n = toggles.count()
        except Exception:
            n = 0
        for i in range(n):
            t = toggles.nth(i)
            try:
                t.click(timeout=800)
                clicked += 1
                time.sleep(pause)
            except Exception:
                continue

        # 2) details/summary
        summaries = root.locator("summary")
        try:
            n2 = summaries.count()
        except Exception:
            n2 = 0
        for i in range(n2):
            s = summaries.nth(i)
            try:
                s.click(timeout=800)
                clicked += 1
                time.sleep(pause)
            except Exception:
                continue

        # 3) label[for] -> associated checkbox/radio not checked
        labels = root.locator("label[for]")
        try:
            n3 = labels.count()
        except Exception:
            n3 = 0
        for i in range(n3):
            lab = labels.nth(i)
            try:
                fid = lab.get_attribute("for") or ""
                fid = fid.strip()
                if not fid:
                    continue
                ctrl = root.locator(f"#{_css_escape(fid)}").first
                if ctrl.count() == 0:
                    continue
                typ = (ctrl.get_attribute("type") or "").lower()
                if typ not in ("checkbox", "radio"):
                    continue
                try:
                    if ctrl.is_checked():
                        continue
                except Exception:
                    # is_checked 失败就跳过，避免误点表单
                    continue

                lab.click(timeout=800)
                clicked += 1
                time.sleep(pause)
            except Exception:
                continue

        if clicked == 0:
            break


def _scroll_container(root, steps: int, pause: float):
    for _ in range(steps):
        try:
            root.evaluate("(el) => { el.scrollTop = el.scrollHeight; }")
        except Exception:
            pass
        time.sleep(pause)


def _extract_sidebar_tree(root, prefer_visible: bool) -> Optional[List[Node]]:
    """
    v5 核心修复：
    - 选 ul 时“可见链接”优先
    - 解析 li 时：
      * 只认 :scope > a[href] 作为本节点链接
      * 若没有直接链接，则作为 group 节点：clone li，删掉 ul/nav 后取 text => 得到 Getting started
    - 额外：去掉“父节点=子节点首项重复”的噪声
    """
    js = r"""
    (root, opts) => {
      const preferVisible = !!opts.preferVisible;
      const base = location.href;

      function abs(href) {
        try { return new URL(href, base).href; } catch(e) { return href; }
      }
      function cleanText(s) { return (s || "").replace(/\s+/g, " ").trim(); }

      function isVisible(el) {
        if (!el) return false;
        const style = window.getComputedStyle(el);
        if (!style) return false;
        if (style.display === "none" || style.visibility === "hidden" || style.opacity === "0") return false;
        if (el.hasAttribute("hidden")) return false;
        if (el.getAttribute("aria-hidden") === "true") return false;
        const rects = el.getClientRects();
        if (!rects || rects.length === 0) return false;
        const r = rects[0];
        return (r.width > 0 && r.height > 0);
      }

      // 选 ul：可见链接数优先
      const uls = Array.from(root.querySelectorAll("ul"));
      if (!uls.length) return null;

      let best = null;
      let bestV = -1;
      let bestA = -1;

      for (const ul of uls) {
        const links = Array.from(ul.querySelectorAll("a[href]"));
        const allCount = links.length;
        let visCount = 0;
        if (preferVisible) {
          for (const a of links) if (isVisible(a)) visCount++;
        } else {
          visCount = allCount;
        }
        if (visCount > bestV || (visCount === bestV && allCount > bestA)) {
          best = ul;
          bestV = visCount;
          bestA = allCount;
        }
      }
      if (!best) return null;

      function findChildUl(li) {
        const subs = Array.from(li.querySelectorAll("ul"));
        for (const sub of subs) {
          if (sub.closest("li") === li) return sub;
        }
        return null;
      }

      function liLabelWithoutSubtree(li) {
        const clone = li.cloneNode(true);
        // 移除子树，避免把子页面链接文本串进来
        clone.querySelectorAll("ul, nav").forEach(x => x.remove());
        return cleanText(clone.textContent);
      }

      function parseUl(ul) {
        const out = [];
        const lis = Array.from(ul.children).filter(x => x.tagName && x.tagName.toLowerCase() === "li");
        for (const li of lis) {
          const directA = li.querySelector(":scope > a[href]");
          let title = "";
          let url = null;

          if (directA) {
            title = cleanText(directA.textContent);
            url = abs(directA.getAttribute("href") || directA.href || "") || null;
          } else {
            title = liLabelWithoutSubtree(li);
            url = null;
          }

          const childUl = findChildUl(li);
          const children = childUl ? parseUl(childUl) : [];

          // 去除 “父节点与首个子节点完全重复” 的噪声（常见于 index 同名页）
          if (url && children && children.length) {
            const c0 = children[0];
            if (c0 && c0.url === url && (cleanText(c0.title) === cleanText(title))) {
              children.shift();
            }
          }

          out.push({ title: title || "(untitled)", url, children });
        }
        return out;
      }

      return parseUl(best);
    }
    """
    try:
        return root.evaluate(js, {"preferVisible": prefer_visible})
    except Exception:
        return None


def _extract_links_flat(root, prefer_visible: bool) -> List[Node]:
    js = r"""
    (root, opts) => {
      const preferVisible = !!opts.preferVisible;
      const base = location.href;

      function abs(href) {
        try { return new URL(href, base).href; } catch(e) { return href; }
      }
      function cleanText(s) { return (s || "").replace(/\s+/g, " ").trim(); }

      function isVisible(el) {
        if (!el) return false;
        const style = window.getComputedStyle(el);
        if (!style) return false;
        if (style.display === "none" || style.visibility === "hidden" || style.opacity === "0") return false;
        if (el.hasAttribute("hidden")) return false;
        if (el.getAttribute("aria-hidden") === "true") return false;
        const rects = el.getClientRects();
        if (!rects || rects.length === 0) return false;
        const r = rects[0];
        return (r.width > 0 && r.height > 0);
      }

      const out = [];
      const seen = new Set();
      const links = Array.from(root.querySelectorAll("a[href]"));
      for (const a of links) {
        if (preferVisible && !isVisible(a)) continue;
        const hrefAttr = a.getAttribute("href") || "";
        if (!hrefAttr) continue;
        if (hrefAttr.toLowerCase().startsWith("javascript:")) continue;
        const url = abs(hrefAttr);
        if (seen.has(url)) continue;
        seen.add(url);
        const title = cleanText(a.textContent) || hrefAttr;
        out.push({ title, url, children: [] });
      }
      return out;
    }
    """
    try:
        return root.evaluate(js, {"preferVisible": prefer_visible}) or []
    except Exception:
        return []


def _pick_target_container(page, user_selector: str):
    if user_selector:
        sel = user_selector.strip()
        if not re.match(r"^(css=|xpath=|text=|id=|data-testid=)", sel):
            sel = "css=" + sel
        return page.locator(sel).first
    # fallback：全页 body（不再做复杂自动猜测，保持你“只抓侧栏”的诉求）
    return page.locator("body").first


def _focus_current_part(tree: List[Node], current_url: str) -> List[Node]:
    """
    在 root 列表中，找“包含当前页面”的那棵 root（命中 url 或子树命中），仅输出它
    """
    cur = (current_url or "").rstrip("/")

    def norm(u: str) -> str:
        return (u or "").rstrip("/")

    def contains(node: Node) -> bool:
        u = norm(node.get("url") or "")
        if u and (u == cur or cur.startswith(u + "/")):
            return True
        for c in node.get("children") or []:
            if contains(c):
                return True
        return False

    for n in tree:
        if contains(n):
            return [n]
    return tree


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--name", default="")
    ap.add_argument("--project-root", default="")

    ap.add_argument("--selector", default="", help="强制指定容器（建议传 sidebar 的最内层 scroll/inner div）")

    ap.add_argument("--scope-origin", default="")
    ap.add_argument("--scope-prefix", default="")
    ap.add_argument("--keep-fragments", action="store_true")

    ap.add_argument("--prefer-visible", action="store_true")
    ap.add_argument("--focus", choices=["none", "current-part"], default="none")

    ap.add_argument("--headful", action="store_true")
    ap.add_argument("--wait-for-enter", action="store_true")
    ap.add_argument("--timeout-ms", type=int, default=60000)

    ap.add_argument("--user-data-dir", default="")
    ap.add_argument("--channel", default="")

    ap.add_argument("--expand-rounds", type=int, default=35)
    ap.add_argument("--scroll-steps", type=int, default=14)
    ap.add_argument("--pause", type=float, default=0.2)

    ap.add_argument("--debug-screenshot", action="store_true")
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

    with sync_playwright() as p:
        browser_type = p.chromium
        launch_kwargs = {"headless": (not args.headful)}
        if args.channel:
            launch_kwargs["channel"] = args.channel

        context = browser_type.launch_persistent_context(str(user_data_dir), **launch_kwargs)
        page = context.new_page()
        page.goto(args.url, wait_until="domcontentloaded", timeout=args.timeout_ms)

        if args.wait_for_enter:
            print("\n[WAIT] 你可在浏览器里展开/滚动目录，然后回终端按回车继续...", flush=True)
            input()

        root = _pick_target_container(page, args.selector)

        # ✅ v5：兼容 Material 的展开逻辑
        _expand_all_smart(root, rounds=args.expand_rounds, pause=args.pause)
        _scroll_container(root, steps=args.scroll_steps, pause=args.pause)
        _expand_all_smart(root, rounds=args.expand_rounds, pause=args.pause)

        current_url = page.url
        scope_origin = args.scope_origin.strip() or f"{urlparse(current_url).scheme}://{urlparse(current_url).netloc}"
        scope_prefix = args.scope_prefix.strip()

        tree = _extract_sidebar_tree(root, prefer_visible=args.prefer_visible)
        if not tree:
            tree = _extract_links_flat(root, prefer_visible=args.prefer_visible)

        tree = _filter_tree(tree, current_url, scope_origin, scope_prefix, keep_fragments=args.keep_fragments)

        if args.focus == "current-part":
            tree = _focus_current_part(tree, current_url)

        if not tree:
            tree = [{"title": page.title() or "Root", "url": current_url, "children": []}]

        (out_dir / "toc.json").write_text(json.dumps(tree, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "toc.md").write_text("\n".join(_render_md(tree)) + "\n", encoding="utf-8")
        urls = _dedupe_preserve_order(_flatten_urls(tree))
        (out_dir / "urls.txt").write_text("\n".join(urls) + "\n", encoding="utf-8")

        if args.debug_screenshot:
            try:
                page.screenshot(path=str(out_dir / "page.png"), full_page=True)
            except Exception:
                pass
            try:
                root.screenshot(path=str(out_dir / "container.png"))
            except Exception:
                pass

        context.close()

    print(f"OK: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
