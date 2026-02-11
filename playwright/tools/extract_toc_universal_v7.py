# extract_toc_universal_v6.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

from playwright.sync_api import sync_playwright

Node = Dict[str, Any]


def _safe_name_from_url(url: str) -> str:
    u = urlparse(url)
    host = re.sub(r"[^0-9A-Za-z._-]+", "_", u.netloc or "site")
    path = re.sub(r"[^0-9A-Za-z._-]+", "_", (u.path.strip("/") or "root"))
    return (host + "_" + path)[:80]


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


def _is_fragment_only(url: str, current_url: str) -> bool:
    try:
        u = urlparse(url)
        cur = urlparse(current_url)
        if not u.fragment:
            return False
        return (u.netloc == cur.netloc) and (u.path == cur.path) and (u.query == cur.query)
    except Exception:
        return False


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
    return re.sub(r"([^0-9A-Za-z_-])", lambda m: "\\" + m.group(1), s)


def _expand_all_smart(root, rounds: int, pause: float):
    # 兼容 Material for MkDocs：checkbox/label toggles + summary + aria-expanded
    for _ in range(rounds):
        clicked = 0

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

        labels = root.locator("label[for]")
        try:
            n3 = labels.count()
        except Exception:
            n3 = 0
        for i in range(n3):
            lab = labels.nth(i)
            try:
                fid = (lab.get_attribute("for") or "").strip()
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

        // 若祖先 aria-hidden=true，也视为不可见
        if (el.closest && el.closest('[aria-hidden="true"]')) return false;

        return true;
      }

      const uls = Array.from(root.querySelectorAll("ul"));
      if (!uls.length) return null;

      function depthFrom(node, boundary) {
        let d = 0;
        let cur = node;
        while (cur && cur !== boundary) {
          d += 1;
          cur = cur.parentElement;
        }
        return d;
      }

      function hasUlAncestorWithin(node, boundary) {
        let cur = node.parentElement;
        while (cur && cur !== boundary) {
          if ((cur.tagName || "").toLowerCase() === "ul") return true;
          cur = cur.parentElement;
        }
        return false;
      }

      function topLevelUls(boundary) {
        const all = Array.from(boundary.querySelectorAll("ul"));
        return all.filter(ul => !hasUlAncestorWithin(ul, boundary));
      }

      let best = null;
      let bestA = -1;

      for (const ul of uls) {
        const links = Array.from(ul.querySelectorAll("a[href]"));
        const allCount = links.length;
        const visBucket = (preferVisible && !isVisible(ul)) ? 0 : 1;
        const bestBucket = best ? ((preferVisible && !isVisible(best)) ? 0 : 1) : -1;
        const thisDepth = depthFrom(ul, root);
        const bestDepth = best ? depthFrom(best, root) : Number.MAX_SAFE_INTEGER;

        if (
          visBucket > bestBucket ||
          (visBucket === bestBucket && allCount > bestA) ||
          (visBucket === bestBucket && allCount === bestA && thisDepth < bestDepth)
        ) {
          best = ul;
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
        clone.querySelectorAll("ul, nav").forEach(x => x.remove());
        return cleanText(clone.textContent);
      }

      function parseUl(ul) {
        const out = [];
        const lis = Array.from(ul.children).filter(x => x.tagName && x.tagName.toLowerCase() === "li");
        for (const li of lis) {
          let directA = li.querySelector(":scope > a[href]");
          if (!directA) {
            const candidates = Array.from(li.querySelectorAll("a[href]"));
            for (const a of candidates) {
              if (a.closest("li") === li) {
                directA = a;
                break;
              }
            }
          }
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

          if (url && children && children.length) {
            const c0 = children[0];
            if (c0 && c0.url === url && cleanText(c0.title) === cleanText(title)) {
              children.shift();
            }
          }

          out.push({ title: title || "(untitled)", url, children });
        }
        return out;
      }

      const navRoot = best.closest("nav") || root;
      const roots = topLevelUls(navRoot);
      const preferredRoots = roots.length ? roots : [best];

      const merged = [];
      for (const ul of preferredRoots) {
        if (preferVisible && !isVisible(ul)) continue;
        merged.push(...parseUl(ul));
      }

      return merged.length ? merged : parseUl(best);
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
        if (el.closest && el.closest('[aria-hidden="true"]')) return false;
        return true;
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


def _locator_from_selector(page, selector: str):
    sel = selector.strip()
    if not sel:
        return page.locator("body").first
    if re.match(r"^(css=|xpath=|text=|id=|data-testid=)", sel):
        return page.locator(sel).first
    return page.locator("css=" + sel).first


def _get_active_top_tab_title(page) -> str:
    """
    尝试读取“当前激活 tab”的标题（Material tabs 常见）
    """
    js = r"""
    () => {
      const sels = [
        ".md-tabs a[aria-current='page']",
        ".md-tabs a.md-tabs__link--active",
        "header nav a[aria-current='page']",
        "header nav a[class*='active']",
        "nav a[aria-current='page']"
      ];
      for (const sel of sels) {
        const el = document.querySelector(sel);
        if (el && el.textContent) {
          return el.textContent.replace(/\s+/g, " ").trim();
        }
      }
      return "";
    }
    """
    try:
        return (page.evaluate(js) or "").strip()
    except Exception:
        return ""


def _norm_title(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


def _common_path_prefix(urls: List[str]) -> str:
    # 返回以 / 结尾的 path 前缀
    paths = []
    for u in urls:
        try:
            p = urlparse(u).path
            if p:
                paths.append(p)
        except Exception:
            continue
    if not paths:
        return "/"

    segs = [p.strip("/").split("/") for p in paths]
    lcp: List[str] = []
    for parts in zip(*segs):
        if all(x == parts[0] for x in parts):
            lcp.append(parts[0])
        else:
            break
    if not lcp:
        return "/"
    return "/" + "/".join(lcp) + "/"


def _focus_current_part(tree: List[Node], current_url: str, active_title: str) -> List[Node]:
    # 1) 优先按 active tab 标题锁定
    at = _norm_title(active_title)
    if at:
        for n in tree:
            if _norm_title(n.get("title") or "") == at:
                return [n]
        # 容错：前缀匹配（有的站点标题带图标/计数）
        for n in tree:
            nt = _norm_title(n.get("title") or "")
            if nt and (nt.startswith(at) or at.startswith(nt)):
                return [n]

    # 2) fallback：用 leaf URL 的“最长公共 path 前缀”匹配当前页 path
    curp = urlparse(current_url).path.rstrip("/") + "/"
    best: Optional[Tuple[int, Node]] = None
    for n in tree:
        leaves = _flatten_urls([n], [])
        if not leaves:
            continue
        pref = _common_path_prefix(leaves)
        if curp.startswith(pref):
            score = len(pref)
            if best is None or score > best[0]:
                best = (score, n)
    if best:
        return [best[1]]

    return tree


def _inject_part_home_as_child(tree: List[Node], active_title: str, current_url: str) -> List[Node]:
    """
    若当前 part 的根节点是无链接分组（常见于 Material），则把当前页 URL
    作为该分组下的第一个叶子，避免“part 首页文档”缺失。
    """
    at = _norm_title(active_title)
    if not at:
        return tree

    out: List[Node] = []
    injected = False
    for n in tree:
        title = n.get("title") or ""
        url = (n.get("url") or "").strip() or None
        children = list(n.get("children") or [])

        if (not injected) and _norm_title(title) == at:
            if not url:
                exists = any((c.get("url") or "").strip() == current_url for c in children)
                if not exists:
                    children = [{"title": title, "url": current_url, "children": []}] + children
                injected = True
            else:
                injected = True

        out.append({"title": title, "url": url, "children": children})

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--name", default="")
    ap.add_argument("--project-root", default="")

    ap.add_argument("--selector", default="")
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

        root = _locator_from_selector(page, args.selector)

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
            active_title = _get_active_top_tab_title(page)
            tree = _focus_current_part(tree, current_url, active_title)
            tree = _inject_part_home_as_child(tree, active_title, current_url)

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
