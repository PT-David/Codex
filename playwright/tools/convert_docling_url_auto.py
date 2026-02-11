# docling_convert_auto.py
from __future__ import annotations

"""
Docling URL/PDF/etc -> Markdown/HTML/JSON/YAML/TEXT
自动检测版本（含 Big5 识别提示）
---------------------------------------------------
策略要点：
A) 若 HTTP Header 或 <meta> 声明 charset：
   - 优先按声明编码解码（strict 优先）
   - strict 失败则对“同一编码家族”做 replace（不乱跳到 gb18030）

B) 若没有 charset：
   - 组合候选编码（utf-8 / gb18030 / big5 家族 等）
   - 用一个简单评分函数选择“更像中文网页”的结果

C) 若检测到 big5：
   - 输出提示：建议改用 docling_convert_big5.py
"""

import argparse
import json
import re
import contextlib
import io
import logging
import sys
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

try:
    import requests
except ImportError as e:
    raise SystemExit(
        "Missing dependency: requests\n"
        "Install in the SAME venv:\n"
        "  py -m pip install -U requests"
    ) from e

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions, TableFormerMode

# OCR option classes may vary by docling version; import defensively
try:
    from docling.datamodel.pipeline_options import (
        OcrAutoOptions,
        EasyOcrOptions,
        RapidOcrOptions,
        TesseractOcrOptions,
        TesseractCliOcrOptions,
    )
except Exception:  # pragma: no cover
    OcrAutoOptions = None  # type: ignore
    EasyOcrOptions = None  # type: ignore
    RapidOcrOptions = None  # type: ignore
    TesseractOcrOptions = None  # type: ignore
    TesseractCliOcrOptions = None  # type: ignore

from docling.document_converter import DocumentConverter, PdfFormatOption

try:
    from docling.document_converter import ImageFormatOption  # type: ignore
except Exception:
    ImageFormatOption = None  # noqa: N816

try:
    from docling_core.types.doc import ImageRefMode
except Exception:
    from docling.datamodel.document import ImageRefMode


def _is_url(s: str) -> bool:
    try:
        u = urlparse(s)
        return u.scheme in ("http", "https") and bool(u.netloc)
    except Exception:
        return False


def _requests_fetch(url: str, headers: Optional[dict] = None, timeout: int = 60):
    h = headers or {}
    return requests.get(url, headers=h, timeout=timeout)


def _load_headers(headers_json: Optional[str], headers_kv: Optional[str]) -> dict:
    """
    headers_json: path to json file containing headers dict, or a JSON string
    headers_kv: "k1:v1;k2:v2"
    """
    headers: dict = {}
    if headers_json:
        p = Path(headers_json)
        if p.exists():
            headers.update(json.loads(p.read_text(encoding="utf-8")))
        else:
            # assume raw json string
            headers.update(json.loads(headers_json))

    if headers_kv:
        # parse k:v;...
        parts = [x.strip() for x in headers_kv.split(";") if x.strip()]
        for kv in parts:
            if ":" not in kv:
                continue
            k, v = kv.split(":", 1)
            headers[k.strip()] = v.strip()

    return headers


def _parse_image_mode(mode: str) -> ImageRefMode:
    mode = (mode or "").lower()
    if mode == "placeholder":
        return ImageRefMode.PLACEHOLDER
    if mode == "referenced":
        return ImageRefMode.REFERENCED
    return ImageRefMode.EMBEDDED


def _safe_stem_from_source(src: str) -> str:
    if _is_url(src):
        # use url path stem-ish
        u = urlparse(src)
        base = (u.path.rsplit("/", 1)[-1] or "url").strip()
        base = re.sub(r"[^0-9A-Za-z._-]+", "_", base)
        return base[:80] or "url"
    p = Path(src)
    base = p.stem
    base = re.sub(r"[^0-9A-Za-z._-]+", "_", base)
    return base[:80] or "file"


def _extract_charset_from_content_type(ct: str) -> Optional[str]:
    # e.g. text/html; charset=big5
    m = re.search(r"charset\s*=\s*([A-Za-z0-9._-]+)", ct, flags=re.I)
    return m.group(1).strip() if m else None


def _extract_charset_from_meta(raw: bytes) -> Optional[str]:
    # search within first N bytes
    head = raw[:4096].decode("latin1", errors="ignore")
    # <meta charset="utf-8">
    m = re.search(r"<meta[^>]+charset=['\"]?([A-Za-z0-9._-]+)", head, flags=re.I)
    if m:
        return m.group(1).strip()
    # <meta http-equiv="Content-Type" content="text/html; charset=big5">
    m = re.search(r"charset\s*=\s*([A-Za-z0-9._-]+)", head, flags=re.I)
    return m.group(1).strip() if m else None


def _ensure_utf8_meta(html_text: str) -> str:
    # make sure <meta charset="utf-8"> exists
    if re.search(r"<meta[^>]+charset=", html_text, flags=re.I):
        # normalize to utf-8 if possible
        html_text = re.sub(
            r"(<meta[^>]+charset=['\"]?)([A-Za-z0-9._-]+)(['\"]?)",
            r"\1utf-8\3",
            html_text,
            flags=re.I,
        )
        return html_text
    # insert after <head ...>
    return re.sub(
        r"(<head[^>]*>)",
        r"\1\n<meta charset=\"utf-8\">",
        html_text,
        count=1,
        flags=re.I,
    )


def _score_text_for_zh(text: str) -> float:
    # simple heuristic: count CJK + punctuation, penalize replacement chars
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    repl = text.count("\ufffd")
    # if mostly ASCII, score low
    return cjk * 2.0 - repl * 3.0


def _decode_with_family_fallback(raw: bytes, enc: str) -> Tuple[str, str, str]:
    """
    Try strict decode with enc. If fails, use replace with SAME family only.
    """
    try:
        return raw.decode(enc, errors="strict"), enc, "strict"
    except Exception:
        return raw.decode(enc, errors="replace"), enc, "replace"


def _decode_html_auto(
    raw: bytes,
    header_charset: Optional[str],
    meta_charset: Optional[str],
    force_encoding: Optional[str] = None,
) -> Tuple[str, str, str, bool]:
    """
    Returns: (text, used_encoding, error_mode, looks_big5)
    """
    if force_encoding:
        txt, used, mode = _decode_with_family_fallback(raw, force_encoding)
        looks_big5 = used.lower().startswith(("big5", "cp950"))
        return txt, used, mode, looks_big5

    declared = meta_charset or header_charset
    if declared:
        # honor declared encoding
        txt, used, mode = _decode_with_family_fallback(raw, declared)
        looks_big5 = used.lower().startswith(("big5", "cp950"))
        return txt, used, mode, looks_big5

    # 无声明：候选集 + 评分
    candidates = [
        "utf-8",
        "utf-8-sig",
        "gb18030",
        "big5hkscs",
        "cp950",
        "big5",
        "latin1",  # 最终兜底：不会失败，但通常分数很低
    ]

    best = None  # (score, text, enc, mode)
    for enc in candidates:
        for mode, err in (("strict", "strict"), ("replace", "replace")):
            try:
                txt = raw.decode(enc, errors=err)
            except Exception:
                continue
            score = _score_text_for_zh(txt)
            if best is None or score > best[0]:
                best = (score, txt, enc, mode)
            if mode == "strict":
                break

    if best is None:
        txt = raw.decode("utf-8", errors="replace")
        return txt, "utf-8", "replace", False

    looks_big5 = best[2].lower().startswith(("big5", "cp950"))
    return best[1], best[2], best[3], looks_big5


def _parse_csv_langs(lang_csv: Optional[str]) -> Optional[list[str]]:
    """Parse 'a,b,c' -> ['a','b','c']; return None if empty."""
    if not lang_csv:
        return None
    langs = [x.strip() for x in lang_csv.split(",") if x.strip()]
    return langs or None


def _model_field_names(cls) -> set[str]:
    # pydantic v2: model_fields; v1: __fields__
    if hasattr(cls, "model_fields"):
        return set(getattr(cls, "model_fields").keys())
    if hasattr(cls, "__fields__"):
        return set(getattr(cls, "__fields__").keys())
    return set()


def _make_ocr_options(*, engine: str, force_full_page: bool, lang_csv: Optional[str], psm: Optional[int]):
    """Create a Docling OCR options object matching CLI args."""
    eng = (engine or "auto").strip().lower()
    langs = _parse_csv_langs(lang_csv)

    # Map engine -> option class
    cls_map = {
        "auto": OcrAutoOptions,
        "easyocr": EasyOcrOptions,
        "rapidocr": RapidOcrOptions,
        "tesseract": TesseractOcrOptions,
        "tesseract_cli": TesseractCliOcrOptions,
    }
    cls = cls_map.get(eng)
    if cls is None:
        return None

    kwargs = {"force_full_page_ocr": bool(force_full_page)}
    if langs is not None:
        kwargs["lang"] = langs
    if psm is not None:
        kwargs["psm"] = int(psm)

    # Drop unsupported kwargs for this docling version
    fields = _model_field_names(cls)
    if fields:
        kwargs = {k: v for k, v in kwargs.items() if k in fields}

    try:
        return cls(**kwargs)
    except Exception:
        # Last resort: try without optional knobs
        try:
            return cls(force_full_page_ocr=bool(force_full_page))
        except Exception:
            return None


class _FilteredStream(io.TextIOBase):
    """Stream wrapper that drops lines containing certain substrings."""
    def __init__(self, stream, drop_substrings: tuple[str, ...]):
        self._stream = stream
        self._drop = drop_substrings
        self._buf = ""

    def write(self, s: str) -> int:  # type: ignore[override]
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if not any(sub in line for sub in self._drop):
                self._stream.write(line + "\n")
        return len(s)

    def flush(self) -> None:  # type: ignore[override]
        if self._buf:
            if not any(sub in self._buf for sub in self._drop):
                self._stream.write(self._buf)
            self._buf = ""
        self._stream.flush()


@contextlib.contextmanager
def _suppress_rapidocr_empty_lines(enabled: bool):
    """Suppress noisy 'RapidOCR returned empty result!' lines (non-fatal)."""
    if not enabled:
        yield
        return
    drop = ("RapidOCR returned empty result!",)
    out = _FilteredStream(sys.stdout, drop)
    err = _FilteredStream(sys.stderr, drop)
    with redirect_stdout(out), redirect_stderr(err):
        yield
    out.flush()
    err.flush()


def build_converter(
    *,
    out_format: str,
    image_export_mode: str = "embedded",  # ✅ 默认对齐 CLI：embedded
    compact_tables: bool = False,
    # pipeline knobs
    ocr: bool = True,  # ✅ 默认对齐 CLI：True
    ocr_engine: str = "auto",  # ✅ 默认对齐 CLI：auto
    force_ocr: bool = False,  # ✅ 默认对齐 CLI：False
    ocr_lang: Optional[str] = None,
    psm: Optional[int] = None,
    tables: bool = True,  # ✅ 默认对齐 CLI：True
    table_mode: str = "accurate",  # ✅ 默认对齐 CLI：accurate
    cell_matching: bool = True,  # 表格列合并异常时可关
    enrich_formula: bool = False,  # ✅ 默认对齐 CLI：False
    enrich_code: bool = False,  # ✅ 默认对齐 CLI：False
    images_scale: float = 1.0,

    need_images: bool,
    artifacts_path: Optional[str],
) -> DocumentConverter:
    pipeline = PdfPipelineOptions()
    pipeline.do_ocr = bool(ocr)

    # --- OCR configuration ---
    if pipeline.do_ocr:
        ocr_opts = _make_ocr_options(engine=ocr_engine, force_full_page=force_ocr, lang_csv=ocr_lang, psm=psm)
        if ocr_opts is not None:
            pipeline.ocr_options = ocr_opts

        # RapidOCR is quite sensitive to rendering resolution; if user kept images_scale at 1.0,
        # upscale a bit to avoid empty detections on small fonts.
        try:
            if (ocr_engine or "").strip().lower() == "rapidocr" and float(images_scale) < 2.0:
                images_scale = 1.0  #初始代码是2.0，后因为跑崩电脑，降至1.0
        except Exception:
            pass

    # --- Tables ---
    pipeline.do_table_structure = bool(tables)
    if tables:
        pipeline.table_structure_options = TableStructureOptions(do_cell_matching=bool(cell_matching))
        pipeline.table_structure_options.mode = (
            TableFormerMode.FAST if table_mode.lower() == "fast" else TableFormerMode.ACCURATE
        )

    # images_scale also affects page rendering used by OCR; set it regardless of export mode.
    try:
        pipeline.images_scale = float(images_scale)
    except Exception:
        pipeline.images_scale = 1.0

    if need_images:
        pipeline.generate_picture_images = True

    if artifacts_path:
        pipeline.artifacts_path = artifacts_path

    format_options = {InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline)}
    if ImageFormatOption is not None:
        format_options[InputFormat.IMAGE] = ImageFormatOption(pipeline_options=pipeline)

    return DocumentConverter(format_options=format_options)


def export_doc(
    doc,
    *,
    out_path: Path,
    fmt: str,
    image_mode: ImageRefMode,
    artifacts_dir: Optional[Path],
    compact_tables: bool,
):
    if fmt == "md":
        doc.save_as_markdown(out_path, artifacts_dir=artifacts_dir, image_mode=image_mode, compact_tables=compact_tables)
    elif fmt == "html":
        doc.save_as_html(out_path, artifacts_dir=artifacts_dir, image_mode=image_mode)
    elif fmt == "json":
        doc.save_as_json(out_path, artifacts_dir=artifacts_dir, image_mode=image_mode)
    elif fmt == "yaml":
        doc.save_as_yaml(out_path, artifacts_dir=artifacts_dir, image_mode=image_mode)
    else:
        out_path.write_text(doc.export_to_text(), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description="Docling converter - Auto encoding detection (with Big5 hint).")
    p.add_argument("sources", nargs="+", help="Input sources: local files or URLs")
    p.add_argument("--outdir", default=".", help="Output directory")
    p.add_argument("--to", dest="out_format", default="md", choices=["md", "html", "json", "yaml", "text"])

    # 账号登录信息获取
    p.add_argument("--headers-json", default=None)
    p.add_argument("--headers", default=None)

    # 图片输出模式（默认 embedded，对齐 docling CLI）
    p.add_argument("--image-export-mode", default="embedded", choices=["placeholder", "embedded", "referenced"])
    p.add_argument("--compact-tables", action=argparse.BooleanOptionalAction, default=False, help="Compact tables in Markdown.")
    p.add_argument("--images-scale", type=float, default=1.0)

    # OCR（默认开启，对齐 CLI）
    p.add_argument("--ocr", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ocr-engine", default="auto", help="auto|rapidocr|tesseract|easyocr . (depends on installed deps)")
    p.add_argument("--force-ocr", action=argparse.BooleanOptionalAction, default=False, help="Force full-page OCR")
    p.add_argument("--ocr-lang", default=None, help="Comma-separated OCR languages (engine-specific)")
    p.add_argument("--psm", type=int, default=None, help="Tesseract PSM (0-13), if supported")

    # 表格（默认开启，对齐 CLI）
    p.add_argument("--tables", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--table-mode", choices=["fast", "accurate"], default="accurate")
    p.add_argument(
        "--cell-matching",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Map table structure back to PDF cells (turn off if columns get merged incorrectly)",
    )
    
   # 公式/代码增强（默认关闭，对齐 CLI）
    p.add_argument("--enrich-formula", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--enrich-code", action=argparse.BooleanOptionalAction, default=False)

    # 模型产生的中间文件保存位置
    p.add_argument("--artifacts-path", default=None, help="Local model artifacts path (offline / custom cache)")

    p.add_argument("--save-utf8-html", action=argparse.BooleanOptionalAction, default=True)

    # 自动检测脚本也允许你“手工强制”编码（极端情况下救急）
    p.add_argument("--force-url-encoding", default=None, help="Force decoding encoding for HTML (e.g. big5hkscs/cp950/big5/utf-8)")

    args = p.parse_args()

    # Reduce RapidOCR noise: empty OCR result is usually non-fatal (e.g., pages/images without text).
    _using_rapidocr = bool(args.ocr) and (str(args.ocr_engine).strip().lower() == "rapidocr")
    if _using_rapidocr:
        logging.getLogger("RapidOCR").setLevel(logging.ERROR)


    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    headers = _load_headers(args.headers_json, args.headers)

    image_mode = _parse_image_mode(args.image_export_mode)
    need_images = image_mode != ImageRefMode.PLACEHOLDER

    converter = build_converter(
        ocr=args.ocr,
        out_format=args.out_format,
        image_export_mode=args.image_export_mode,
        compact_tables=args.compact_tables,
        ocr_engine=args.ocr_engine,
        force_ocr=args.force_ocr,
        ocr_lang=args.ocr_lang,
        psm=args.psm,
        cell_matching=args.cell_matching,
        enrich_formula=args.enrich_formula,
        enrich_code=args.enrich_code,
        tables=args.tables,
        table_mode=args.table_mode,
        images_scale=args.images_scale,
        need_images=need_images,
        artifacts_path=args.artifacts_path,
    )


    ext_map = {"md": "md", "html": "html", "json": "json", "yaml": "yaml", "text": "txt"}

    for src in args.sources:
        stem = _safe_stem_from_source(src)
        out_path = outdir / f"{stem}.{ext_map[args.out_format]}"

        art_dir = None
        if image_mode == ImageRefMode.REFERENCED:
            art_dir = outdir / f"{stem}_artifacts"
            art_dir.mkdir(parents=True, exist_ok=True)

        if _is_url(src):
            r = _requests_fetch(src, headers=headers)
            ct = r.headers.get("Content-Type", "")
            ce = r.headers.get("Content-Encoding", "")

            print(f" Content-Type: {ct}")
            print(f" Content-Encoding: {ce}")

            if "html" in ct.lower():
                raw = r.content
                header_cs = _extract_charset_from_content_type(ct)
                meta_cs = _extract_charset_from_meta(raw)

                text, used, mode, looks_big5 = _decode_html_auto(
                    raw,
                    header_charset=header_cs,
                    meta_charset=meta_cs,
                    force_encoding=args.force_url_encoding,
                )

                print(f"  header_charset={header_cs}, meta_charset={meta_cs}")
                print(f"  used={used} ({mode})")
                if looks_big5 and not args.force_url_encoding:
                    print("  NOTE: looks like Big5 family page. If output is still garbled, run docling_convert_big5.py")

                text = _ensure_utf8_meta(text)

                utf8_html_path = outdir / f"{stem}.utf8.html"
                if args.save_utf8_html:
                    utf8_html_path.write_text(text, encoding="utf-8", errors="strict")
                    print(f"  saved utf8 html -> {utf8_html_path}")

                with _suppress_rapidocr_empty_lines(_using_rapidocr):
                    result = converter.convert_string(text, InputFormat.HTML, name=f"{stem}.html")
                export_doc(result.document, out_path=out_path, fmt=args.out_format,
                           image_mode=image_mode, artifacts_dir=art_dir, compact_tables=args.compact_tables)
                continue

            # 非 HTML：直接交给 docling convert(url, headers=...)
            with _suppress_rapidocr_empty_lines(_using_rapidocr):
                result = converter.convert(src, headers=headers)
            export_doc(result.document, out_path=out_path, fmt=args.out_format,
                       image_mode=image_mode, artifacts_dir=art_dir, compact_tables=args.compact_tables)
            continue

        # 本地文件
        with _suppress_rapidocr_empty_lines(_using_rapidocr):
            result = converter.convert(src)
        export_doc(result.document, out_path=out_path, fmt=args.out_format,
                   image_mode=image_mode, artifacts_dir=art_dir, compact_tables=args.compact_tables)


if __name__ == "__main__":
    main()
