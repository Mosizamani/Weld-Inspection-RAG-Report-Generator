"""Weld inspection pipeline: PDF RAG + optional RocketRide `agents.pipe` / `vision_agent.pipe`, with Gemini API fallback."""

from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from rocketride import RocketRideClient
from rocketride.schema import Question

# Pipeline files (same directory as this module); load `.env` from here so Jupyter/cwd does not matter
_DIR = Path(__file__).resolve().parent
load_dotenv(_DIR / ".env")

# Local engine URL used by the VS Code / CLI RocketRide DAP server
_DEFAULT_ROCKETRIDE_URI = "http://localhost:5565"
# Direct Gemini API (google-generativeai). `gemini-2.0-flash` is blocked for new keys — use 2.5+ (override via GEMINI_MODEL).
_DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

# Pipeline paths (relative to this module)
AGENTS_RAG_PIPE = _DIR / "agents.pipe"
VISION_PIPE = _DIR / "vision_agent.pipe"

_DEFAULT_RAG_PIPE_TARGET = "RAG_Node"
_DEFAULT_VISION_PIPE_TARGET = "Vision_Node"

_CODE_STANDARDS = {
    "aws_d11": "AWS D1.1",
    "icc": "International Building Code (ICC/IBC family)",
}

_USE_CHAT_RAG = (os.getenv("RAG_PIPE_USE_CHAT") or "").lower() in ("1", "true", "yes")


def _auth() -> str:
    return (os.getenv("ROCKETRIDE_APIKEY") or os.getenv("ROCKETRIDE_API_KEY") or "").strip()


def _uri() -> str:
    u = (os.getenv("ROCKETRIDE_URI") or _DEFAULT_ROCKETRIDE_URI).strip()
    return u.rstrip("/")


def _rag_target_provider() -> str:
    if _USE_CHAT_RAG:
        return "chat"
    return (os.getenv("RAG_PIPE_PROVIDER") or _DEFAULT_RAG_PIPE_TARGET).strip() or "chat"


def _vision_target_provider() -> str:
    if (os.getenv("VISION_PIPE_USE_WEBHOOK") or "").lower() in ("1", "true", "yes"):
        return "webhook"
    return (os.getenv("VISION_PIPE_PROVIDER") or _DEFAULT_VISION_PIPE_TARGET).strip() or "webhook"


def _connect_client() -> RocketRideClient:
    return RocketRideClient(uri=_uri(), auth=_auth())


def _gemini_api_key() -> str:
    return (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("ROCKETRIDE_GEMINI_API_KEY")
        or ""
    ).strip()


def _gemini_model_name() -> str:
    return (os.getenv("GEMINI_MODEL") or _DEFAULT_GEMINI_MODEL).strip()


def _prefer_direct_gemini() -> bool:
    return (os.getenv("USE_DIRECT_GEMINI") or "").strip().lower() in ("1", "true", "yes")


def _rocketride_unreachable(exc: BaseException) -> bool:
    if isinstance(exc, (ConnectionRefusedError, TimeoutError)):
        return True
    if isinstance(exc, OSError) and getattr(exc, "errno", None) in (61, 111):  # refused mac/linux
        return True
    msg = str(exc).lower()
    return (
        "errno 61" in msg
        or "connection refused" in msg
        or "connect call failed" in msg
        or "multiple exceptions" in msg
    )


def _connection_help() -> str:
    return (
        f"RocketRide DAP is not reachable at {_uri()!r}. Start the RocketRide engine, "
        "or set GEMINI_API_KEY in `.env` so the app can call Gemini directly when the engine is down "
        "(optional: USE_DIRECT_GEMINI=1 to skip RocketRide for text steps)."
    )


def _question_to_prompt(q: Question) -> str:
    chunks: List[str] = []
    if (q.role or "").strip():
        chunks.append(f"Role:\n{q.role}")
    for ins in q.instructions or []:
        chunks.append(f"### {ins.subtitle}\n{ins.instructions}")
    for i, g in enumerate(q.goals or [], 1):
        chunks.append(f"### Goal {i}\n{g}")
    for c in q.context or []:
        chunks.append(f"### Context\n{c}")
    for ex in q.examples or []:
        chunks.append(f"### Example\nGiven: {ex.given}\nExpected result:\n{ex.result}")
    for qt in q.questions or []:
        chunks.append(f"### Question\n{qt.text}")
    if q.expectJson:
        chunks.append("Respond with valid JSON only (no markdown code fences).")
    return "\n\n".join(chunks)


def _gemini_generate_text(prompt: str, *, expect_json: bool) -> str:
    import google.generativeai as genai

    key = _gemini_api_key()
    if not key:
        raise RuntimeError("No Gemini API key configured")
    genai.configure(api_key=key)
    model = genai.GenerativeModel(_gemini_model_name())
    cfg_kw: Dict[str, Any] = {}
    if expect_json:
        cfg_kw["response_mime_type"] = "application/json"
    cfg = genai.types.GenerationConfig(**cfg_kw) if cfg_kw else None
    response = model.generate_content(prompt, generation_config=cfg)
    return (response.text or "").strip()


async def _question_via_gemini(q: Question) -> Dict[str, Any]:
    prompt = _question_to_prompt(q)
    expect = bool(q.expectJson)
    text = await asyncio.to_thread(_gemini_generate_text, prompt, expect_json=expect)
    if expect:
        try:
            data: Any = json.loads(text)
        except (json.JSONDecodeError, TypeError, ValueError):
            data = text
        return {"answers": [data]}
    return {"answers": [text]}


async def _weld_vision_via_gemini(raw: bytes) -> Dict[str, Any]:
    import io

    import google.generativeai as genai
    from PIL import Image

    key = _gemini_api_key()
    if not key:
        raise RuntimeError("No Gemini API key configured")
    genai.configure(api_key=key)
    model = genai.GenerativeModel(_gemini_model_name())
    img = Image.open(io.BytesIO(raw))
    prompt = (
        "You are a Certified Welding Inspector (CWI) performing visual examination per AWS D1.1 practice. "
        "From this weld image, assess defect categories: porosity, undercut, slag. "
        "Return only valid minified JSON with keys: porosity (boolean or brief evidence object), "
        "undercut (boolean or brief evidence), slag (boolean or brief evidence), notes (string)."
    )

    def _call() -> Any:
        return model.generate_content(
            [prompt, img],
            generation_config=genai.types.GenerationConfig(response_mime_type="application/json"),
        )

    response = await asyncio.to_thread(_call)
    text = (response.text or "").strip()
    try:
        data: Any = json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        data = {"parse_error": True, "raw_response": text}
    return {"answers": [data]}


def _finalize_pipe_body(body: Any, stage: str) -> Dict[str, Any]:
    ap = _normalize_answers_to_json(body) if isinstance(body, dict) else body
    if isinstance(ap, str):
        try:
            j = json.loads(ap)
        except (json.JSONDecodeError, TypeError, ValueError):
            j = ap
        ap = j
    return {
        "ok": True,
        "stage": stage,
        "pipeline": "agents.pipe",
        "raw": body,
        "answers_parsed": ap,
    }


async def _run_agents_pipe(
    q: Question,
    *,
    stage: str,
) -> Dict[str, Any]:
    """Run `agents.pipe` via RocketRide, or Gemini directly if unreachable / USE_DIRECT_GEMINI."""
    if not AGENTS_RAG_PIPE.is_file():
        return _as_structured_error(stage, f"Missing pipeline file: {AGENTS_RAG_PIPE}")

    key = _gemini_api_key()
    if _prefer_direct_gemini():
        if not key:
            return _as_structured_error(
                stage,
                "USE_DIRECT_GEMINI is set but GEMINI_API_KEY (or GOOGLE_API_KEY / ROCKETRIDE_GEMINI_API_KEY) is missing.",
            )
        try:
            body = await _question_via_gemini(q)
        except Exception as exc:  # noqa: BLE001
            return _as_structured_error(stage, str(exc))
        out = _finalize_pipe_body(body, stage)
        out["target_provider"] = "gemini_direct"
        out["transport"] = "gemini_direct"
        return out

    prov = _rag_target_provider()
    body: Any = None
    last_exc: Optional[BaseException] = None
    client = _connect_client()
    try:
        await client.connect()
        use_result = await client.use(filepath=str(AGENTS_RAG_PIPE))
        token = use_result.get("token")
        if not token:
            return _as_structured_error(stage, "use() did not return a task token")
        objinfo: Dict[str, Any] = {"name": stage, "size": 1}
        pipe = await client.pipe(
            token,
            objinfo,
            "application/rocketride-question",
            provider=prov,
        )
        try:
            await pipe.open()
            await pipe.write(bytes(q.model_dump_json(), "utf-8"))
            body = await pipe.close() or {}
            last_exc = None
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if pipe.is_opened:
                try:
                    await pipe.close()
                except Exception:
                    pass
    except Exception as exc:  # noqa: BLE001
        last_exc = exc
    finally:
        try:
            await client.disconnect()
        except Exception:
            pass

    if body is not None:
        out = _finalize_pipe_body(body, stage)
        out["target_provider"] = prov
        out["transport"] = "rocketride"
        return out

    if key and last_exc is not None and _rocketride_unreachable(last_exc):
        try:
            body = await _question_via_gemini(q)
        except Exception as exc2:  # noqa: BLE001
            return _as_structured_error(
                stage,
                f"RocketRide failed ({last_exc!s}); direct Gemini fallback failed ({exc2!s}). {_connection_help()}",
            )
        out = _finalize_pipe_body(body, stage)
        out["target_provider"] = "gemini_direct"
        out["transport"] = "gemini_direct"
        return out

    err = str(last_exc) if last_exc else "RocketRide request failed"
    if last_exc is not None and _rocketride_unreachable(last_exc):
        err = f"{err}. {_connection_help()}"
    return _as_structured_error(stage, err)


def _format_snippet_brief(docs: List[Dict[str, Any]]) -> str:
    if not docs:
        return ""
    parts: List[str] = []
    for i, d in enumerate(docs, 1):
        m = d.get("metadata") or {}
        head = f"[{i}] " + (m.get("title") or m.get("code_standard") or m.get("parent") or "chunk")
        body = (d.get("page_content") or "")[: 1200]
        parts.append(f"{head}\n{body}\n")
    return "\n".join(parts).strip()


def _reference_pdf_dir() -> Path:
    raw = (
        (os.getenv("INSPECTION_REFERENCE_PDF_DIR") or os.getenv("USER_RAG_PDF_DIR") or "")
        .strip()
    )
    if raw:
        return Path(raw).expanduser().resolve()
    return _DIR / "data" / "inspection_reference_pdfs"


def _extract_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        return ""
    try:
        reader = PdfReader(str(path))
        parts: List[str] = []
        for page in reader.pages:
            t = page.extract_text() or ""
            if t.strip():
                parts.append(t)
        return "\n\n".join(parts)
    except Exception:
        return ""


def _chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200) -> List[str]:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    chunks: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        end = min(i + chunk_size, n)
        chunks.append(text[i:end])
        if end >= n:
            break
        i = max(0, end - overlap)
    return chunks


def _tokens_for_overlap(s: str) -> set[str]:
    return {w.lower() for w in re.findall(r"[A-Za-z0-9]+", s) if len(w) > 1}


def _score_chunk_for_query(chunk: str, query: str, notes: str) -> int:
    ct = _tokens_for_overlap(chunk)
    qn = _tokens_for_overlap(f"{query} {notes}")
    if not qn:
        return min(len(ct), 500)
    return len(ct & qn)


def _load_reference_pdf_snippets(
    query: str,
    user_notes: str,
    *,
    max_chunks: int = 10,
    chunk_size: int = 1500,
) -> List[Dict[str, Any]]:
    """
    Read all PDFs under INSPECTION_REFERENCE_PDF_DIR (default ``data/inspection_reference_pdfs``),
    chunk text, rank chunks by simple token overlap with the query and notes, and return the top slices.
    """
    root = _reference_pdf_dir()
    if not root.is_dir():
        return []
    scored: List[tuple[int, str, str]] = []
    for path in sorted(root.glob("*.pdf")):
        if not path.is_file():
            continue
        raw = _extract_pdf_text(path)
        if not raw.strip():
            continue
        label = path.name
        for ch in _chunk_text(raw, chunk_size=chunk_size):
            sc = _score_chunk_for_query(ch, query, user_notes)
            scored.append((sc, label, ch))
    if not scored:
        return []
    scored.sort(key=lambda x: (-x[0], x[1]))
    out: List[Dict[str, Any]] = []
    for sc, label, ch in scored:
        if len(out) >= max_chunks:
            break
        body = ch[: 1200]
        out.append(
            {
                "page_content": body,
                "metadata": {
                    "title": label,
                    "source": "inspection_reference_pdfs",
                    "score_hint": sc,
                },
            }
        )
    return out


def _normalize_answers_to_json(body: Optional[Dict[str, Any]]) -> Any:
    """Return first structured answer, or a plain dict of the full body for JSON-only callers."""
    if not body:
        return None
    if "answers" in body and body["answers"] is not None:
        a = body["answers"]
        if isinstance(a, list) and a:
            if len(a) == 1 and a[0] is not None:
                return a[0]
            return a
    return body


def _as_structured_error(stage: str, err: str) -> Dict[str, Any]:
    return {
        "ok": False,
        "stage": stage,
        "error": err,
    }


def _envelope_ok(payload: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    return {"ok": True, **extra, "result": payload}


async def get_rag_context(
    query: str,
    user_notes: str,
) -> Dict[str, Any]:
    """
    Run `agents.pipe` (or direct Gemini fallback) with a structured Question on RAG_Node.
    Context is built from ``data/inspection_reference_pdfs`` (or ``INSPECTION_REFERENCE_PDF_DIR``):
    PDF text is chunked and ranked by overlap with the query and inspector notes.
    """
    pdf_hits = _load_reference_pdf_snippets(query, user_notes)
    notes_context = f"User / inspector notes: {user_notes}\n" if (user_notes or "").strip() else ""
    code_scope = (
        f"Relevant code scope: { _CODE_STANDARDS['aws_d11'] } and { _CODE_STANDARDS['icc'] }."
    )
    pdf_brief = _format_snippet_brief(pdf_hits)
    if pdf_brief:
        code_scope = (
            code_scope + "\n\nExcerpts from inspection reference PDFs (project folder):\n" + pdf_brief
        )

    q = Question(expectJson=True)
    q.addInstruction(
        "Citations and scope",
        "Ground answers in AWS D1.1 and ICC/IBC building code material when applicable. "
        "If a topic is not covered, say so explicitly.",
    )
    if pdf_brief:
        q.addInstruction(
            "Reference PDF corpus",
            "Supplementary excerpts from the project's inspection reference PDFs are included in context; use "
            "them when they apply to the question (specs, WPS, drawings as text, or local requirements).",
        )
    if notes_context or pdf_brief:
        q.addContext(notes_context + code_scope)
    else:
        q.addContext(code_scope)
    q.addExample(
        "List base-plate requirements from notes",
        {
            "citations": [{"code": "AWS D1.1", "clause": "…"}],
            "summary": "…",
            "limitations": "…",
        },
    )
    q.addQuestion(query)

    pipe_out = await _run_agents_pipe(q, stage="get_rag_context")
    if not pipe_out.get("ok"):
        return pipe_out

    parsed = pipe_out.get("answers_parsed")
    body = pipe_out.get("raw")
    return _envelope_ok(
        {"query": query, "user_notes": user_notes, "code_standards": list(_CODE_STANDARDS.values())},
        {
            "pipeline": "agents.pipe",
            "target_provider": pipe_out.get("target_provider"),
            "transport": pipe_out.get("transport"),
            "mongo_hits": [],
            "pdf_hits": pdf_hits,
            "reference_pdf_dir": str(_reference_pdf_dir()),
            "raw": body,
            "answers_parsed": parsed,
        },
    )


async def detect_weld_defects(image_path: str) -> Dict[str, Any]:
    """
    Send weld image bytes through `vision_agent.pipe`, or call Gemini Vision directly if
    RocketRide is unreachable and GEMINI_API_KEY is set (same behavior as text RAG fallback).
    """
    path = Path(image_path)
    if not path.is_file():
        return _as_structured_error("detect_weld_defects", f"Not a file: {image_path}")

    raw = path.read_bytes()
    b64 = base64.standard_b64encode(raw).decode("ascii")
    mime, _ = mimetypes.guess_type(str(path))
    mime = mime or "application/octet-stream"

    if not VISION_PIPE.is_file():
        return _as_structured_error("detect_weld_defects", f"Missing pipeline file: {VISION_PIPE}")

    key = _gemini_api_key()
    if _prefer_direct_gemini():
        if not key:
            return _as_structured_error(
                "detect_weld_defects",
                "USE_DIRECT_GEMINI is set but GEMINI_API_KEY (or GOOGLE_API_KEY / ROCKETRIDE_GEMINI_API_KEY) is missing.",
            )
        try:
            body = await _weld_vision_via_gemini(raw)
        except Exception as exc:  # noqa: BLE001
            return _as_structured_error("detect_weld_defects", str(exc))
        parsed = _normalize_answers_to_json(body)
        if isinstance(parsed, str):
            try:
                parsed = json.loads(parsed)
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
        gm = _gemini_model_name()
        return _envelope_ok(
            {"defects_analysis": parsed},
            {
                "pipeline": "vision_agent.pipe",
                "image_path": str(path.resolve()),
                "mime_type": mime,
                "image_base64": b64,
                "target_provider": "gemini_direct",
                "transport": "gemini_direct",
                "cwi_model": gm,
                "defects_focus": ["porosity", "undercut", "slag"],
                "raw": body,
            },
        )

    prov = _vision_target_provider()
    size = max(len(raw), 1)
    body: Any = None
    last_exc: Optional[BaseException] = None
    client = _connect_client()
    try:
        await client.connect()
        use_result = await client.use(filepath=str(VISION_PIPE))
        token = use_result.get("token")
        if not token:
            return _as_structured_error("detect_weld_defects", "use() did not return a task token")
        objinfo: Dict[str, Any] = {"name": path.name, "size": size}
        pipe = await client.pipe(token, objinfo, mime, provider=prov)
        try:
            await pipe.open()
            await pipe.write(raw)
            body = await pipe.close() or {}
            last_exc = None
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if pipe.is_opened:
                try:
                    await pipe.close()
                except Exception:
                    pass
    except Exception as exc:  # noqa: BLE001
        last_exc = exc
    finally:
        try:
            await client.disconnect()
        except Exception:
            pass

    if body is not None:
        parsed: Any
        if isinstance(body, dict):
            parsed = _normalize_answers_to_json(body)
            try:
                if isinstance(parsed, str):
                    parsed = json.loads(parsed)
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
        else:
            parsed = body
        return _envelope_ok(
            {"defects_analysis": parsed},
            {
                "pipeline": "vision_agent.pipe",
                "image_path": str(path.resolve()),
                "mime_type": mime,
                "image_base64": b64,
                "target_provider": prov,
                "transport": "rocketride",
                "cwi_model": "gemini-2_0-flash",
                "defects_focus": ["porosity", "undercut", "slag"],
                "raw": body,
            },
        )

    if key and last_exc is not None and _rocketride_unreachable(last_exc):
        try:
            body = await _weld_vision_via_gemini(raw)
        except Exception as exc2:  # noqa: BLE001
            return _as_structured_error(
                "detect_weld_defects",
                f"RocketRide failed ({last_exc!s}); Gemini vision fallback failed ({exc2!s}). {_connection_help()}",
            )
        parsed = _normalize_answers_to_json(body)
        if isinstance(parsed, str):
            try:
                parsed = json.loads(parsed)
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
        gm = _gemini_model_name()
        return _envelope_ok(
            {"defects_analysis": parsed},
            {
                "pipeline": "vision_agent.pipe",
                "image_path": str(path.resolve()),
                "mime_type": mime,
                "image_base64": b64,
                "target_provider": "gemini_direct",
                "transport": "gemini_direct",
                "cwi_model": gm,
                "defects_focus": ["porosity", "undercut", "slag"],
                "raw": body,
            },
        )

    err = str(last_exc) if last_exc else "Vision pipe failed"
    if last_exc is not None and _rocketride_unreachable(last_exc):
        err = f"{err}. {_connection_help()}"
    return _as_structured_error("detect_weld_defects", err)


def _context_blob(label: str, data: Any, max_chars: int = 32_000) -> str:
    if data is None:
        return f"{label}: (none)\n"
    if isinstance(data, (dict, list)):
        try:
            s = json.dumps(data, default=str, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            s = str(data)
    else:
        s = str(data)
    if len(s) > max_chars:
        s = s[: max_chars - 3] + "..."
    return f"{label}:\n{s}\n\n"


def _llm_unwrap_text(answers_parsed: Any) -> str:
    """Return a string answer from a parsed pipe response (string, dict, or first list item)."""
    if answers_parsed is None:
        return ""
    if isinstance(answers_parsed, str):
        t = answers_parsed.strip()
        if t.startswith("{") or t.startswith("["):
            try:
                parsed = json.loads(answers_parsed)
                if isinstance(parsed, str):
                    return parsed.strip()
                if isinstance(parsed, dict) and "text" in parsed:
                    return str(parsed["text"]).strip()
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
        return t
    if isinstance(answers_parsed, dict):
        for key in (
            "polished_report",
            "field_inspection_report",
            "report",
            "text",
            "answer",
        ):
            v = answers_parsed.get(key)
            if v is not None and str(v).strip():
                return str(v).strip()
        return json.dumps(answers_parsed, default=str, ensure_ascii=False, indent=2)
    if isinstance(answers_parsed, list) and answers_parsed:
        return _llm_unwrap_text(answers_parsed[0])
    return str(answers_parsed).strip()


def _orchestrator_parse_json(answers_parsed: Any) -> Optional[Dict[str, Any]]:
    d: Any = answers_parsed
    if isinstance(d, str):
        try:
            d = json.loads(d)
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
    if not isinstance(d, dict):
        return None
    return d


async def generate_final_report(
    rag_data: Any,
    vision_data: Any,
    inspector_notes: str,
) -> Dict[str, Any]:
    """
    Orchestrator: compare vision weld findings to RAG regulatory limits (AWS D1.1 / ICC),
    then Agent 3 (separate template) for grammar and formatting. Returns a professional
    Field Inspection Report: project header, visual findings, code references, and
    a final line that must read ``NON-COMPLIANT`` when a vision metric exceeds
    the applicable RAG-stated limit.
    """
    notes = (inspector_notes or "").strip()
    bundle = _context_blob("RAG and regulatory data", rag_data) + _context_blob("Vision (image) analysis", vision_data)
    if notes:
        bundle = f"Inspector / site notes:\n{notes}\n\n" + bundle

    q1 = Question(expectJson=True)
    q1.addInstruction(
        "Role: Field inspection orchestrator (high-reasoning)",
        "You are a certified welding/structural field-inspection lead. "
        "Compare the Vision agent findings (measured or scored defects: porosity, undercut, slag, etc.) "
        "against the limits, tolerances, and requirements stated or implied in the inspection reference PDF "
        "excerpts and regulatory framing in the RAG payload "
        f"({ _CODE_STANDARDS['aws_d11'] } and { _CODE_STANDARDS['icc'] }). "
        "If the Vision data indicates a defect, dimension, or severity that exceeds a limit established "
        "by that material (or, when RAG is silent on a limit, you must not invent numeric limits; "
        "state the gap and use NEEDS_FURTHER_REVIEW or INCONCLUSIVE as appropriate), "
        "set compliance_status to the string 'NON-COMPLIANT' and explain exactly why. "
        "If Vision findings are within the RAG-sourced limits, use 'COMPLIANT'. "
        "Use clear professional language suitable for a formal report.",
    )
    q1.addInstruction(
        "Field Inspection Report (structure)",
        "In 'field_inspection_report', output Markdown with these sections in order: "
        "## Header (Project Information) — use any project/site info present in the inputs or state 'Not provided'; "
        "## Visual Findings — from the vision analysis, tables where helpful; "
        "## Regulatory Reference (AWS/ICC) — cite clauses/sections supported by the reference PDF excerpts in the RAG payload, not invented citations; "
        "## Final Compliance Status — one line beginning with: **COMPLIANT** or **NON-COMPLIANT** or **NEEDS_FURTHER_REVIEW** "
        "and echo compliance_status. "
        "The comparison rationale must make clear whether a vision-reported condition exceeds a RAG limit.",
    )
    q1.addContext(bundle)
    q1.addExample(
        "Synthesis",
        {
            "compliance_status": "NON-COMPLIANT",
            "rationale": "Porosity / aggregate defect severity X exceeds the acceptance limit Y stated under AWS D1.1 …",
            "compared_against": "E.g. AWS D1.1 table/clause or ICC/IBC reference as given in RAG (paraphrased, not fake clause numbers if absent).",
            "field_inspection_report": (
                "# Field Inspection Report\n\n## Header (Project Information)\n…\n\n## Visual Findings\n…\n\n"
                "## Regulatory Reference (AWS/ICC)\n…\n\n## Final Compliance Status\n**NON-COMPLIANT** — …\n"
            ),
        },
    )
    q1.addQuestion(
        "Produce the JSON only: 'compliance_status' (one of: COMPLIANT, NON-COMPLIANT, NEEDS_FURTHER_REVIEW, INCONCLUSIVE), "
        "optional 'rationale' and 'compared_against', and 'field_inspection_report' (full Markdown for the four sections).",
    )

    o = await _run_agents_pipe(q1, stage="orchestrate_report")
    if not o.get("ok"):
        return o

    parsed1 = o.get("answers_parsed")
    odict = _orchestrator_parse_json(parsed1) or {}
    if not odict.get("field_inspection_report") and _llm_unwrap_text(parsed1):
        odict = {
            "compliance_status": "INCONCLUSIVE",
            "rationale": "Unstructured model output was wrapped into the report field.",
            "compared_against": "",
            "field_inspection_report": _llm_unwrap_text(parsed1),
        }
    draft = (odict.get("field_inspection_report") or "").strip() or _llm_unwrap_text(parsed1)
    if not draft:
        return _as_structured_error("orchestrate_report", "Orchestrator returned no field_inspection_report text")
    com = (odict.get("compliance_status") or "").strip() or "INCONCLUSIVE"
    com_u = com.upper()
    if com_u == "NON-COMPLIANT" and "NON-COMPLIANT" not in draft and "Non-compliant" not in draft:
        draft = draft + "\n\n## Final Compliance Status\n**NON-COMPLIANT** — (See compliance_status in structured output.)\n"

    q2 = Question(expectJson=False)
    q2.addInstruction(
        "Agent 3: Grammar, clarity, and formatting (do not re-judge compliance)",
        "You are a technical editor and document formatter. Polish the given Field Inspection Report for professional tone, "
        "correct grammar, consistent headings, and clear Markdown. "
        "Do not change the factual or technical content: do not add measurements, do not remove NON-COMPLIANT flags, "
        "and do not weaken any stated code references. Do not re-run compliance; preserve **COMPLIANT** / **NON-COMPLIANT** "
        "/ **NEEDS_FURTHER_REVIEW** in the final section. Keep the four top-level section headings. "
        "Return only the polished report text, no preface or explanation.",
    )
    q2.addContext(draft)
    q2.addQuestion("Return the polished report only, as clean Markdown.")

    p = await _run_agents_pipe(q2, stage="format_report")
    if not p.get("ok"):
        return _envelope_ok(
            {
                "compliance_status": com,
                "field_inspection_report_draft": draft,
                "field_inspection_report": draft,
                "orchestrator_structured": odict or parsed1,
                "polish_failed": True,
                "polish_error": p,
            },
            {
                "pipeline": "agents.pipe",
                "orchestrator_target_provider": o.get("target_provider"),
                "orchestrator_raw": o.get("raw"),
            },
        )

    polished = _llm_unwrap_text(p.get("answers_parsed")).strip() or draft
    if not polished.strip():
        polished = draft
    return _envelope_ok(
        {
            "compliance_status": com,
            "orchestrator_structured": odict or parsed1,
            "field_inspection_report_draft": draft,
            "field_inspection_report": polished,
        },
        {
            "pipeline": "agents.pipe",
            "orchestrator_target_provider": o.get("target_provider"),
            "orchestrator_raw": o.get("raw"),
            "polish_target_provider": p.get("target_provider"),
            "polish_raw": p.get("raw"),
        },
    )


async def check_engine() -> bool:
    """Connect, ping, and disconnect. Returns True if the engine responds."""
    client = _connect_client()
    try:
        await client.connect()
        await client.ping()
        return True
    finally:
        await client.disconnect()


async def main() -> None:
    uri = _uri()
    print(f"Target RocketRide DAP: {uri}")
    try:
        ok = await check_engine()
    except Exception as exc:  # noqa: BLE001
        print(f"RocketRide engine is not reachable on {uri!r} ({type(exc).__name__}: {exc})")
    else:
        if ok:
            print(f"RocketRide engine is reachable on {uri!r}.")


if __name__ == "__main__":
    asyncio.run(main())
