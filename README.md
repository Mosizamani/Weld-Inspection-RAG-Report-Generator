## Weld Inspection RAG Report Generator

Purpose
Small Python service and Jupyter demo for visual weld inspection: combine inspector text, reference PDF excerpts, and an LLM, then compare automated weld-image analysis to that regulatory/spec context and emit a structured field inspection narrative (compliance framing, Markdown report).

## Architecture

Layer	Role
app.py
Async API: get_rag_context, detect_weld_defects, generate_final_report, optional check_engine.
RocketRide .pipe graphs
Default path: agents.pipe (question → Gemini node → answers), vision_agent.pipe (image → vision/LLM steps).
Direct Gemini
If localhost:5565 RocketRide DAP is down (or USE_DIRECT_GEMINI=1), the same Question payloads and a dedicated vision multimodal call go to google-generativeai, keyed by GEMINI_API_KEY (or GOOGLE_API_KEY / ROCKETRIDE_GEMINI_API_KEY).

## RAG behavior (PDF-only corpus)
Corpus: data/inspection_reference_pdfs/*.pdf (override: INSPECTION_REFERENCE_PDF_DIR).
Ingest: No separate vector DB; text is extracted with pypdf, split into overlapping chunks, and ranked by token overlap with blueprint query + inspector notes.
Prompting: Top chunks plus fixed AWS D1.1 / ICC framing are injected into a rocketride.schema.Question, then sent through agents.pipe or direct Gemini (default model gemini-2.5-flash via GEMINI_MODEL).

## Vision and reporting
Vision: Weld image bytes are sent through vision_agent.pipe when RocketRide is up; otherwise Gemini multimodal with a CWI-style JSON-focused prompt (porosity, undercut, slag).
Final report: generate_final_report runs a two-stage Question flow via the same agents.pipe/Gemini path: synthesize Markdown + compliance_status, then a light polish pass without changing compliance facts.
## Configuration and environment
.env loaded from the app.py directory (load_dotenv(_DIR / ".env")) so Jupyter cwd does not affect secrets.
env.example documents RocketRide URIs/API keys vs GEMINI_API_KEY / USE_DIRECT_GEMINI.

## Dependencies (high level)
python-dotenv, rocketride, google-generativeai, Pillow, pypdf, notebook stack (jupyter, ipywidgets, nest_asyncio). Reference PDF directory contents are **gitignore**d (*.pdf).

## Constraints and caveats
PDF extraction depends on selectable text (scanned-only PDFs need OCR upstream).
Retrieval is lexical overlap over local chunks, not embedding-based retrieval in this repo path.
RocketRide profiles in .pipe files must stay aligned with whichever Gemini profile the hosted engine exposes if you rely on localhost execution.
