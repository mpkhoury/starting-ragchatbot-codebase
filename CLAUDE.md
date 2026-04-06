# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Use `uv` for all dependency management — never use `pip` directly.

**Install dependencies:**
```bash
uv sync
```

**Add a dependency:**
```bash
uv add <package>
```

**Remove a dependency:**
```bash
uv remove <package>
```

**Run the server:**
```bash
cd backend && uv run uvicorn app:app --reload --port 8000
```

Or use the helper script from the project root:
```bash
./run.sh
```

The app serves at `http://localhost:8000` (web UI) and `http://localhost:8000/docs` (API docs).

**Environment setup:** Create a `.env` file in the project root with:
```
ANTHROPIC_API_KEY=your_key_here
```

There is no test suite in this project.

## Architecture

This is a RAG (Retrieval-Augmented Generation) system for querying course materials. The backend is a FastAPI app; the frontend is plain HTML/CSS/JS served as static files by FastAPI itself.

### Request flow

1. User submits a query via the web UI or `POST /api/query`
2. `RAGSystem.query()` builds a prompt and calls Claude via `AIGenerator.generate_response()`
3. Claude may invoke the `search_course_content` tool (Anthropic tool use)
4. `ToolManager` routes the tool call to `CourseSearchTool.execute()`
5. `CourseSearchTool` calls `VectorStore.search()`, which uses ChromaDB + sentence-transformers embeddings
6. The search result is returned to Claude, which produces the final answer
7. The exchange is saved in `SessionManager` for conversation history

### Key components

- **`backend/app.py`** — FastAPI entrypoint. Mounts frontend static files at `/`. On startup, loads all `.txt/.pdf/.docx` files from `../docs/` into the vector store.
- **`backend/rag_system.py`** — `RAGSystem` is the main orchestrator; wires all components together.
- **`backend/ai_generator.py`** — Wraps the Anthropic SDK. Handles the two-turn tool-use loop: first call may return `stop_reason="tool_use"`, then `_handle_tool_execution` runs the tools and makes a second call.
- **`backend/vector_store.py`** — ChromaDB wrapper with two collections: `course_catalog` (course-level metadata) and `course_content` (text chunks). Course name resolution uses vector similarity on the catalog before filtering content.
- **`backend/search_tools.py`** — `Tool` ABC + `CourseSearchTool` + `ToolManager`. Adding a new tool means subclassing `Tool`, implementing `get_tool_definition()` and `execute()`, then calling `tool_manager.register_tool()`.
- **`backend/document_processor.py`** — Parses plain-text course files into `Course`/`Lesson`/`CourseChunk` objects and splits content into overlapping sentence-based chunks.
- **`backend/session_manager.py`** — In-memory session store. History is passed as a string in the system prompt (not as structured messages).
- **`backend/config.py`** — All tuneable parameters (`CHUNK_SIZE`, `CHUNK_OVERLAP`, `MAX_RESULTS`, `MAX_HISTORY`, `CHROMA_PATH`, model name).

### Course document format

Plain-text files in `docs/` must follow this structure:
```
Course Title: <title>
Course Link: <url>
Course Instructor: <name>

Lesson 1: <lesson title>
Lesson Link: <url>
<lesson content...>

Lesson 2: <lesson title>
...
```

The `course_title` field is used as the ChromaDB document ID — duplicate titles are skipped on reload.
