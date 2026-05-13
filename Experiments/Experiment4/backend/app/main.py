"""FastAPI API and static frontend server for the ethnography assistant."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


REPO_ROOT = Path(__file__).resolve().parents[2]
ETHNO_SYSTEM_DIR = REPO_ROOT / "ethno_ai_system"
FRONTEND_DIR = REPO_ROOT / "frontend"

if str(ETHNO_SYSTEM_DIR) not in sys.path:
    sys.path.insert(0, str(ETHNO_SYSTEM_DIR))

from evaluation.results_writer import build_session_artifact, save_session_artifacts  # noqa: E402
from main import run_session  # noqa: E402
from utils.excel_loader import get_all_personas, get_persona_by_row, load_personas  # noqa: E402


class SessionRequest(BaseModel):
    """Request body for running one ethnographic session."""

    row: int = Field(default=1, ge=1)
    keep_memory: bool = False
    save_results: bool = True


class SessionResponse(BaseModel):
    """API response for one completed session."""

    artifact: Dict[str, Any]
    files: Dict[str, str] = Field(default_factory=dict)


app = FastAPI(
    title="Ethnography Assistant API",
    description="FastAPI wrapper around the local multi-agent ethnography system.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> Dict[str, str]:
    """Return a small readiness response."""
    return {"status": "ok"}


@app.get("/api/personas")
def list_personas() -> Dict[str, Any]:
    """Return available persona rows from the Excel dataset."""
    try:
        personas = get_all_personas()
    except Exception as exc:  # pragma: no cover - surfaced through API
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "count": len(personas),
        "personas": [
            {
                "row": index + 1,
                "name": persona["Name"],
                "problem": persona["Problem"],
                "person_details": persona["Person Details"],
                "ethnographic_solution": persona["Ethnographic Solution"],
            }
            for index, persona in enumerate(personas)
        ],
    }


@app.get("/api/personas/{row}")
def get_persona(row: int) -> Dict[str, Any]:
    """Return one persona row by 1-based row number."""
    try:
        persona = get_persona_by_row(row)
    except IndexError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - surfaced through API
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "row": row,
        "name": persona["Name"],
        "problem": persona["Problem"],
        "person_details": persona["Person Details"],
        "ethnographic_solution": persona["Ethnographic Solution"],
    }


@app.post("/api/sessions", response_model=SessionResponse)
async def create_session(request: SessionRequest) -> SessionResponse:
    """Run the full agent pipeline for one persona row."""
    try:
        dataframe = load_personas()
        if request.row > len(dataframe):
            raise IndexError(f"Row {request.row} is outside the dataset range 1..{len(dataframe)}")

        session_state = await asyncio.to_thread(
            run_session,
            row_number=request.row,
            cleanup=not request.keep_memory,
        )
        artifact = build_session_artifact(session_state)
        files: Dict[str, str] = {}

        if request.save_results:
            saved_paths = await asyncio.to_thread(save_session_artifacts, session_state)
            files = {key: str(path.relative_to(REPO_ROOT)) for key, path in saved_paths.items()}

        return SessionResponse(artifact=artifact, files=files)
    except IndexError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - surfaced through API
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="frontend-assets")


@app.get("/")
def frontend_index() -> FileResponse:
    """Serve the browser UI."""
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend is missing.")
    return FileResponse(index_path)
