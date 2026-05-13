# Ethnography Assistant

This repository contains a local multi-agent ethnography assistant with:

- a FastAPI backend in `backend/`
- a browser frontend in `frontend/`
- the existing agent system in `ethno_ai_system/`

The frontend is plain HTML, CSS, and JavaScript, so there is no Node.js build step.

## Run on Linux

Clone the repository, then run:

```bash
bash scripts/setup_linux.sh
```

Start Ollama in another terminal and make sure the model is available:

```bash
ollama serve
ollama pull llama3.2
```

Start the web app:

```bash
bash scripts/run_linux.sh
```

Open:

```text
http://localhost:8000
```

## Configuration

Copy `.env.example` to `.env` if you want to override defaults:

```bash
cp .env.example .env
```

Common values:

```env
OLLAMA_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434
DETERMINISTIC_MODE=true
JSON_RETRY_COUNT=3
MAX_ETHNOGRAPHY_DOMAINS=4
```

## API

- `GET /api/health`
- `GET /api/personas`
- `GET /api/personas/{row}`
- `POST /api/sessions`

Example session request:

```bash
curl -X POST http://localhost:8000/api/sessions \
  -H "Content-Type: application/json" \
  -d '{"row": 1, "save_results": true, "keep_memory": false}'
```

## Existing CLI

The original command-line workflow still works:

```bash
cd ethno_ai_system
python main.py --row 1
```
