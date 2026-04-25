# Ethnographic Multi-Agent AI Interview System

This project is a local prototype for ethnographic interviewing with three AI agents, a ChromaDB memory layer, a persona simulator, and an evaluation pipeline. The code is written in very simple Python so it is easy to inspect and extend.

## What the system does

1. Agent 1 collects the core problem and creates a structured intake summary.
2. Agent 2 asks ethnographic follow-up questions across relevant life domains.
3. Agent 3 retrieves stored summaries from ChromaDB, finds patterns, and gives ranked recommendations.
4. The whole flow runs through LangGraph.
5. Test scripts can run a single persona row or all rows in the Excel file.

## Project structure

```text
ethno_ai_system/
    main.py
    config.py
    requirements.txt
    README.md
    data/
        personas.xlsx
    agents/
        agent1_intake.py
        agent2_ethnography.py
        agent3_analysis.py
    memory/
        vector_store.py
    simulator/
        persona_agent.py
    evaluation/
        test_agent1.py
        test_agent2.py
        test_agent3.py
        test_full_system.py
        metrics.py
        plots.py
    utils/
        llm.py
        prompts.py
        excel_loader.py
        logger.py
```

## Install requirements

Create a virtual environment if you want one, then install dependencies:

```powershell
cd "D:\git repos\Ethnography Assistant\Experiments\Experiment4\ethno_ai_system"
pip install -r requirements.txt
```

## Start Ollama

Make sure Ollama is installed and the local server is running:

```powershell
ollama serve
```

## Pull llama3.2

Pull the model one time:

```powershell
ollama pull llama3.2
```

## Run a single session

```powershell
python main.py --row 1
```

## Run a single full-system evaluation

```powershell
python evaluation/test_full_system.py --row 1
```

## Run all rows

```powershell
python evaluation/test_full_system.py --all
```

## Run individual agent evaluations

```powershell
python evaluation/test_agent1.py --row 1
python evaluation/test_agent2.py --row 1
python evaluation/test_agent3.py --row 1
```

## Output files

The evaluation scripts create:

- `results/agent1_scores.csv`
- `results/agent2_scores.csv`
- `results/agent3_scores.csv`
- `results/full_system_scores.csv`
- `results/plots/summary_metrics.png`
- `results/plots/rowwise_performance.png`

Logs are also written to the `logs/` folder.

## Replace the vector store later

Only `memory/vector_store.py` needs to change if you want to replace ChromaDB with a graph vector database. The rest of the code only calls these functions:

- `init_store()`
- `add_document(session_id, category, text)`
- `search(session_id, query, top_k=3)`
- `delete_session(session_id)`

## Deterministic mode

This project defaults to deterministic settings for easier testing:

- temperature is `0.0`
- JSON parsing retries are enabled

You can change behavior with a `.env` file:

```env
OLLAMA_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434
DETERMINISTIC_MODE=true
JSON_RETRY_COUNT=3
MAX_ETHNOGRAPHY_DOMAINS=4
```

## Starter dataset note

The included `data/personas.xlsx` is a starter sample file so the project can run immediately. Replace it with your full 100-row Excel file using the required columns:

- `Name`
- `Problem`
- `Person Details`
- `Ethnographic Solution`

## Troubleshooting

If `python` is not recognized:

- use the Python executable from your virtual environment
- or use your system-specific launcher

If Ollama is running but the model call fails:

- confirm `ollama serve` is active
- confirm `ollama pull llama3.2` has already completed
- confirm `.env` points to the correct `OLLAMA_BASE_URL`

If ChromaDB creates stale data:

- delete the `chroma_store/` folder
- rerun the test

If the Excel file cannot be loaded:

- make sure the file is named `personas.xlsx`
- make sure the sheet has the required column names
- make sure `openpyxl` installed successfully
