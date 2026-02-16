# kb_engine_playground

A **backend-only** showcase of how to use [LangChain](https://python.langchain.com/) to enable RAG: automated **fetch → parse → chunk → embed → store** into Postgres (pgvector), with a **RAG agent** you can use via **CLI** or **LangGraph Studio**.

**License:** [MIT](LICENSE).

---

## The demo (no separate frontend)

The **demo** is **LangGraph Studio**: you chat there, powered by a **small Ollama model**, and the agent is hooked to the **vector DB** via the **search_kb** tool. So you get a working prototype without building a frontend—the KB engine was never designed to be a frontend app.

1. **Ingestion works** → docs are in pgvector.  
2. **Vector search works** → the agent has a `search_kb` tool that runs proximity search.  
3. **You run `langgraph dev`** → Studio is the UI; Ollama is the LLM; the agent calls the vector DB when it needs context.

**Example in Studio:** ask *"How do I set up Google Cloud architecture to support 2FA?"* — the agent will call `search_kb` against the ingested AWS/GCP docs and answer from that context.

---

## What this project does

| Component | Description |
|-----------|-------------|
| **Ingestion pipeline** | Fetch documents from configured sources → parse (LangChain document loaders) → chunk → embed → store in **Postgres + pgvector**. |
| **Config-driven sources** | Add or remove doc sources via a single config file; the pipeline supports **N sources** (URLs, paths, or other loaders). |
| **Scheduled runs** | **Kubernetes CronJob** (MicroK8s-friendly) runs ingestion on a schedule so the vector DB stays up to date. |
| **RAG agent** | A LangChain/LangGraph agent that retrieves from the vector DB and answers questions. Use it via **CLI** or **LangGraph Studio** (local). |

---

## Demo: Cloud architecture RAG

The default doc sources use **easy-to-ingest** formats: **official PDF whitepapers** (no JS, full-text) and **GitHub** (READMEs and docs). This keeps ingestion reliable and demo-ready.

| Source | Type | Content |
|--------|------|---------|
| **AWS Overview** | PDF | [aws-overview.pdf](https://docs.aws.amazon.com/pdfs/whitepapers/latest/aws-overview/aws-overview.pdf) |
| **AWS Well-Architected Framework** | PDF | [wellarchitected-framework.pdf](https://docs.aws.amazon.com/pdfs/wellarchitected/latest/framework/wellarchitected-framework.pdf) |
| **AWS Security Best Practices** | PDF | [aws-security-best-practices.pdf](https://docs.aws.amazon.com/pdfs/whitepapers/latest/aws-security-best-practices/aws-security-best-practices.pdf) |
| **GCP python-docs-samples** | GitHub | [GoogleCloudPlatform/python-docs-samples](https://github.com/googlecloudplatform/python-docs-samples) — README, `docs/`, `getting_started/` (.md, .rst, .txt) |

For **GitHub** sources, set `GITHUB_PERSONAL_ACCESS_TOKEN` in `.env` (or in the CronJob secret) for higher rate limits; see [Config: doc sources](#config-doc-sources).

**Example questions after ingestion:**  
*"What does the AWS Well-Architected Framework say about the reliability pillar?"* · *"Summarize AWS security best practices."* · *"How do I get started with Google Cloud Python samples?"*

---

## Tech stack

- **Python** + **LangChain** (document loaders, text splitters, embeddings, vector stores)
- **Postgres** + **pgvector**
- **Docker** & **Docker Compose** (local dev)
- **MicroK8s** (local K8s; CronJob for ingestion)

---

## Run and test

### In a dev container (recommended)

Develop and run everything inside a container: same Python and deps as prod. **Postgres (pgvector) and Ollama start with the dev container** so the stack is self-contained.

1. **Open in dev container**  
   In Cursor/VS Code: **Command Palette** → **Dev Containers: Reopen in Container** (or **Open in Container** when opening the folder).  
   The first time, it builds the app image and starts Postgres + Ollama; the workspace is mounted at `/workspaces/kb_engine_playground`.

2. **First time only: pull the embedding model**  
   Ollama runs in a dedicated container. From a terminal on the **host** (in the project directory):
   ```bash
   docker compose -f .devcontainer/docker-compose.yml exec ollama ollama pull mxbai-embed-large
   ```
   (Optional: for the RAG LLM, also run `ollama pull llama3.2`.)

3. **Configure environment (inside the container)**
   ```bash
   cp .env.example .env
   # Edit .env if needed: optional GITHUB_PERSONAL_ACCESS_TOKEN for GitHub doc sources.
   # DATABASE_URL and OLLAMA_BASE_URL are already set for the dev stack.
   ```

4. **Init DB and run ingestion**
   ```bash
   python scripts/init_db.py
   python -m src.ingestion.pipeline
   ```

5. **Test the RAG agent**
   ```bash
   python -m src.cli cli "What does the AWS Well-Architected Framework say about reliability?"
   ```
   Or start the server for LangGraph Studio / API:
   ```bash
   python -m src.cli server
   ```
   Then open http://localhost:8123 or point [LangGraph Studio](https://github.com/langchain-ai/langgraph-studio) at it.

All of the above runs in the dev container; Postgres and Ollama are already up. No need to run `docker compose` from the host for day-to-day dev (except the one-time model pull).

### With Docker Compose only (no dev container)

If you prefer to run on the host or in CI, the stack is self-contained: **Postgres (pgvector) and Ollama** run as services.

1. **Clone and enter the project**
   ```bash
   cd kb_engine_playground
   ```

2. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env if needed (e.g. optional GITHUB_PERSONAL_ACCESS_TOKEN).
   # DATABASE_URL and OLLAMA_BASE_URL are set by docker-compose for the services.
   ```

3. **Start Postgres + Ollama**
   ```bash
   docker compose up -d postgres ollama
   ```
   Wait for Postgres to be ready (e.g. 10–15 seconds).

4. **First time only: pull the embedding model** (and optionally the LLM)
   ```bash
   docker compose exec ollama ollama pull mxbai-embed-large
   docker compose exec ollama ollama pull llama3.2   # optional, for RAG answers
   ```

5. **Optional: ensure pgvector extension** (if the DB is fresh)
   ```bash
   docker compose run --rm ingestion python scripts/init_db.py
   ```

6. **Run ingestion** (fetch → parse → chunk → embed → store)
   ```bash
   docker compose run --rm ingestion
   ```

7. **Validate embeddings and chat with the RAG agent**
   - **Search the vector DB** (validate that ingestion worked):
     ```bash
     docker compose run --rm agent python -m src.cli search "reliability pillar" -k 5
     ```
   - **CLI**
     ```bash
     docker compose run --rm agent python -m src.cli cli "Your question here"
     ```
   - **LangGraph Studio / API**  
     Start the server: `docker compose run --rm -p 8123:8123 agent python -m src.cli server`. Then call `POST /chat` with `{"message": "..."}` or point [LangGraph Studio](https://github.com/langchain-ai/langgraph-studio) at the graph.

---

## Project layout

```
kb_engine_playground/
├── README.md
├── LICENSE
├── langgraph.json          # LangGraph Studio / langgraph dev entrypoint (rag_agent)
├── docker-compose.yml
├── .env.example
├── .devcontainer/          # Dev container (Open in Container)
│   ├── devcontainer.json
│   ├── docker-compose.yml
│   └── Dockerfile
├── config/
│   └── doc_sources.yaml    # N doc sources (URLs, PDFs, GitHub, etc.)
├── src/
│   ├── ingestion/          # fetch → parse → chunk → embed → store
│   │   ├── __init__.py
│   │   ├── loaders.py      # config-driven document loading
│   │   ├── pipeline.py     # full pipeline
│   │   └── store.py        # pgvector upsert
│   ├── agent/              # RAG agent (search_kb tool + LangGraph)
│   │   ├── __init__.py
│   │   ├── graph.py        # LangGraph RAG graph + agent with vector-search tool
│   │   └── server.py       # FastAPI server for /chat
│   └── cli.py              # CLI entrypoint (ingestion, search, cli, server)
├── k8s/                    # CronJob, ConfigMap, Secrets (MicroK8s)
│   ├── cronjob-ingestion.yaml
│   └── ...
├── scripts/                # one-off helpers (e.g. init_db.py)
└── requirements.txt
```

---

## Config: doc sources

Edit `config/doc_sources.yaml` to add or remove sources. Each source has:

- **id**: unique key (used for namespacing or metadata).
- **type**: `url`, `urls`, `pdf_url`, `pdf_urls`, or `github_repo`.
- **url** / **urls** / **repo**, etc., depending on type.

**Supported types:**

| type | Required fields | Description |
|------|-----------------|-------------|
| `url` | `url` | Single HTML page (WebBaseLoader). |
| `urls` | `urls` | Multiple HTML pages. |
| `pdf_url` | `url` | Single PDF from URL (downloaded, then PyPDFLoader). |
| `pdf_urls` | `urls` | Multiple PDFs from URLs. |
| `github_repo` | `repo` | Files from a GitHub repo. Optional: `branch`, `include_paths`, `file_extensions`. Requires `GITHUB_PERSONAL_ACCESS_TOKEN` in env for reliable rate limits. |

Example (cloud architecture demo—see `config/doc_sources.yaml`):

```yaml
sources:
  - id: aws_well_architected
    type: pdf_url
    url: "https://docs.aws.amazon.com/pdfs/wellarchitected/latest/framework/wellarchitected-framework.pdf"
  - id: gcp_python_docs_samples
    type: github_repo
    repo: "GoogleCloudPlatform/python-docs-samples"
    branch: "main"
    include_paths: ["README", "docs/", "getting_started/"]
    file_extensions: [".md", ".rst", ".txt"]
```

The ingestion pipeline loads these with LangChain (WebBaseLoader, PyPDFLoader, GithubFileLoader), chunks (using **language-aware text splitters** for GitHub: Markdown and RST use LangChain's `RecursiveCharacterTextSplitter.from_language`), embeds with Ollama (e.g. **mxbai-embed-large**), and stores them in pgvector.

### Embedding model limit and chunking

**mxbai-embed-large** has a **512-token** max sequence length ([Hugging Face model card](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1)). The pipeline respects this by:

- **Chunk size**: Default chunk size is **500 characters** (safely under 512 tokens). Override with `EMBED_MAX_CHARS` in `.env` if your model supports more.
- **No truncation**: Chunking uses 500 chars + overlap. LangChain’s splitter can sometimes produce chunks larger than `chunk_size` when there is no separator (e.g. long lines in GitHub `.md`/`.rst`). The pipeline **re-splits** any such chunk into 500-char pieces instead of truncating, so no content is dropped and the embedding model never receives overlength input.

---

## Running ingestion

- **Local (Docker Compose)** — run everything in the container (Postgres + Ollama + ingestion):
  ```bash
  docker compose up -d postgres ollama
  # First time: init DB and pull embedding model
  docker compose run --rm ingestion python scripts/init_db.py
  docker compose exec ollama ollama pull mxbai-embed-large
  # Run ingestion (fetch PDFs + GitHub → parse → chunk → embed → store)
  docker compose run --rm ingestion
  # Validate: search the vector DB
  docker compose run --rm agent python -m src.cli search "reliability pillar" -k 5
  ```

- **Kubernetes (MicroK8s)**  
  Apply the CronJob and related manifests (see [k8s/](#k8s-microk8s)).
  ```bash
  kubectl apply -f k8s/
  ```

---

## RAG agent

- **CLI**
  ```bash
  docker compose run --rm agent python -m src.cli cli "What does the AWS Well-Architected Framework say about reliability?"
  ```

- **LangGraph Studio (the demo)**  
  The agent exposes a **search_kb** tool (vector DB). Chat in Studio; a **small Ollama model** powers the LLM and the agent uses the vector DB to answer.

  1. **Install the CLI** (once):  
     `pip install "langgraph-cli[inmem]"`
  2. **Postgres + Ollama up** (e.g. `docker compose up -d postgres ollama`). **Pull a small LLM** for the chat (e.g. `docker compose exec ollama ollama pull llama3.2` or `ollama pull phi3:mini` for lighter hardware). Set `.env`: `DATABASE_URL`, `OLLAMA_BASE_URL` (e.g. `http://localhost:11435` if Ollama is on host port 11435).
  3. **From the project root:**  
     `langgraph dev`  
  Open **http://127.0.0.1:2024**, select **rag_agent**, and chat. Example: *"How do I set up Google Cloud architecture to support 2FA?"* — the agent will call **search_kb** on the vector DB and answer from the ingested docs.

  Alternatively, run the FastAPI server (`docker compose run --rm -p 8123:8123 agent python -m src.cli server`) and use `POST /chat` or point [LangGraph Studio](https://github.com/langchain-ai/langgraph-studio) at it.

---

## K8s (MicroK8s)

Manifests in `k8s/` define:

- **CronJob** for the ingestion job (e.g. daily or hourly).
- **ConfigMap** for `doc_sources.yaml` (or mount from a volume).
- **Secrets** for `DATABASE_URL` and any API keys (embedding/LLM).

Use `microk8s kubectl apply -f k8s/` (or alias `kubectl` to `microk8s kubectl`).

---

## Showcasing this repo

- **As a backend portfolio piece:** The repo is ready to push. It shows config-driven ingestion, embedding-model-aware chunking, pgvector, and a LangGraph agent with a vector-search tool. On GitHub, set a clear **description** and **topics** (e.g. `rag`, `langchain`, `pgvector`, `langgraph`, `docker`).
- **Push to GitHub:** Create a new repo on GitHub (no need to add a README there). Then from the project root:
  ```bash
  git init
  git add .
  git commit -m "Initial commit: KB engine RAG pipeline + LangGraph Studio demo"
  git branch -M main
  git remote add origin https://github.com/YOUR_USERNAME/kb_engine_playground.git
  git push -u origin main
  ```
  Ensure `.env` is not committed (it’s in `.gitignore`). After pushing, add the description and topics in the repo settings.
- **To put a live demo on the web:** Add a minimal UI (e.g. Streamlit or a small Next.js page) that calls your RAG API, and deploy the stack (e.g. Fly.io, Railway, or Render) so visitors can try it without running Docker locally.

---

## License

See [LICENSE](LICENSE). MIT — use and adapt for your own RAG pipelines.
