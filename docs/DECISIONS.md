# Technology & Design Decisions (options considered → choice → why)

Ground rules driving every decision:
1. **POC, zero cost** — no paid services, no API keys, no cloud calls at runtime.
2. **Runs locally on a 32 GB Apple-Silicon Mac** (Ollama already installed).
3. **Satisfies the PDF** (LangGraph, vector DB, memory, ≥4 tools, ≥3 skills, LangSmith evaluators).
4. Prefer **few moving parts** and actively maintained, permissively licensed packages (security policy).

Scoring legend: ✅ good fit · ⚠️ acceptable/with caveats · ❌ rejected.

---

## D1. Agentic design pattern
| Option | Verdict | Notes |
|---|---|---|
| Single ReAct agent (LLM loops over tools until done) | ⚠️ | Simple, but persistence, memory gating and verification would depend on the model remembering to do them. Weak against small local models. |
| Plan-and-execute | ⚠️ | Investigation path depends on retrieved evidence (conflicts/gaps), so an upfront plan is often wrong and costs an extra LLM call. |
| Multi-agent supervisor (triager, investigator, responder agents) | ⚠️ | Over-engineered for POC; multiplies local-LLM calls/latency. |
| Fixed pipeline (RAG chain) | ❌ | Violates “model must decide when a tool is needed”. |
| **ReAct investigator inside an evaluator–optimizer graph + HITL** | ✅ **Chosen** | Model chooses tools (flexible); graph guarantees lifecycle, routing, bounded loops, verification, interrupts. Maps 1:1 to the PDF flow (retrieve → investigate/ask → resolve/escalate → evaluate → persist → index). |

## D2. LLM runtime and model
| Option | Verdict | Why |
|---|---|---|
| OpenAI / Anthropic / Gemini APIs | ❌ | Paid / API-key based — excluded by your constraint. |
| Hosted free tiers (Groq, HF Inference, OpenRouter free) | ❌ | Still API-based, rate limits, key management, data leaves machine. |
| llama.cpp / LM Studio / vLLM | ⚠️ | Work, but Ollama is already installed and you asked for Ollama explicitly. |
| **Ollama** | ✅ **Chosen** | Local, free, already installed, first-class LangChain integration, serves both chat and embeddings. |

Model candidates (all via Ollama):
| Model | Tool calling | Size | Verdict |
|---|---|---|---|
| `llama3:latest` (installed) | ❌ no native tool support in this tag | 4.7 GB | Rejected for agent role |
| `mistral:latest` (installed) | ⚠️ limited/unreliable | 4.4 GB | Rejected as primary |
| `deepseek-r1:8b` (installed) | ❌ reasoning model, no tools | 5.2 GB | Rejected for tool loop (could be a judge) |
| `gpt-oss:20b` (installed) | ✅ tools + reasoning | 13 GB | **Fallback** — no download, slower |
| **`qwen3:8b`** (to pull) | ✅ strong tool use / JSON for its size | ~5 GB | ✅ **Primary** — best speed/reliability balance for 8B-class on this hardware |
| `llama3.1:8b` | ✅ | ~4.9 GB | Viable alternative; qwen3 generally better at structured output |
Model is a single config value (`CHAT_MODEL`) so it can be swapped without code changes.

## D3. Embedding model
| Option | Verdict | Why |
|---|---|---|
| OpenAI / Cohere / Voyage embeddings | ❌ | Paid / API. |
| `sentence-transformers` (bge-small, MiniLM) | ⚠️ | Free & good, but pulls PyTorch (heavy) and adds a second runtime; wheel availability risk on newest Python. |
| **Ollama `nomic-embed-text`** | ✅ **Chosen** | One runtime (Ollama), small (~270 MB), 768-d, good retrieval quality, long context (8K), batch `/api/embed`. |
| Ollama `mxbai-embed-large` / `bge-m3` | ⚠️ | Higher quality but 2–4× slower to embed 28K rows; can swap if recall is poor. |

## D4. Vector database
| Option | Verdict | Why |
|---|---|---|
| Pinecone / Weaviate Cloud / Qdrant Cloud | ❌ | Paid/API. |
| FAISS | ⚠️ | Fast, but no metadata filtering or persistence/upsert story → we would hand-build it. |
| LanceDB | ⚠️ | Good embedded option; smaller LangChain ecosystem, fewer examples. |
| Qdrant (local mode/Docker) | ⚠️ | Excellent, but local mode is less standard; Docker adds infra the PDF says not to spend time on. |
| pgvector | ❌ | Needs a Postgres server — overkill for POC. |
| sqlite-vec | ⚠️ | Would unify with SQLite, but young project/extension loading friction on macOS. |
| **ChromaDB (persistent client)** | ✅ **Chosen** | Embedded, zero-ops, metadata `where` filters, upsert/add for the *growing* index, first-class `langchain-chroma`. |

## D5. Keyword / hybrid search
| Option | Verdict | Why |
|---|---|---|
| Vector only | ⚠️ | Misses exact error codes/product names common in tickets. |
| `rank-bm25` in memory | ⚠️ | Another dependency, rebuilt at start, doesn't update incrementally. |
| Elasticsearch/OpenSearch | ❌ | Heavy infrastructure. |
| **SQLite FTS5 (BM25) + Reciprocal Rank Fusion** | ✅ **Chosen** | Built into SQLite (no new dependency), incremental inserts for live cases, RRF needs no score calibration. |

## D6. Structured (case) database
| Option | Verdict | Why |
|---|---|---|
| PostgreSQL / MySQL | ❌ | Server to run; unnecessary at this scale. |
| MongoDB / TinyDB | ⚠️ | Schemaless is not needed; relational queries (stats, history) are natural in SQL. |
| DuckDB | ⚠️ | Great for analytics, weaker for concurrent small writes/transactions. |
| **SQLite (WAL)** | ✅ **Chosen** | Stdlib, local, transactional, supports FTS5, same engine as the LangGraph checkpointer → single-file ops, trivial backup. |

## D7. Short-term memory / checkpointing
| Option | Verdict | Why |
|---|---|---|
| `InMemorySaver` | ❌ | Lost on restart → breaks “resume later” across processes (CLI). |
| Postgres/Redis checkpointer | ❌ | Extra infra. |
| **`langgraph-checkpoint-sqlite` (`SqliteSaver`)** | ✅ **Chosen** | Durable interrupt/resume across CLI invocations, official, no infra. |

## D8. Long-term memory
| Option | Verdict | Why |
|---|---|---|
| LangGraph `Store` (in-memory / sqlite) | ⚠️ | Fits conceptually; but we need custom columns (kind, confidence, last_used) and gating logic anyway. |
| Mem0 / Zep / LangMem | ⚠️ | Extra dependencies; Mem0/Zep lean on cloud or additional services; opaque “what is stored” — assessment explicitly evaluates *what we decide to remember*. |
| Store whole conversations | ❌ | Violates “avoid storing unnecessary information”. |
| **Own `customer_memory` SQLite table + LLM extraction with fixed schema + deterministic filter** | ✅ **Chosen** | Transparent, testable, small, demonstrates the decision policy. Can be wrapped as a LangGraph `Store` later. |

## D9. Orchestration features
| Decision | Choice | Why |
|---|---|---|
| Human-in-the-loop | `langgraph.types.interrupt` + `Command(resume=…)` | Current LangGraph-recommended mechanism (replaces static `interrupt_before` for dynamic questions). |
| Tool execution | `bind_tools` + `ToolNode` in a bounded loop | Model decides tools; framework executes. |
| Parallelism | Fan-out of 4 retrieval branches with reducer on `evidence` | Real latency gain; earns the bonus point. |
| Prebuilt `create_react_agent` | Not used for the whole system | Hides topology — assessment grades topology/state design; we may use it as a *building block* only if it simplifies `investigate`. |

## D10. Skills mechanism
| Option | Verdict | Why |
|---|---|---|
| One big system prompt with all skills | ❌ | Explicitly disallowed by PDF; wastes context on small models. |
| Hard-coded prompt per node | ⚠️ | Not “reusable/dynamic”. |
| **Markdown `SKILL.md` + frontmatter, catalogue in prompt, triage pre-selects, `load_skill` tool for on-demand** | ✅ **Chosen** | Mirrors progressive-disclosure pattern; reusable, testable, small context footprint. |

## D11. Evaluation
| Option | Verdict | Why |
|---|---|---|
| LangSmith cloud datasets + hosted tracing | ⚠️ | PDF names LangSmith, but cloud account = API key; free tier exists yet conflicts with “API-free POC” if made mandatory. Kept as **optional**. |
| **LangSmith SDK `evaluate()` with local examples and custom evaluators (`upload_results=False`)** | ✅ **Chosen** | Uses LangSmith evaluator interface and dataset schema as required, no account/key. |
| RAGAS / DeepEval | ⚠️ | Overlapping, heavier, default to OpenAI judges; would need re-pointing to Ollama. |
| LLM judge | **Ollama local model** | Free, private; known weaker than frontier judges → mitigate with code-based checks for retrieval/tool/behaviour and rubric prompts for judge scores. |
| Hold-out split | 200 tickets excluded from index | Prevents test leakage. |

## D12. Interface
| Option | Verdict | Why |
|---|---|---|
| React/Streamlit UI | ❌ | PDF: don't spend time on UI. |
| **Typer CLI (+ `rich`)** | ✅ **Chosen** | Fits interrupt/resume (`ticket` → `resume`), scriptable demo. |
| FastAPI | ⚠️ optional | Thin wrapper over same graph; included only if approved (+0.5 d). |

## D13. Language/tooling
| Decision | Choice | Why |
|---|---|---|
| Python | **3.12** (not the installed 3.14.6) | Newest interpreter risks missing wheels for chromadb/numpy ecosystem; 3.12 is mainstream-supported by all chosen libs. `uv` installs it automatically. |
| Package manager | `uv` (installed 0.7.21) | Fast, lockfile (`uv.lock`) satisfies dependency-lock rule. |
| Validation | Pydantic v2 | Structured outputs, settings, tool schemas. |
| Quality | ruff, mypy, pytest, pre-commit, gitleaks, pip-audit | Meets security/quality rules at zero cost. |

## D14. Dataset handling
| Decision | Choice | Why |
|---|---|---|
| Source | HF `Tobi-Bueck/customer-support-tickets`, filter `language=en` | PDF mandates this as sole knowledge. |
| Download | `datasets`/`huggingface_hub` public download (no token) | Public dataset; one-time. |
| Embedded text | subject + body only (answer stored, not embedded) | Query is a problem statement; matching on problem avoids retrieving by answer wording. |
| IDs | Synthetic `H-######` | Dataset has none; needed for `get_ticket_by_id` and citations. |
| Customer ID | Caller-supplied | Dataset has none; honest limitation. |

## D15. Confidence score
LLM self-reported confidence is poorly calibrated on small models → **computed**: evidence agreement 0.4 + retrieval strength 0.2 + verifier grounding 0.3 + classification certainty 0.1. Trust threshold for indexing: ≥ 0.7.

## D16. Knowledge-growth safety
Only cases that are (a) user-accepted, (b) verifier-grounded, (c) confidence ≥ 0.7, (d) not escalated get indexed; marked `source=live`. Rationale: stops hallucinated answers from compounding into the knowledge base.

---

## Package versions verified on PyPI at planning time
langgraph 1.2.22 · langchain-ollama 1.1.0 · langchain-chroma 1.1.0 · chromadb 1.5.9 · langgraph-checkpoint-sqlite 3.1.3 · langsmith 0.15.0 · openevals 0.2.5 · datasets 5.2.0 · pydantic 2.14.3 · typer 0.29.2 · fastapi 0.163.2 (optional). Final pins come from `uv.lock` after install + `pip-audit`.
