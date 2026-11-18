-- Application database (SQLite, WAL)
CREATE TABLE IF NOT EXISTS tickets (
  ticket_id  TEXT PRIMARY KEY,
  source     TEXT NOT NULL CHECK (source IN ('hist','live')),
  subject    TEXT, body TEXT, answer TEXT,
  queue TEXT, type TEXT, priority TEXT, language TEXT,
  tags_json  TEXT NOT NULL DEFAULT '[]',
  holdout    INTEGER NOT NULL DEFAULT 0,
  indexed    INTEGER NOT NULL DEFAULT 0,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_tickets_cls ON tickets(queue, type, priority);

CREATE VIRTUAL TABLE IF NOT EXISTS tickets_fts USING fts5(
  ticket_id UNINDEXED, subject, body, tags, tokenize='porter unicode61'
);

CREATE TABLE IF NOT EXISTS customers (
  customer_id TEXT PRIMARY KEY, created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS cases (
  case_id TEXT PRIMARY KEY,
  customer_id TEXT NOT NULL,
  thread_id TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN
    ('OPEN','AWAITING_USER','PROPOSED','RESOLVED','ESCALATED','INDEXED')),
  subject TEXT, body TEXT,
  queue TEXT, type TEXT, priority TEXT, tags_json TEXT DEFAULT '[]',
  resolution TEXT, analysis TEXT, confidence REAL,
  escalation_required INTEGER DEFAULT 0, escalation_reason TEXT, escalation_target TEXT,
  evidence_json TEXT DEFAULT '[]',
  accepted INTEGER, indexed INTEGER DEFAULT 0,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_cases_customer ON cases(customer_id, created_at);
CREATE INDEX IF NOT EXISTS idx_cases_thread ON cases(thread_id);

CREATE TABLE IF NOT EXISTS case_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  case_id TEXT NOT NULL, ts TEXT DEFAULT CURRENT_TIMESTAMP,
  kind TEXT NOT NULL, payload_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_case ON case_events(case_id);

CREATE TABLE IF NOT EXISTS customer_memory (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  customer_id TEXT NOT NULL,
  kind TEXT NOT NULL CHECK (kind IN ('environment','pattern','preference','caveat','outcome')),
  text TEXT NOT NULL,
  source_case_id TEXT, confidence REAL DEFAULT 0.8,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  last_used_at TEXT DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(customer_id, text)
);
CREATE INDEX IF NOT EXISTS idx_mem_customer ON customer_memory(customer_id);
