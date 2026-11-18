"""Dataset loading + cleaning + stratified hold-out split."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

TAG_COLS = [f"tag_{i}" for i in range(1, 9)]
_WS = re.compile(r"[ \t]+")
_NL = re.compile(r"\n{3,}")


def clean_text(s: object) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    t = str(s).replace("\\n", "\n").replace("\r", "")
    t = _WS.sub(" ", t)
    return _NL.sub("\n\n", t).strip()


def load_english(csv_paths: Path | list[Path]) -> pd.DataFrame:
    """Load one or more dataset CSVs, keep English only, dedupe on (subject, body)."""
    paths = [csv_paths] if isinstance(csv_paths, Path) else list(csv_paths)
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df = df[df["language"].str.lower() == "en"].copy()
    for col in ("subject", "body", "answer"):
        df[col] = df[col].map(clean_text)
    df = df[(df["body"] != "") & (df["answer"] != "")]
    df = df.drop_duplicates(subset=["subject", "body"]).reset_index(drop=True)
    for col in ("queue", "type", "priority"):
        df[col] = df[col].astype(str).str.strip()
    df["priority"] = df["priority"].str.lower()
    df["tags"] = df[TAG_COLS].apply(
        lambda r: sorted({str(v).strip() for v in r if isinstance(v, str) and v.strip()}), axis=1
    )
    df["ticket_id"] = [f"H-{i + 1:06d}" for i in range(len(df))]
    return df


def assign_holdout(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Stratified (by queue) random hold-out for evaluation; never indexed."""
    df = df.copy()
    df["holdout"] = 0
    if n <= 0:
        return df
    frac = min(1.0, n / len(df))
    idx = df.groupby("queue", group_keys=False).sample(frac=frac, random_state=seed).index[:n]
    df.loc[idx, "holdout"] = 1
    return df


def embed_text(subject: str, body: str, max_chars: int = 1800) -> str:
    return f"{subject}\n{body}"[:max_chars]
