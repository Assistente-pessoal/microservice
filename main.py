from __future__ import annotations

import os
import re
import unicodedata
from datetime import date, datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

APP_TIMEZONE = ZoneInfo(os.getenv("VOX_TIMEZONE", "America/Sao_Paulo"))
app = FastAPI(
    title="VOX Interpretation Service",
    version="0.2.0",
    description="Interprets natural-language reminder text for the Agenda API.",
)


class InterpretRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


class InterpretResponse(BaseModel):
    title: str
    date: Optional[str] = None
    time: Optional[str] = None
    domain: str = "pessoal"
    priority: str = "média"


def normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower()


def current_local_date() -> date:
    return datetime.now(APP_TIMEZONE).date()


MONTHS = {
    "janeiro": 1, "fevereiro": 2, "marco": 3, "abril": 4,
    "maio": 5, "junho": 6, "julho": 7, "agosto": 8,
    "setembro": 9, "outubro": 10, "novembro": 11, "dezembro": 12,
}
WEEKDAYS = {
    "segunda": 0, "terca": 1, "quarta": 2, "quinta": 3,
    "sexta": 4, "sabado": 5, "domingo": 6,
}
WEEKDAY_PATTERN = r"(?:segunda|terca|quarta|quinta|sexta|sabado|domingo)(?:-feira)?"
MONTH_PATTERN = "|".join(MONTHS)
MONTH_DATE_RE = re.compile(
    rf"\b(?:dia\s+)?(\d{{1,2}})\s+de\s+({MONTH_PATTERN})(?:\s+de\s+(\d{{4}}))?\b"
)
NUMERIC_DATE_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b")
DAY_NUMBER_RE = re.compile(r"\bdia\s+(\d{1,2})\b")


def _year_for_month_day(month: int, day: int, base: date, explicit_year: Optional[int]) -> Optional[date]:
    years = [explicit_year] if explicit_year is not None else [base.year, base.year + 1]
    for year in years:
        try:
            candidate = date(year, month, day)
        except ValueError:
            continue
        if explicit_year is not None or candidate >= base:
            return candidate
    return None


def parse_date(text: str, today: Optional[date] = None) -> Optional[date]:
    normalized = normalize(text)
    base = today or current_local_date()

    if re.search(r"\bdepois de amanha\b", normalized):
        return base + timedelta(days=2)
    if re.search(r"\bamanha\b", normalized):
        return base + timedelta(days=1)
    if re.search(r"\bhoje\b", normalized):
        return base

    month_match = MONTH_DATE_RE.search(normalized)
    if month_match:
        day = int(month_match.group(1))
        month = MONTHS[month_match.group(2)]
        year_text = month_match.group(3)
        return _year_for_month_day(month, day, base, int(year_text) if year_text else None)

    numeric_match = NUMERIC_DATE_RE.search(normalized)
    if numeric_match:
        day, month = int(numeric_match.group(1)), int(numeric_match.group(2))
        year_text = numeric_match.group(3)
        year = int(year_text) if year_text else None
        if year_text and len(year_text) == 2:
            year += 2000
        return _year_for_month_day(month, day, base, year)

    day_match = DAY_NUMBER_RE.search(normalized)
    if day_match:
        day = int(day_match.group(1))
        candidates = ((base.year, base.month), (base.year + (base.month == 12), 1 if base.month == 12 else base.month + 1))
        for year, month in candidates:
            try:
                candidate = date(year, month, day)
            except ValueError:
                continue
            if candidate >= base:
                return candidate
        return None

    weekday_match = re.search(rf"\b({WEEKDAY_PATTERN})\b", normalized)
    if weekday_match:
        weekday_name = weekday_match.group(1).split("-")[0]
        target_weekday = WEEKDAYS[weekday_name]
        delta = (target_weekday - base.weekday()) % 7
        # A named weekday means its next occurrence; if it is today, use next week.
        # "próxima segunda" follows the same rule instead of incorrectly forcing +7 days.
        if delta == 0:
            delta = 7
        return base + timedelta(days=delta)

    return None


TIME_RE = re.compile(
    r"\b(?:as\s*)?(\d{1,2})(?::(\d{2})|h(\d{2})?|(?=\s+horas?\b)|"
    r"(?=\s+(?:da|de)\s+(?:manha|tarde|noite)\b)|(?=$|[.,;!?]))"
    r"(?:\s+(?:da|de)\s+(manha|tarde|noite))?\b"
)


def parse_time(text: str) -> Optional[str]:
    normalized = normalize(text)
    if re.search(r"\bmeia[- ]noite\b", normalized):
        return "00:00"
    if re.search(r"\bmeio[- ]dia\b", normalized):
        return "12:00"

    match = TIME_RE.search(normalized)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2) or match.group(3) or "0")
    period = match.group(4)

    if period == "manha":
        if hour == 12:
            hour = 0
        elif not 1 <= hour <= 11:
            return None
    elif period == "tarde":
        if 1 <= hour <= 11:
            hour += 12
        elif not 12 <= hour <= 23:
            return None
    elif period == "noite":
        if hour == 12:
            hour = 0
        elif 1 <= hour <= 11:
            hour += 12
        elif not 0 <= hour <= 23:
            return None

    if 0 <= hour <= 23 and 0 <= minute <= 59:
        return f"{hour:02d}:{minute:02d}"
    return None


def _has_explicit_date(text: str) -> bool:
    normalized = normalize(text)
    return bool(NUMERIC_DATE_RE.search(normalized) or MONTH_DATE_RE.search(normalized) or DAY_NUMBER_RE.search(normalized))


def _has_explicit_time(text: str) -> bool:
    normalized = normalize(text)
    return bool(
        re.search(r"\b\d{1,2}:\d{2}\b", normalized)
        or re.search(r"\b\d{1,2}h\d{0,2}\b", normalized)
        or re.search(r"\b(?:as\s*)\d{1,2}(?=\s*(?::|h)|\s+horas?\b|\s+(?:da|de)\s+(?:manha|tarde|noite)\b|[.,;!?]|$)", normalized)
        or re.search(r"\b(?:meia[- ]noite|meio[- ]dia)\b", normalized)
    )


def parse_domain(text: str) -> str:
    normalized = normalize(text)
    if any(word in normalized for word in ("consulta", "medico", "medica", "remedio", "exame", "dentista", "saude")):
        return "saúde"
    if any(word in normalized for word in ("prova", "estudar", "faculdade", "aula", "atividade", "trabalho da faculdade", "professor")):
        return "estudos"
    if any(word in normalized for word in ("reuniao", "cliente", "chefe", "empresa", "trabalho", "relatorio", "apresentacao")):
        return "trabalho"
    return "pessoal"


def parse_priority(text: str) -> str:
    normalized = normalize(text)
    if any(word in normalized for word in ("urgente", "prioridade alta", "muito importante", "nao posso esquecer", "o mais rapido possivel")):
        return "alta"
    if any(word in normalized for word in ("sem pressa", "quando puder", "prioridade baixa", "nao e urgente")):
        return "baixa"
    return "média"


def clean_title(text: str) -> str:
    title = text.strip()
    clean_patterns = (
        r"\bdepois de amanh[ãa]\b",
        r"\bamanh[ãa]\b",
        r"\bhoje\b",
        rf"\b(?:na\s+)?(?:nesta\s+)?(?:proxima\s+|proximo\s+)?{WEEKDAY_PATTERN}\b",
        rf"\b(?:dia\s+)?\d{{1,2}}\s+de\s+(?:{MONTH_PATTERN})(?:\s+de\s+\d{{4}})?\b",
        r"\b(?:dia\s+)?\d{1,2}/\d{1,2}(?:/\d{2,4})?\b",
        r"\bdia\s+\d{1,2}\b",
        r"\b(?:às|as)\s*\d{1,2}(?::\d{2}|h\d{0,2})?(?:\s*horas?)?(?:\s+(?:da|de)\s+(?:manhã|manha|tarde|noite))?\b",
        r"\b\d{1,2}:\d{2}\b",
        r"\b\d{1,2}h\d{0,2}(?:\s+(?:da|de)\s+(?:manhã|manha|tarde|noite))?\b",
        r"\b(?:meia[- ]noite|meio[- ]dia)\b",
        r"\b(?:por favor|com urgência|com urgencia|urgente|prioridade alta|prioridade baixa)\b",
    )
    for pattern in clean_patterns:
        title = re.sub(pattern, " ", title, flags=re.IGNORECASE)

    title = re.sub(r"^\s*[:\-–]+\s*", "", title)
    title = re.sub(
        r"^\s*(?:por favor,?\s*)?(?:me\s+)?(?:lembre|lembra)(?:-me)?\s+(?:de\s+)?",
        "",
        title,
        flags=re.IGNORECASE,
    )
    title = re.sub(
        r"^\s*(?:por favor,?\s*)?(?:agendar|agende|colocar na agenda|adicionar na agenda)\s+",
        "",
        title,
        flags=re.IGNORECASE,
    )
    title = re.sub(r"^\s*(?:nao esquecer de|lembrar de|preciso|tenho que)\s+", "", title, flags=re.IGNORECASE)
    title = re.sub(r"^\s*[:\-–]+\s*", "", title)
    title = re.sub(r"\s+", " ", title).strip(" .,!;:-")
    return title or text.strip()


def interpret_text(text: str) -> InterpretResponse:
    clean_text = text.strip()
    due_date = parse_date(clean_text)
    due_time = parse_time(clean_text)
    if _has_explicit_date(clean_text) and due_date is None:
        raise HTTPException(status_code=422, detail="A data informada não é válida ou não pôde ser interpretada. Revise o lembrete.")
    if _has_explicit_time(clean_text) and due_time is None:
        raise HTTPException(status_code=422, detail="O horário informado não é válido ou não pôde ser interpretado. Revise o lembrete.")
    title = clean_title(clean_text)
    if not title:
        raise HTTPException(status_code=422, detail="Não foi possível identificar o texto do lembrete. Revise a frase.")
    return InterpretResponse(
        title=title,
        date=due_date.isoformat() if due_date else None,
        time=due_time,
        domain=parse_domain(clean_text),
        priority=parse_priority(clean_text),
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "vox-ai-service", "mode": "rule-based-mvp"}


@app.post("/interpret", response_model=InterpretResponse)
def interpret(request: InterpretRequest) -> InterpretResponse:
    return interpret_text(request.text)
