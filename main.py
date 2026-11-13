from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from datetime import date, datetime, timedelta
from typing import Literal, Optional
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field

load_dotenv()
logger = logging.getLogger(__name__)
APP_TIMEZONE = ZoneInfo(os.getenv("VOX_TIMEZONE", "America/Sao_Paulo"))
app = FastAPI(
    title="VOX Interpretation Service",
    version="0.2.0",
    description="Interprets natural-language reminder text for the Agenda API.",
)

DOMAINS = ("saúde", "estudos", "trabalho", "pessoal")
PRIORITIES = ("alta", "média", "baixa")
WEEKDAYS_PT = ("segunda-feira", "terça-feira", "quarta-feira", "quinta-feira", "sexta-feira", "sábado", "domingo")


class InterpretRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


class InterpretResponse(BaseModel):
    title: str
    date: Optional[str] = None
    time: Optional[str] = None
    domain: Literal["saúde", "estudos", "trabalho", "pessoal"] = "pessoal"
    priority: Literal["alta", "média", "baixa"] = "média"


class RuleBasedInterpreter:
    """Produces a predictable interpretation without external services."""

    def interpret(self, text: str) -> InterpretResponse:
        clean_text = text.strip()
        due_date = parse_date(clean_text)
        due_time = parse_time(clean_text)
        return InterpretResponse(
            title=clean_title(clean_text),
            date=due_date.isoformat() if due_date else None,
            time=due_time,
            domain=parse_domain(clean_text),
            priority=parse_priority(clean_text),
        )


class GroqValidator:
    """Validates and improves a rule-based interpretation with Groq.

    The model output is never trusted as-is: each field is checked on its own and,
    when invalid, the rule-based value is kept. Any failure falls back to the candidate.
    """

    def __init__(self) -> None:
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
        try:
            self.timeout = float(os.getenv("GROQ_TIMEOUT_SECONDS", "8"))
            if self.timeout <= 0:
                raise ValueError
        except (TypeError, ValueError):
            logger.warning("Invalid GROQ_TIMEOUT_SECONDS; using 8 seconds")
            self.timeout = 8.0
        self.last_call = "not-called"  # not-called | ok | fallback
        self._client = None
        if self.api_key:
            try:
                import groq  # noqa: F401
            except ImportError:
                logger.error("GROQ_API_KEY is set but the 'groq' package is not installed; Groq disabled")
                self.api_key = None

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _client_or_create(self):
        if self._client is None and self.api_key:
            from groq import Groq

            # Short timeout + one retry: the rule-based result is a good fallback, so do not make users wait.
            self._client = Groq(api_key=self.api_key, timeout=self.timeout, max_retries=1)
        return self._client

    def _system_prompt(self) -> str:
        today = current_local_date()
        return (
            "Você valida lembretes em português do Brasil. "
            "Retorne somente um objeto JSON com as chaves title, date, time, domain e priority. "
            "date deve estar em YYYY-MM-DD ou ser null; time em HH:mm ou null. "
            "domain deve ser saúde, estudos, trabalho ou pessoal. "
            "priority deve ser alta, média ou baixa. "
            f"Hoje é {WEEKDAYS_PT[today.weekday()]}, {today.isoformat()} (fuso {APP_TIMEZONE.key}). "
            "Um dia da semana citado significa a próxima ocorrência futura; 'dia N' significa a próxima vez que esse dia do mês ocorre. "
            "Corrija o candidato somente quando o texto justificar a correção."
        )

    def validate(self, text: str, candidate: InterpretResponse) -> InterpretResponse:
        try:
            client = self._client_or_create()
            if client is None:
                return candidate

            user_prompt = json.dumps(
                {"text": text, "candidate": candidate.model_dump()},
                ensure_ascii=False,
            )
            completion = client.chat.completions.create(
                model=self.model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = completion.choices[0].message.content
            if not content:
                self.last_call = "fallback"
                return candidate
            result = self.merge(candidate, json.loads(content))
            self.last_call = "ok"
            return result
        except Exception:
            self.last_call = "fallback"
            logger.exception("Groq validation failed; using rule-based interpretation")
            return candidate

    @staticmethod
    def merge(candidate: InterpretResponse, raw: object) -> InterpretResponse:
        """Field-by-field sanitisation of the model output; invalid fields keep the candidate value."""
        if not isinstance(raw, dict):
            return candidate

        title = raw.get("title")
        title = title.strip() if isinstance(title, str) and title.strip() else candidate.title

        if "date" in raw and raw["date"] is None:
            due_date = candidate.date
        else:
            due_date = _valid_date(raw.get("date")) or candidate.date

        if "time" in raw and raw["time"] is None:
            due_time = candidate.time
        else:
            due_time = _valid_time(raw.get("time")) or candidate.time

        return InterpretResponse(
            title=title,
            date=due_date,
            time=due_time,
            domain=_match(raw.get("domain"), DOMAINS) or candidate.domain,
            priority=_match(raw.get("priority"), PRIORITIES) or candidate.priority,
        )


class HybridInterpreter:
    """Combines deterministic parsing with optional Groq validation."""

    def __init__(self) -> None:
        self.rule_based = RuleBasedInterpreter()
        self.groq = GroqValidator()

    @property
    def mode(self) -> str:
        return "groq-validation" if self.groq.enabled else "rule-based-fallback"

    def interpret(self, text: str) -> InterpretResponse:
        candidate = self.rule_based.interpret(text)
        return self.groq.validate(text, candidate)


def normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower()


def current_local_date() -> date:
    return datetime.now(APP_TIMEZONE).date()


def parse_date(text: str, today: Optional[date] = None) -> Optional[date]:
    normalized = normalize(text)
    base = today or current_local_date()

    if re.search(r"\bdepois de amanha\b", normalized):
        return base + timedelta(days=2)
    if re.search(r"\bamanha\b", normalized):
        return base + timedelta(days=1)
    if re.search(r"\bhoje\b", normalized):
        return base

    # Explicit Brazilian date: 12/10 or 12/10/2026.
    match = re.search(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b", normalized)
    if match:
        day, month = int(match.group(1)), int(match.group(2))
        year_raw = match.group(3)
        year = int(year_raw) if year_raw else base.year
        if year_raw and len(year_raw) == 2:
            year += 2000
        try:
            parsed = date(year, month, day)
        except ValueError:
            return None
        if not year_raw and parsed < base:
            parsed = date(base.year + 1, month, day)
        return parsed

    # Day number without month: "dia 18".
    match = re.search(r"\bdia\s+(\d{1,2})\b", normalized)
    if match:
        day = int(match.group(1))
        for year, month in ((base.year, base.month), (base.year + 1 if base.month == 12 else base.year, 1 if base.month == 12 else base.month + 1)):
            try:
                parsed = date(year, month, day)
            except ValueError:
                continue
            if parsed >= base:
                return parsed
        return None

    weekdays = {
        "segunda": 0,
        "terca": 1,
        "quarta": 2,
        "quinta": 3,
        "sexta": 4,
        "sabado": 5,
        "domingo": 6,
    }
    weekday_pattern = r"\b(segunda|terca|quarta|quinta|sexta|sabado|domingo)(?:-feira)?\b"
    weekday_match = re.search(weekday_pattern, normalized)
    if weekday_match:
        target_weekday = weekdays[weekday_match.group(1)]
        delta = (target_weekday - base.weekday()) % 7
        # Treat a named weekday as the next upcoming occurrence, not earlier today.
        if delta == 0 or re.search(r"\b(proxima|proximo)\s+" + weekday_pattern, normalized):
            delta = 7
        return base + timedelta(days=delta)

    return None


def parse_time(text: str) -> Optional[str]:
    normalized = normalize(text)
    patterns = (
        # The first pattern accepts "às 18", "às 18h30" and "às 18 horas",
        # but does not mistake phrases such as "as 3 matérias" for a time.
        r"\bas\s*(\d{1,2})(?::(\d{2})|h(\d{2})?|(?=\s+horas?\b)|(?=$|[.,;!?]))",
        r"\b(\d{1,2}):(\d{2})\b",
        r"\b(\d{1,2})h(\d{2})?\b",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if not match:
            continue
        hour = int(match.group(1))
        minute_text = match.group(2) or match.group(3) or "0"
        minute = int(minute_text)
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return f"{hour:02d}:{minute:02d}"
    return None


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
        r"\b(?:na|nesta|proxima|proximo)?\s*(?:segunda|terça|terca|quarta|quinta|sexta|sábado|sabado|domingo)(?:-feira)?\b",
        r"\b(?:dia\s+)?\d{1,2}/\d{1,2}(?:/\d{2,4})?\b",
        r"\bdia\s+\d{1,2}\b",
        r"\b(?:às|as)\s*\d{1,2}(?::\d{2}|h\d{0,2})?(?:\s*horas?)?\b",
        r"\b\d{1,2}:\d{2}\b",
        r"\b\d{1,2}h\d{0,2}\b",
        r"\b(?:por favor|com urgência|urgente|prioridade alta|prioridade baixa)\b",
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


_TIME_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


def _valid_date(value: object) -> Optional[str]:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value.strip()).isoformat()
    except ValueError:
        return None


def _valid_time(value: object) -> Optional[str]:
    if isinstance(value, str) and _TIME_RE.match(value.strip()):
        return value.strip()
    return None


def _match(value: object, allowed: tuple[str, ...]) -> Optional[str]:
    """Accepts 'Saude'/'saúde'/' SAÚDE ' and returns the canonical accented value."""
    if not isinstance(value, str):
        return None
    wanted = normalize(value.strip())
    for option in allowed:
        if normalize(option) == wanted:
            return option
    return None


def interpret_text(text: str) -> InterpretResponse:
    return INTERPRETER.interpret(text)


INTERPRETER = HybridInterpreter()


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "vox-ai-service",
        "mode": INTERPRETER.mode,
        "groq_last_call": INTERPRETER.groq.last_call,
    }


@app.post("/interpret", response_model=InterpretResponse)
def interpret(request: InterpretRequest) -> InterpretResponse:
    return interpret_text(request.text)
