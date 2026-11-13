import json
import os
import unittest
from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

from main import GroqValidator, InterpretResponse, RuleBasedInterpreter


def fake_client(content=None, error=None):
    def create(**kwargs):
        if error:
            raise error
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


class GroqMergeTests(unittest.TestCase):
    def setUp(self):
        self.candidate = InterpretResponse(
            title="estudar para a prova", date="2026-11-10", time="19:00", domain="estudos", priority="média"
        )

    def validator(self, client):
        validator = GroqValidator()
        validator.api_key = "test-key"
        validator._client = client
        return validator

    def test_valid_correction_is_applied(self):
        raw = {"title": "Estudar para a prova", "date": "2026-11-11", "time": "20:30", "domain": "Estudos", "priority": "ALTA"}
        result = self.validator(fake_client(json.dumps(raw))).validate("texto", self.candidate)
        self.assertEqual((result.date, result.time, result.priority), ("2026-11-11", "20:30", "alta"))
        self.assertEqual(result.title, "Estudar para a prova")

    def test_invalid_fields_keep_candidate_values(self):
        raw = {"title": "  ", "date": "amanhã", "time": "25:99", "domain": "financeiro", "priority": None}
        result = self.validator(fake_client(json.dumps(raw))).validate("texto", self.candidate)
        self.assertEqual(result, self.candidate)

    def test_explicit_null_keeps_rule_based_date_and_time(self):
        raw = {"title": "estudar", "date": None, "time": None}
        result = self.validator(fake_client(json.dumps(raw))).validate("texto", self.candidate)
        self.assertEqual((result.date, result.time), (self.candidate.date, self.candidate.time))

    def test_explicit_null_stays_null_when_candidate_has_no_schedule(self):
        candidate = self.candidate.model_copy(update={"date": None, "time": None})
        raw = {"title": "estudar", "date": None, "time": None}
        result = self.validator(fake_client(json.dumps(raw))).validate("texto", candidate)
        self.assertIsNone(result.date)
        self.assertIsNone(result.time)

    def test_invalid_timeout_uses_default(self):
        with patch.dict(os.environ, {"GROQ_TIMEOUT_SECONDS": "invalid"}):
            validator = GroqValidator()
        self.assertEqual(validator.timeout, 8.0)

    def test_accent_insensitive_domain_and_priority(self):
        raw = {"domain": "saude", "priority": "media"}
        result = self.validator(fake_client(json.dumps(raw))).validate("texto", self.candidate)
        self.assertEqual((result.domain, result.priority), ("saúde", "média"))

    def test_garbage_empty_and_errors_fall_back(self):
        for client in (fake_client("não é json"), fake_client("[1, 2]"), fake_client(""), fake_client(error=TimeoutError())):
            validator = self.validator(client)
            with patch("main.logger"):
                result = validator.validate("texto", self.candidate)
            self.assertEqual(result, self.candidate)

    def test_without_api_key_returns_candidate_without_calling_groq(self):
        validator = GroqValidator()
        validator.api_key = None
        validator._client = None
        self.assertFalse(validator.enabled)
        self.assertEqual(validator.validate("texto", self.candidate), self.candidate)

    def test_rule_based_still_matches_response_contract(self):
        result = RuleBasedInterpreter().interpret("Urgente: agendar consulta médica dia 12/10 às 09:30")
        self.assertEqual((result.title, result.time, result.domain, result.priority), ("consulta médica", "09:30", "saúde", "alta"))


if __name__ == "__main__":
    unittest.main()
