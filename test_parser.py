import unittest
from datetime import date

from fastapi import HTTPException

from main import clean_title, interpret_text, parse_date, parse_domain, parse_priority, parse_time


class VoxParserTests(unittest.TestCase):
    def test_tomorrow(self):
        self.assertEqual(parse_date("Me lembre de estudar amanhã", date(2026, 10, 9)), date(2026, 10, 10))

    def test_date_and_time(self):
        text = "Me lembre de entregar o trabalho dia 12/10 às 18h30"
        self.assertEqual(parse_date(text, date(2026, 10, 9)), date(2026, 10, 12))
        self.assertEqual(parse_time(text), "18:30")

    def test_domain_and_priority(self):
        text = "Urgente: lembrar da consulta médica amanhã às 9h"
        self.assertEqual(parse_domain(text), "saúde")
        self.assertEqual(parse_priority(text), "alta")
        self.assertEqual(parse_time(text), "09:00")

    def test_does_not_treat_article_as_time(self):
        self.assertIsNone(parse_time("Preciso revisar as 3 matérias da prova"))

    def test_title_cleanup(self):
        self.assertEqual(clean_title("Me lembre de entregar o trabalho amanhã às 18h."), "entregar o trabalho")

    def test_title_cleanup_with_hour_word(self):
        self.assertEqual(
            clean_title("Me lembre de entregar o trabalho amanhã às 18 horas"),
            "entregar o trabalho",
        )

    def test_urgent_medical_reminder_title(self):
        text = "Urgente: agendar consulta médica dia 12/10 às 09:30"
        self.assertEqual(clean_title(text), "consulta médica")
        self.assertEqual(parse_domain(text), "saúde")
        self.assertEqual(parse_priority(text), "alta")

    def test_relative_date_and_study_domain(self):
        text = "Me lembre de estudar para a prova amanhã às 19h"
        self.assertEqual(parse_date(text, date(2026, 10, 9)), date(2026, 10, 10))
        self.assertEqual(parse_time(text), "19:00")
        self.assertEqual(parse_domain(text), "estudos")
        self.assertEqual(clean_title(text), "estudar para a prova")

    def test_next_weekday_always_lands_on_requested_weekday(self):
        # 2026-11-22 is Sunday; next Monday is 2026-11-23, not Sunday + 7 days.
        self.assertEqual(parse_date("na próxima segunda", date(2026, 11, 22)), date(2026, 11, 23))
        # If the requested weekday is today, use the next week's occurrence.
        self.assertEqual(parse_date("segunda-feira", date(2026, 11, 23)), date(2026, 11, 30))
        self.assertEqual(parse_date("próxima sexta", date(2026, 11, 24)), date(2026, 11, 27))

    def test_written_month_names_and_clean_title(self):
        self.assertEqual(parse_date("pagar aluguel dia 5 de janeiro", date(2026, 11, 23)), date(2027, 1, 5))
        self.assertEqual(parse_date("consulta em 12 de outubro de 2027", date(2026, 11, 23)), date(2027, 10, 12))
        self.assertEqual(clean_title("Me lembre de pagar aluguel dia 5 de janeiro"), "pagar aluguel")

    def test_period_of_day_conversion(self):
        self.assertEqual(parse_time("tomar remédio às 9 da noite amanhã"), "21:00")
        self.assertEqual(parse_time("tomar remédio às 3 da tarde"), "15:00")
        self.assertEqual(parse_time("tomar remédio às 9 da manhã"), "09:00")
        self.assertEqual(parse_time("evento à meia-noite"), "00:00")
        self.assertEqual(parse_time("evento ao meio-dia"), "12:00")
        self.assertEqual(clean_title("Me lembre de tomar remédio às 9 da noite amanhã"), "tomar remédio")

    def test_explicit_invalid_date_or_time_is_rejected(self):
        with self.assertRaises(HTTPException) as invalid_date:
            interpret_text("Me lembre de ir ao médico dia 31/11 às 9h")
        self.assertEqual(invalid_date.exception.status_code, 422)
        with self.assertRaises(HTTPException) as invalid_time:
            interpret_text("Me lembre de ir ao médico amanhã às 25h")
        self.assertEqual(invalid_time.exception.status_code, 422)


if __name__ == "__main__":
    unittest.main()
