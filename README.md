# VOX AI Service — parser MVP

Serviço Python que recebe somente texto e devolve os campos estruturados esperados pela `agenda-api`.

> Esta versão é intencionalmente um parser determinístico para validar a integração, não um modelo de IA. Substitua/expanda `interpret_text()` para conectar o provedor de IA escolhido antes de avaliar a meta de precisão de 90% descrita no plano.

> O pacote `tzdata` é uma dependência explícita porque o Python no Windows normalmente não inclui a base IANA de fusos horários usada por `ZoneInfo("America/Sao_Paulo")`.

## Executar

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

- `GET /health`
- `POST /interpret` com JSON `{ "text": "Me lembre de estudar amanhã às 19h" }`

Formato de saída: `title`, `date` (`YYYY-MM-DD` ou `null`), `time` (`HH:mm` ou `null`), `domain` e `priority`.

## Testes do parser

```bash
python -m unittest -v
```
