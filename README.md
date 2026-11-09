# VOX AI Service — parser MVP

Serviço Python que recebe somente texto e devolve os campos estruturados esperados pela `agenda-api`.

> O serviço usa o parser determinístico como base e, quando `GROQ_API_KEY` está configurada, envia o candidato ao Groq para validação e correção estruturada. Sem essa chave, funciona automaticamente em modo fallback, sem chamada externa.

> O pacote `tzdata` é uma dependência explícita porque o Python no Windows normalmente não inclui a base IANA de fusos horários usada por `ZoneInfo("America/Sao_Paulo")`.

## Executar

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Para habilitar a validação pelo Groq, preencha o arquivo `.env` na raiz do projeto:

```env
GROQ_API_KEY=sua-chave-do-groq
GROQ_MODEL=openai/gpt-oss-20b
```

O arquivo `.env` é carregado automaticamente e não deve ser commitado.

O endpoint `GET /health` informa `groq-validation` quando o Groq está habilitado e `rule-based-fallback` caso contrário.

## Docker

```bash
docker build -t vox-ai-service .
docker run --rm -p 8000:8000 vox-ai-service
```

- `GET /health`
- `POST /interpret` com JSON `{ "text": "Me lembre de estudar amanhã às 19h" }`

Formato de saída: `title`, `date` (`YYYY-MM-DD` ou `null`), `time` (`HH:mm` ou `null`), `domain` e `priority`.

## Testes do parser

```bash
python -m unittest -v
```
