# ⚽ FootballHub — Backend Python (Fase 1)

Probabili formazioni, calciomercato, previsioni ML, tutto real-time.

---

## Struttura

```
football-hub/
├── main.py                  # FastAPI app + tutti gli endpoint REST + WebSocket
├── config.py                # Configurazione da .env
├── tasks.py                 # Celery task schedulati
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
│
├── db/
│   ├── models.py            # Tutte le tabelle SQLAlchemy
│   └── database.py          # Engine, sessioni, init
│
├── scraper/
│   ├── football_data_client.py   # football-data.org (storico + live)
│   ├── api_football_client.py    # API-Football (formazioni ufficiali + stats)
│   ├── odds_client.py            # The Odds API (quote 20+ bookmaker)
│   ├── lineup_scraper.py         # Scraping Gazzetta + Fantacalcio
│   └── transfer_scraper.py       # Sky Sport, TMW, Romano (Fase 2)
│
├── ml/
│   └── predictor.py              # Modello XGBoost + Poisson (Fase 2)
│
└── nlp/
    └── transfer_analyzer.py      # NLP mercato (Fase 2)
```

---

## Avvio rapido

### 1. Configura le API key

```bash
cp .env.example .env
# Apri .env e inserisci le tue chiavi
```

Registrati su:
- **football-data.org** → gratis, chiave immediata
- **rapidapi.com** → cerca "API-Football", piano free 100 req/giorno
- **the-odds-api.com** → gratis, 500 req/mese

### 2. Avvia con Docker (consigliato)

```bash
docker-compose up -d
```

Questo avvia: PostgreSQL, Redis, FastAPI, Celery worker, Celery beat scheduler, Flower monitor.

### 3. Senza Docker (sviluppo locale)

```bash
# Crea venv
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Installa dipendenze
pip install -r requirements.txt
playwright install chromium
python -m spacy download it_core_news_lg

# Avvia PostgreSQL e Redis (con Docker solo per i DB)
docker-compose up -d postgres redis

# Crea il database
python -c "from db.database import init_db; init_db()"

# Avvia API
uvicorn main:app --reload --port 8000

# In un altro terminale: avvia Celery worker
celery -A tasks worker --loglevel=info

# In un altro terminale: avvia scheduler
celery -A tasks beat --loglevel=info
```

### 4. Import dati storici (solo prima volta)

```bash
# Avvia l'import di 3 stagioni di Serie A, Champions, Premier
curl -X POST "http://localhost:8000/api/admin/import?secret=change-this-in-production&competition=SA"
```

Oppure direttamente:
```bash
celery -A tasks call tasks.initial_data_import
```

---

## Endpoint principali

| Metodo | URL | Descrizione |
|--------|-----|-------------|
| GET | `/health` | Stato backend |
| GET | `/api/matches` | Lista partite (filtri: date, status, competition) |
| GET | `/api/matches/today` | Partite di oggi |
| GET | `/api/matches/{id}` | Dettaglio partita |
| GET | `/api/matches/{id}/lineups` | Formazioni (ufficiali o probabili) |
| GET | `/api/matches/{id}/prediction` | Previsione ML |
| GET | `/api/transfers` | Rumors mercato |
| GET | `/api/injuries` | Infortuni attivi |
| WS  | `/ws` | WebSocket per aggiornamenti real-time |

### WebSocket — messaggi ricevuti dal server

```json
{ "type": "live_update",      "matches": ["ext_id_1", ...] }
{ "type": "official_lineups", "match_ids": [42, 43] }
{ "type": "new_transfer",     "transfer_id": 7 }
```

---

## Collegamento al frontend HTML

Apri `football_hub.html` e nella sezione Setup inserisci:
```
URL Backend: http://localhost:8000
```

Il frontend si connetterà automaticamente al WebSocket per gli aggiornamenti live.

---

## Costi API stimati (uso normale Serie A)

| Fonte | Piano | Costo | Richieste |
|-------|-------|-------|-----------|
| football-data.org | Free | €0 | 10 req/min |
| API-Football | Free | €0 | 100 req/giorno |
| The Odds API | Free | €0 | 500 req/mese |
| **Totale test** | | **€0** | |
| API-Football Pro | Pro | ~€10/mese | illimitato |

---

## Fase 3 — Completata

- `chatbot/agent.py` — agente LangChain con 8 tools sul DB live
- `api/chat_router.py` — endpoint REST `/api/chat/` + streaming SSE
- `frontend/football_hub.html` — frontend completamente connesso al backend

## Avvio completo (tutte e 3 le fasi)

```bash
# 1. Avvia i servizi
docker-compose up -d

# 2. Import dati storici (solo prima volta)
curl -X POST "http://localhost:8000/api/admin/import?secret=YOUR_SECRET&competition=SA"

# 3. Addestra il modello ML
python ml/train.py

# 4. Apri il frontend
open frontend/football_hub.html
# → Vai in Setup, inserisci http://localhost:8000, clicca "Testa connessione"
```
