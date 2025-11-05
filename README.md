# FYND AI — Security Intelligence Chatbot & Data Pipeline
**Unified Open Data Engine + AI Chat Interface for Real-Time Safety Insights**

**Author:** [Anthony Ikenna Ogbonna (@OGBmetrix)](https://github.com/OGBmetrix)  
**Core Language:** Python  
**Data Format:** Parquet (Optimized Analytics Dataset)

---

## Overview

**FYND AI** is a modular **Security Intelligence Data Pipeline + AI Chatbot** designed to unify and reason over multiple public safety datasets.

The system merges *Toronto Open Data* (crime, traffic, police operations, budget) with live feeds from **emergency alerts**, **news APIs**, **urban safety data**, and **StatsCan demographics** — all powered by an AI layer capable of intelligent, conversational responses.

It serves as the foundation for a **city-scale safety assistant**, enabling:
- Real-time public safety insights
- Dynamic AI-driven analytics
- Geolocation-based crime mapping
- Emergency alert monitoring
- Contextual news summarization
- Human-like security reasoning

---

## Core Objective

To merge, enrich, and reason over diverse security data streams — producing an analytics-ready master dataset and an AI chatbot that can answer, *“What’s happening near me?”* with clarity and confidence.

### FYND AI Powers:
✅ Real-time security assessments  
✅ Threat pattern discovery  
✅ Resource and budget efficiency analysis  
✅ Live updates from official APIs  
✅ Chat-based reasoning using FAISS + LLMs

---

## Data Sources Integrated

| Layer | Example Datasets | Focus |
|:------|:----------------|:------|
| **Traffic & Collisions** | CYCLIST_KSI, PEDESTRIAN_KSI, FATALS_KSI, TRAFFIC_COLLISIONS | Collision density, KSI analysis |
| **Crimes & Incidents** | Assault, Robbery, Theft Over, Break & Enter, MCI | Violent & property crime trends |
| **Police & Operations** | Patrol Zones, Arrested & Charged Persons, Dispatch Calls | Response efficiency & presence index |
| **Budget & Resources** | Gross Operating Budget, Expenditures, Personnel by Rank | Fiscal transparency & manpower metrics |
| **Urban Safety APIs** | City of Toronto Wellbeing datasets | Community resilience indicators |
| **Alerts & Weather** | Alert Ready, Environment Canada Feeds | Disaster, weather, or emergency updates |
| **News & Media** | GNews / Mediastack / NewsData | Real-time crime & safety headlines |
| **Census & Demographics** | StatsCan Web Data Service | Population, income, and risk correlations |

---

## Pipeline Features

- **Automatic Schema Detection** — Column name aliasing via `schema_map.yaml`
- **Smart File Classification** — Auto-routes datasets into *traffic*, *crime*, *operations*, *budget*
- **Unified Spatial Model** — Standardizes data under `location_name`, `year`, `latitude`, `longitude`
- **Weighted Security Index** — Normalized `security_score` (0–1) combining crime, collisions & police presence
- **FAISS Knowledge Base (RAG)** — Semantic reasoning layer for pattern-based Q&A
- **Audit Reporting** — Generates summary logs and missing-value statistics
- **AI Intent Parsing** — Understands user intent, asks clarifying questions, returns confidence scores

---

## Data Output

The master dataset `security_master.parquet` (and `security_master_geo.parquet`) includes unified indicators such as:

| Column | Description |
|:--------|:-------------|
| `location_name` | Division or community name |
| `year` | Year of record |
| `violent_crime_count` | Aggregated assault/robbery metrics |
| `property_crime_count` | Theft and break & enter data |
| `total_crime_count` | Combined total |
| `crime_rate_per_1000` | Population-adjusted rate |
| `ksi_total` | Killed or Seriously Injured count |
| `total_collisions` | All recorded collisions |
| `police_presence_index` | Dispatch + arrests indicator |
| `gross_budget` | Total division-level budget |
| `total_personnel` | Staff count |
| `security_score` | Weighted public safety indicator |

---

## System Architecture

```text
        ┌─────────────────────────────┐
        │        Streamlit UI         │
        │     (chatbot_ui.py)         │
        └───────────┬─────────────────┘
                    │
                    ▼
     ┌──────────────┴──────────────────────┐
     │  FYND AI Chatbot Orchestrator       │
     │  ── intent_parser + query_knowledge │
     └───────┬────────┬────────┬──────────┘
             │        │        │
             ▼        ▼        ▼
     live_data   alert_fetcher  news_fetcher
     urban_safety_fetcher  census_fetcher
             │        │        │
             ▼        ▼        ▼
         FAISS Vector DB + Parquet Master Dataset

Repository Structure
security_data_pipeline/
├── data/
│   ├── raw/                      # Input CSVs
│   ├── security_master.parquet   # Core unified dataset
│   ├── security_master_geo.parquet  # Enriched dataset
│   └── knowledge/                # RAG training notes
├── scripts/
│   ├── data_pipeline.py          # Core ETL
│   ├── validate_count.py         # Data validation
│   ├── validate_schema.py        # Schema alias checks
│   ├── reverse_geocode.py        # Optional geocoding
│   ├── division_map.py           # D11–D55 mapping
│   ├── train_knowledge_base.py   # FAISS index builder
│   ├── query_knowledge.py        # Knowledge search
│   ├── live_data_fetcher.py      # Toronto Police API
│   ├── urban_safety_fetcher.py   # Community APIs
│   ├── alert_fetcher.py          # Alerts / RSS feeds
│   ├── news_fetcher.py           # News & media updates
│   ├── census_fetcher.py         # StatsCan demographics
│   ├── intent_parser.py          # AI-based query classifier
│   └── chatbot_ui.py             # Streamlit chatbot frontend
├── config/
│   └── schema_map.yaml           # Alias map & category keywords
└── reports/
    ├── audit_summary.md          # Data integrity summary
    └── README.md                 # Local documentation

Tech Stack
**Component**	          **Description**
Python	            Core data pipeline & AI integration
Streamlit	          Interactive UI framework
Pandas / NumPy	    Data wrangling & analytics
PyYAML	            Schema configuration
FAISS	              Vector similarity search (RAG knowledge layer)
OpenAI API	        Intent parsing & summarization
Plotly	            Visual analytics and charts
Geopy	              Reverse geocoding (optional)
Parquet (PyArrow)	  Optimized output storage
Requests / RSS	    External API connectors

AI Chatbot Intelligence

Intent Detection: Determines what the user is asking (“crime trends,” “alerts,” “budget data”).

Clarification Engine: Asks guiding questions when uncertain.

Knowledge Search (FAISS): Retrieves relevant facts from CSV/API text embeddings.

Confidence Awareness: Reports answer confidence and shows fallback messages like:

“FYND AI is working tirelessly to improve our datasets.”

Multi-API Fusion: Combines static data + live feeds for dynamic responses.

 How to Run
# 1. Activate your environment
cd security_data_pipeline
python -m venv venv
venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Build your unified dataset
python scripts/data_pipeline.py --raw_dir data/raw --out_dir data --config config/schema_map.yaml

# 4. Train the knowledge base
python scripts/train_knowledge_base.py

# 5. Launch chatbot
streamlit run scripts/chatbot_ui.py

Error Handling Highlights
| Issue                      | Fix                                                                      |
| :------------------------- | :----------------------------------------------------------------------- |
| **Missing FAISS index**    | Run `train_knowledge_base.py` to generate `.faiss` and `.pkl` files      |
| **CKAN 404 (Toronto API)** | Update `resource_id` in `live_data_fetcher.py`                           |
| **API timeouts**           | Cached fallback enabled; adjust `timeout` in fetchers                    |
| **OpenAI Key Missing**     | Set `OPENAI_API_KEY` via environment variable                            |
| **Path errors (Windows)**  | Always run from project root (`FYND_AI_ChatBot/security_data_pipeline/`) |

 Future Roadmap
| Sprint | Goal                       | Example                          |
| :----- | :------------------------- | :------------------------------- |
| 2.1    | Add geolocation reasoning  | “Show crimes near Queen Street.” |
| 2.2    | Add trend visualization    | Time-based comparative charts    |
| 2.3    | Integrate StatsCan API     | Real census demographics         |
| 2.4    | Restore Budget analytics   | Analyze resource distribution    |
| 2.5    | Build Whistleblower module | Secure encrypted reporting       |

 Example Queries

“What’s the current crime trend in Toronto?”
“Show me the top 5 neighbourhoods with car thefts.”
“Any active emergency alerts today?”
“Compare population vs. crime rate in North York.”
“What happened downtown yesterday?”

License

This project uses open public datasets licensed under the City of Toronto Open Data License.
The code and logic are © 2025 Anthony Ikenna Ogbonna and released under the MIT License.
