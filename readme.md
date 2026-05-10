# MakerPlan — LLM-Guided Micro-Manufacturing Prototype Planner

**Group 2 | Living Lab 2026 | Community Phone Repair Shop, Richards Bay**

MakerPlan is an AI-powered prototype planning system for community maker
spaces. Makers describe a product idea and their local constraints, and
the system generates a practical step-by-step prototype plan using machine
learning and a large language model.

---

## What It Does

- Accepts a product idea, available materials, tools, budget, and skill level
- Runs three ML models to predict build feasibility, estimated cost, and required skill level
- Calls LLaMA 3.3 70B (via Groq) to generate 2-3 prototype variants with full build instructions
- Stores all plans in a local SQLite database
- Provides a maker feedback loop so unrealistic plans can be revised by the AI
- Displays live analytics with Chart.js visualisations

---

## Project Structure
```
phone-repair-planner/
├── app.py                   # Flask backend — all routes
├── database.py              # SQLite setup + CSV loader
├── ml_model.py              # Train and predict functions for 3 ML models
├── prompts.py               # LLM prompt builder
├── evaluate_models.py       # ML evaluation script — prints metrics + saves charts
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── Maker-Dataset.csv        # 600 synthetic maker space records
├── cost_model.pkl           # Trained Random Forest Regressor (cost)
├── feas_model.pkl           # Trained Random Forest Classifier (feasibility)
├── skill_model.pkl          # Trained Random Forest Classifier (skill level)
├── encoders.pkl             # Label encoders for all categorical columns
├── maker_planner.db         # SQLite database (auto-created on first run)
└── templates/
    ├── index.html           # New plan — constraint capture form
    ├── result.html          # Prototype plan output + ML predictions
    ├── dashboard.html       # All plans with status management
    ├── analytics.html       # Live KPIs and Chart.js visualisations
    └── ethics.html          # Risk register, consent, AI charter
```

---

## Setup Instructions

### 1. Install dependencies
```bash
pip install flask groq scikit-learn pandas matplotlib
```

### 2. Add your Groq API key
Open `app.py` and replace the API key on this line:
```python
client = Groq(api_key="YOUR_KEY_HERE")
```
Get a free key at: https://console.groq.com

### 3. Place the dataset
Make sure `Maker-Dataset.csv` is in the project root folder.

### 4. Train the ML models
```bash
python ml_model.py
```
This creates `cost_model.pkl`, `feas_model.pkl`, `skill_model.pkl`, and `encoders.pkl`.

### 5. Run the app
```bash
python app.py
```
Visit: http://127.0.0.1:5000

---

## Application Routes

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Maker constraint-capture form |
| `/submit` | POST | Run ML + call LLM + save plan |
| `/result/<id>` | GET | Show prototype plan and ML predictions |
| `/dashboard` | GET | All plans with filtering and status management |
| `/update/<id>` | POST | Update plan status and assignment |
| `/feedback/<id>` | POST | Submit feedback and trigger LLM plan revision |
| `/analytics` | GET | Live KPIs, charts, and plan metrics table |
| `/ethics` | GET | Risk register, consent, safeguards, AI charter |

---

## ML Models

| Model | Algorithm | Predicts | Input Features |
|---|---|---|---|
| Cost Predictor | Random Forest Regressor | Build cost in ZAR | skill_level, category |
| Feasibility Predictor | Random Forest Classifier | Build feasible YES/NO | skill_level, category |
| Complexity Classifier | Random Forest Classifier | Required skill level | skill_level, category |

### Retrain models
```bash
python ml_model.py
```

### Evaluate models
```bash
python evaluate_models.py
```
Prints MAE, R², accuracy scores, classification reports, and saves
`model_evaluation.png` with four evaluation charts.

---

## Dataset

**File:** `Maker-Dataset.csv`
**Records:** 600 synthetic maker space records
**Columns:** product_idea, category, skill_level, available_materials,
available_tools, budget_zar, estimated_cost_zar, build_feasible,
build_time, revisions_needed

| Attribute | Detail |
|---|---|
| Feasible builds | 501 (83.5%) |
| Not feasible | 99 (16.5%) |
| Skill levels | Beginner 201, Intermediate 183, Advanced 216 |
| Product types | Phone Stand, Charging Dock, Cable Organiser, Repair Bench Organiser, Phone Amplifier, Protective Case |

---

## LLM Integration

- **Provider:** Groq (free tier)
- **Model:** `llama-3.3-70b-versatile`
- **Prompt includes:** product idea, materials, tools, budget, skill level, 5 similar past builds, ML predictions
- **Output format:** Feasibility assessment, 3 prototype variants, materials list, step-by-step build instructions, maintenance guide, cost and time estimates, safety notes

---

## Resetting the Database

If you change the dataset or schema:
```bash
del maker_planner.db      # Windows
python app.py             # Recreates database automatically on startup
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python Flask |
| LLM | LLaMA 3.3 70B via Groq API |
| ML | scikit-learn Random Forest |
| Database | SQLite |
| Frontend | HTML, CSS, JavaScript (Jinja2) |
| Charts | Chart.js 4.4 |
| Fonts | DM Sans + DM Mono (Google Fonts) |

---

## Living Lab Design Cycles

| Cycle | Focus | Output |
|---|---|---|
| Cycle 1 | Initial system with repair shop data | Working Flask app with ML and LLM |
| Cycle 2 | Pivot — architecture redesign for maker planning | New dataset, schema, user journey |
| Cycle 3 | Full prototype planner implementation | Complete MakerPlan system |

---

## Success Metric

> Percentage of AI-generated prototype plans judged feasible by community
> technicians at the partner maker space.

Tracked live on the Analytics page at `/analytics`.