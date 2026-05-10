from flask import Flask, render_template, request, redirect, url_for
from groq import Groq
from database import get_db, init_db, get_similar_builds
from prompts import build_maker_prompt
from ml_model import predict

app = Flask(__name__)

client = Groq(api_key="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")

# -------------------------------------------------------
# ROUTE 1: Home — Maker submits a product idea
# -------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


# -------------------------------------------------------
# ROUTE 2: Process form — Run ML + LLM, save plan
# -------------------------------------------------------
@app.route("/submit", methods=["POST"])
def submit():
    maker_name   = request.form["maker_name"]
    product_idea = request.form["product_idea"]
    materials    = request.form["materials"]
    tools        = request.form["tools"]
    budget       = request.form["budget"]
    skill_level  = request.form["skill_level"]

    # Detect category from product idea
    category_map = {
        "stand":     "display",
        "dock":      "charging",
        "charger":   "charging",
        "organiser": "organisation",
        "holder":    "organisation",
        "case":      "protection",
        "amplifier": "audio",
        "speaker":   "audio",
        "box":       "organisation",
        "rack":      "organisation",
        "mat":       "protection",
    }
    detected_category = "organisation"
    for keyword, cat in category_map.items():
        if keyword in product_idea.lower():
            detected_category = cat
            break

    # Run ML predictions
    ml_predictions = predict(detected_category, skill_level)

    # Fetch similar past builds for context
    history = get_similar_builds(
        product_idea=product_idea,
        skill_level=skill_level
    )

    # Build prompt and call LLM
    prompt = build_maker_prompt(
        maker_name, product_idea, materials,
        tools, budget, skill_level,
        history, ml_predictions
    )
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    llm_plan = response.choices[0].message.content

    # Save to database
    conn = get_db()
    conn.execute("""
        INSERT INTO plans 
        (maker_name, product_idea, materials, tools, budget, 
         skill_level, llm_plan, ml_feasible, ml_cost, 
         ml_skill, ml_confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (maker_name, product_idea, materials, tools, budget,
          skill_level, llm_plan,
          ml_predictions["build_feasible"],
          ml_predictions["predicted_cost"],
          ml_predictions["required_skill"],
          ml_predictions["confidence"]))
    conn.commit()
    plan_id = conn.execute(
        "SELECT last_insert_rowid()"
    ).fetchone()[0]
    conn.close()

    return redirect(url_for("result", plan_id=plan_id))


# -------------------------------------------------------
# ROUTE 3: Result — Show the prototype plan
# -------------------------------------------------------
@app.route("/result/<int:plan_id>")
def result(plan_id):
    conn = get_db()
    plan = conn.execute(
        "SELECT * FROM plans WHERE id = ?", (plan_id,)
    ).fetchone()
    conn.close()
    return render_template("result.html", plan=plan)


# -------------------------------------------------------
# ROUTE 4: Dashboard — All prototype plans
# -------------------------------------------------------
@app.route("/dashboard")
def dashboard():
    conn = get_db()
    plans = conn.execute(
        "SELECT * FROM plans ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return render_template("dashboard.html", plans=plans)


# -------------------------------------------------------
# ROUTE 5: Update plan status
# -------------------------------------------------------
@app.route("/update/<int:plan_id>", methods=["POST"])
def update(plan_id):
    new_status = request.form["status"]
    assigned   = request.form["assigned_to"]
    conn = get_db()
    conn.execute("""
        UPDATE plans SET status = ?, assigned_to = ?
        WHERE id = ?
    """, (new_status, assigned, plan_id))
    conn.commit()
    conn.close()
    return redirect(url_for("dashboard"))


# -------------------------------------------------------
# ROUTE 6: Feedback loop — LLM revises plan
# -------------------------------------------------------
@app.route("/feedback/<int:plan_id>", methods=["POST"])
def feedback(plan_id):
    maker_feedback = request.form["feedback"]
    conn = get_db()
    plan = conn.execute(
        "SELECT * FROM plans WHERE id = ?", (plan_id,)
    ).fetchone()

    revision_prompt = f"""
You previously generated this prototype plan:

{plan['llm_plan']}

A local maker or technician reviewed it and flagged these issues:
"{maker_feedback}"

Please revise the prototype plan addressing these concerns.
Keep the same format but adjust recommendations to be more 
realistic for local conditions in South Africa.
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": revision_prompt}]
    )
    revised_plan = response.choices[0].message.content

    conn.execute(
        "UPDATE plans SET llm_plan = ? WHERE id = ?",
        (revised_plan, plan_id)
    )
    conn.commit()
    conn.close()
    return redirect(url_for("result", plan_id=plan_id))

# -------------------------------------------------------
# ROUTE: Analytics page
# -------------------------------------------------------
@app.route("/analytics")
def analytics():
    conn = get_db()
    plans = conn.execute(
        "SELECT * FROM plans ORDER BY created_at DESC"
    ).fetchall()
    conn.close()

    total    = len(plans)
    feasible = sum(1 for p in plans if p['ml_feasible'] == 'YES')
    done     = sum(1 for p in plans if p['status'] == 'Done')

    # Skill level counts
    skill_data = {}
    for p in plans:
        s = p['skill_level']
        skill_data[s] = skill_data.get(s, 0) + 1

    # Product idea counts
    product_data = {}
    for p in plans:
        prod = p['product_idea']
        product_data[prod] = product_data.get(prod, 0) + 1

    # Status counts
    status_data = {}
    for p in plans:
        st = p['status']
        status_data[st] = status_data.get(st, 0) + 1

    # Budget vs ML cost
    budgets  = [int(p['budget']) for p in plans]
    ml_costs = [int(p['ml_cost'].replace('R','')) for p in plans if p['ml_cost']]
    plan_ids = [p['id'] for p in plans]

    # Average budget
    avg_budget = f"R{int(sum(budgets)/len(budgets))}" if budgets else "R0"

    stats = {
        "total_plans":      total,
        "feasible_count":   feasible,
        "infeasible_count": total - feasible,
        "feasibility_rate": round((feasible/total*100), 1) if total > 0 else 0,
        "done_count":       done,
        "avg_budget":       avg_budget,
        "feas_data":        {"yes": feasible, "no": total - feasible},
        "skill_data":       skill_data,
        "product_data":     product_data,
        "status_data":      status_data,
        "budgets":          budgets,
        "ml_costs":         ml_costs,
        "plan_ids":         plan_ids,
    }

    return render_template("analytics.html", stats=stats, plans=plans)
# -------------------------------------------------------
# ROUTE 7: Ethics page
# -------------------------------------------------------
@app.route("/ethics")
def ethics():
    return render_template("ethics.html")


# -------------------------------------------------------
# START
# -------------------------------------------------------
if __name__ == "__main__":
    init_db()
    app.run(debug=True)