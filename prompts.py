def build_maker_prompt(maker_name, product_idea, materials, 
                       tools, budget, skill_level, 
                       history=None, ml_predictions=None):

    # Format similar past builds
    history_text = ""
    if history and len(history) > 0:
        history_text = "\nSIMILAR PAST BUILDS FROM OUR MAKER SPACE:\n"
        for h in history:
            history_text += (
                f"- {h['product_idea']} | Skill: {h['skill_level']} | "
                f"Materials: {h['available_materials']} | "
                f"Cost: R{h['estimated_cost_zar']} | "
                f"Time: {h['build_time']} | "
                f"Feasible: {h['build_feasible']}\n"
            )

    # Format ML predictions
    ml_text = ""
    if ml_predictions:
        ml_text = f"""
OUR ML MODEL PREDICTIONS:
- Predicted build cost: {ml_predictions['predicted_cost']}
- Build feasibility: {ml_predictions['build_feasible']}
- Required skill level: {ml_predictions['required_skill']}
- Confidence: {ml_predictions['confidence']}
"""

    prompt = f"""
You are an expert prototype planning assistant for a community 
maker space in South Africa. A local maker wants to build something 
using locally available materials and tools.

Your job is to generate a realistic, practical prototype plan 
tailored to their exact constraints.

---

MAKER DETAILS:
- Name: {maker_name}
- Product Idea: {product_idea}
- Available Materials: {materials}
- Available Tools: {tools}
- Budget: R{budget}
- Skill Level: {skill_level}
{history_text}
{ml_text}
---

Please respond in the following format EXACTLY:

FEASIBILITY ASSESSMENT: [Can this be built with the given constraints? Yes/No and why in one sentence]

PROTOTYPE VARIANTS:
1. [Low-Cost Option] - Brief description, estimated cost in ZAR, estimated build time
2. [Durable Option] - Brief description, estimated cost in ZAR, estimated build time
3. [Easy-to-Build Option] - Brief description, estimated cost in ZAR, estimated build time

RECOMMENDED VARIANT: [Which variant best suits this maker's skill and budget, and why]

MATERIALS LIST:
- [Material 1]: quantity, estimated cost in ZAR, where to source locally
- [Material 2]: quantity, estimated cost in ZAR, where to source locally
(list all materials needed)

STEP-BY-STEP BUILD INSTRUCTIONS:
1. [Step one]
2. [Step two]
3. [Step three]
(continue as needed)

MAINTENANCE AND REPAIR GUIDE:
- [How to maintain this prototype]
- [How to repair common issues]

COST ESTIMATE: R[amount] - R[amount]
TIME ESTIMATE: [X] hours/days

SAFETY NOTES:
- [Any important safety warnings for this build]

Keep language simple and practical. 
This is for a real community maker space in South Africa.
"""
    return prompt