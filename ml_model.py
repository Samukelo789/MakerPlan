import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
import pickle
import os

def load_data():
    path = os.path.join(os.path.dirname(__file__), "Maker-Dataset.csv")
    df = pd.read_csv(path)
    return df

def train_models():
    print(" Loading maker dataset...")
    df = load_data()
    print(f" Loaded {len(df)} records")

    # Encode text columns into numbers
    encoders = {}
    for col in ['product_idea', 'category', 'skill_level',
                'available_materials', 'available_tools', 'build_feasible']:
        le = LabelEncoder()
        df[col + '_enc'] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # Features used for prediction
    features = ['skill_level_enc', 'category_enc']

    # -------------------------------------------------------
    # MODEL 1: Cost Predictor (Regression)
    # Predicts: how much will this prototype cost to build?
    # -------------------------------------------------------
    print("\n Training Model 1: Cost Predictor...")
    X = df[features]
    y = df['estimated_cost_zar']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    cost_model = RandomForestRegressor(n_estimators=100, random_state=42)
    cost_model.fit(X_train, y_train)
    mae = mean_absolute_error(y_test, cost_model.predict(X_test))
    print(f" Cost Predictor trained! Average error: R{mae:.2f}")

    # -------------------------------------------------------
    # MODEL 2: Feasibility Predictor (Classification)
    # Predicts: can this prototype actually be built? (YES/NO)
    # -------------------------------------------------------
    print("\n Training Model 2: Feasibility Predictor...")
    y_feas = df['build_feasible_enc']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_feas, test_size=0.2, random_state=42)
    feas_model = RandomForestClassifier(n_estimators=100, random_state=42)
    feas_model.fit(X_train, y_train)
    acc = accuracy_score(y_test, feas_model.predict(X_test))
    print(f"Feasibility Predictor trained! Accuracy: {acc*100:.1f}%")

    # -------------------------------------------------------
    # MODEL 3: Complexity Classifier (Classification)
    # Predicts: what skill level is needed for this build?
    # -------------------------------------------------------
    print("\n Training Model 3: Complexity Classifier...")
    y_skill = df['skill_level_enc']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_skill, test_size=0.2, random_state=42)
    skill_model = RandomForestClassifier(n_estimators=100, random_state=42)
    skill_model.fit(X_train, y_train)
    acc2 = accuracy_score(y_test, skill_model.predict(X_test))
    print(f"Complexity Classifier trained! Accuracy: {acc2*100:.1f}%")

    # Save all models and encoders
    print("\n Saving models...")
    with open("cost_model.pkl",  "wb") as f: pickle.dump(cost_model,  f)
    with open("feas_model.pkl",  "wb") as f: pickle.dump(feas_model,  f)
    with open("skill_model.pkl", "wb") as f: pickle.dump(skill_model, f)
    with open("encoders.pkl",    "wb") as f: pickle.dump(encoders,    f)
    print("All models saved!")


def load_models():
    with open("cost_model.pkl",  "rb") as f: cost_model  = pickle.load(f)
    with open("feas_model.pkl",  "rb") as f: feas_model  = pickle.load(f)
    with open("skill_model.pkl", "rb") as f: skill_model = pickle.load(f)
    with open("encoders.pkl",    "rb") as f: encoders    = pickle.load(f)
    return cost_model, feas_model, skill_model, encoders


def predict(category, skill_level):
    """
    Given a product category and skill level, returns:
    - predicted build cost
    - predicted feasibility (YES/NO)
    - predicted complexity
    - confidence score
    """
    cost_model, feas_model, skill_model, encoders = load_models()

    try:
        category_enc  = encoders['category'].transform([category.lower()])[0]
    except:
        category_enc  = 0

    try:
        skill_enc = encoders['skill_level'].transform([skill_level])[0]
    except:
        skill_enc = 0

    features = [[skill_enc, category_enc]]

    predicted_cost      = round(cost_model.predict(features)[0])
    predicted_feas_enc  = feas_model.predict(features)[0]
    predicted_feas      = encoders['build_feasible'].inverse_transform(
                            [predicted_feas_enc])[0]
    confidence          = round(max(feas_model.predict_proba(features)[0]) * 100, 1)
    predicted_skill_enc = skill_model.predict(features)[0]
    predicted_skill     = encoders['skill_level'].inverse_transform(
                            [predicted_skill_enc])[0]

    return {
        "predicted_cost":       f"R{predicted_cost}",
        "build_feasible":       predicted_feas,
        "required_skill":       predicted_skill,
        "confidence":           f"{confidence}%"
    }


if __name__ == "__main__":
    train_models()