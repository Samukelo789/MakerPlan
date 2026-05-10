import sqlite3
import csv
import os

def get_db():
    conn = sqlite3.connect("maker_planner.db")
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()

    # Table 1: Makers/Technicians at the shop
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS makers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            skills TEXT NOT NULL,
            skill_level TEXT NOT NULL,
            available INTEGER DEFAULT 1
        )
    """)

    # Table 2: Prototype plan requests
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            maker_name TEXT NOT NULL,
            product_idea TEXT NOT NULL,
            materials TEXT NOT NULL,
            tools TEXT NOT NULL,
            budget TEXT NOT NULL,
            skill_level TEXT NOT NULL,
            status TEXT DEFAULT 'Pending',
            assigned_to TEXT DEFAULT 'Unassigned',
            llm_plan TEXT DEFAULT '',
            ml_feasible TEXT DEFAULT '',
            ml_cost TEXT DEFAULT '',
            ml_skill TEXT DEFAULT '',
            ml_confidence TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Table 3: Maker dataset
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS maker_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_idea TEXT NOT NULL,
            category TEXT NOT NULL,
            skill_level TEXT NOT NULL,
            available_materials TEXT NOT NULL,
            available_tools TEXT NOT NULL,
            budget_zar INTEGER NOT NULL,
            estimated_cost_zar INTEGER NOT NULL,
            build_feasible TEXT NOT NULL,
            build_time TEXT NOT NULL,
            revisions_needed INTEGER NOT NULL
        )
    """)

    # Load makers only once
    cursor.execute("SELECT COUNT(*) FROM makers")
    if cursor.fetchone()[0] == 0:
        sample_makers = [
            ("Sipho",  "screen repair, hardware assembly, soldering", "Advanced"),
            ("Tariq",  "software tools, 3D design, electronics", "Intermediate"),
            ("Zain",   "woodwork, metalwork, recycled materials", "Advanced"),
        ]
        cursor.executemany(
            "INSERT INTO makers (name, skills, skill_level) VALUES (?, ?, ?)",
            sample_makers
        )
        print("✅ Makers loaded!")

    # Load CSV dataset only once
    cursor.execute("SELECT COUNT(*) FROM maker_history")
    if cursor.fetchone()[0] == 0:
        csv_path = os.path.join(os.path.dirname(__file__), "Maker-Dataset.csv")
        if os.path.exists(csv_path):
            with open(csv_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = [
                    (
                        row["product_idea"],
                        row["category"],
                        row["skill_level"],
                        row["available_materials"],
                        row["available_tools"],
                        int(row["budget_zar"]),
                        int(row["estimated_cost_zar"]),
                        row["build_feasible"],
                        row["build_time"],
                        int(row["revisions_needed"]),
                    )
                    for row in reader
                ]
            cursor.executemany("""
                INSERT INTO maker_history
                (product_idea, category, skill_level, available_materials,
                 available_tools, budget_zar, estimated_cost_zar,
                 build_feasible, build_time, revisions_needed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, rows)
            print(f"✅ Loaded {len(rows)} maker history records!")
        else:
            print("⚠️ Maker-Dataset.csv not found!")

    conn.commit()
    conn.close()
    print("✅ Database ready!")


def get_similar_builds(product_idea=None, skill_level=None):
    """Fetch similar past builds to give the AI better context"""
    conn = get_db()
    if product_idea and skill_level:
        rows = conn.execute("""
            SELECT * FROM maker_history
            WHERE product_idea LIKE ? OR skill_level = ?
            LIMIT 5
        """, (f"%{product_idea}%", skill_level)).fetchall()
    else:
        rows = conn.execute("""
            SELECT * FROM maker_history LIMIT 5
        """).fetchall()
    conn.close()
    return rows