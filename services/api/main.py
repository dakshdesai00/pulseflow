import logging
import os
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from sqlalchemy import create_engine, text

# Basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pulseflow.api")

# Project root (two levels up from this file)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def resolve_path(rel_path: str) -> str:
    """Attempt to locate a file that may exist in several likely locations.

    The function checks several candidate locations and returns the first existing path.
    If none exists it returns the most likely path under the project root (but does not assume the file exists).
    This keeps behavior predictable in containers (where files are often copied to /app) and in local dev editors.
    """
    base_name = os.path.basename(rel_path)
    candidates = []

    # 1) same directory as this file (useful when Docker COPY . . places files at /app)
    candidates.append(os.path.join(os.path.dirname(__file__), base_name))

    # 2) current working directory (where uvicorn is often started)
    candidates.append(os.path.join(os.getcwd(), rel_path))
    candidates.append(os.path.join(os.getcwd(), base_name))

    # 3) typical container location
    candidates.append(os.path.join("/app", rel_path))
    candidates.append(os.path.join("/app", base_name))

    # 4) repository project path
    candidates.append(os.path.join(PROJECT_ROOT, rel_path))
    candidates.append(os.path.join(PROJECT_ROOT, base_name))

    # 5) relative to this file (preserve previous behavior)
    candidates.append(os.path.join(os.path.dirname(__file__), rel_path))

    for p in candidates:
        p = os.path.abspath(p)
        if os.path.exists(p):
            return p

    # fallback to project path (most predictable for local dev)
    return os.path.abspath(os.path.join(PROJECT_ROOT, rel_path))


# Provide a sensible default for local development when DATABASE_URL is not set
_default_db = f"sqlite:///{os.path.join(PROJECT_ROOT, 'pulseflow.db')}"
DATABASE_URL = os.getenv("DATABASE_URL", _default_db)


def create_database_if_missing(
    db_url: str, max_retries: int = 10, retry_interval: int = 2
):
    """If using PostgreSQL, connect to the server and create the target database if it doesn't exist.

    This helps when docker-compose spins up services and Postgres initializes, ensuring the expected DB is present.
    The function retries a few times while Postgres starts up.
    """
    import time

    from sqlalchemy.engine.url import make_url

    try:
        url = make_url(db_url)
    except Exception:
        logger.debug(
            "Could not parse DATABASE_URL, skipping DB auto-create: %s", db_url
        )
        return

    driver = (url.drivername or "").lower()
    if driver.startswith("sqlite"):
        return
    if "postgres" not in driver:
        return

    target_db = url.database
    if not target_db:
        return

    # Build admin URL that connects to the 'postgres' database so we can CREATE DATABASE
    admin_user = url.username or os.getenv("POSTGRES_USER", "postgres")
    admin_password = url.password or os.getenv("POSTGRES_PASSWORD", "postgres")
    host = url.host or "localhost"
    port = url.port or 5432
    admin_url = (
        f"{url.drivername}://{admin_user}:{admin_password}@{host}:{port}/postgres"
    )

    for attempt in range(1, max_retries + 1):
        try:
            admin_engine = create_engine(admin_url)
            with admin_engine.connect() as conn:
                exists = conn.execute(
                    text("SELECT 1 FROM pg_database WHERE datname = :d"),
                    {"d": target_db},
                ).scalar()
                if not exists:
                    logger.info("Database %s not found — creating", target_db)
                    conn.execute(text(f'CREATE DATABASE "{target_db}"'))
                    logger.info("Database %s created", target_db)
                else:
                    logger.info("Database %s already exists", target_db)
            return
        except Exception as e:
            logger.warning(
                "Database not ready or cannot create DB yet (%s) — attempt %d/%d",
                e,
                attempt,
                max_retries,
            )
            time.sleep(retry_interval)

    logger.error(
        "Failed to ensure database %s exists after %d attempts", target_db, max_retries
    )


# attempt to ensure the database exists when using Postgres
create_database_if_missing(DATABASE_URL)

engine = create_engine(DATABASE_URL)


# ============================================================
# DATABASE SETUP
# ============================================================


def create_tables():
    with engine.begin() as conn:
        conn.execute(
            text("""
            CREATE TABLE IF NOT EXISTS merchant_scores (
                card1 BIGINT PRIMARY KEY,

                total_gmv DOUBLE PRECISION,
                avg_amount DOUBLE PRECISION,
                std_amount DOUBLE PRECISION,
                max_amount DOUBLE PRECISION,
                min_amount DOUBLE PRECISION,
                max_avg_ratio DOUBLE PRECISION,
                cv DOUBLE PRECISION,

                total_transactions INTEGER,
                fraud_count INTEGER,
                fraud_rate DOUBLE PRECISION,

                pct_product_c DOUBLE PRECISION,
                pct_product_w DOUBLE PRECISION,

                avg_hour DOUBLE PRECISION,
                pct_credit DOUBLE PRECISION,
                pct_night_txn DOUBLE PRECISION,

                cluster INTEGER,
                cluster_label VARCHAR(50),

                is_anomaly INTEGER,
                anomaly_score DOUBLE PRECISION,

                T DOUBLE PRECISION,
                E INTEGER,

                survival_30 DOUBLE PRECISION,
                survival_60 DOUBLE PRECISION,
                survival_90 DOUBLE PRECISION
            )
            """)
        )

        conn.execute(
            text("""
            CREATE TABLE IF NOT EXISTS transactions (
                transaction_id BIGINT PRIMARY KEY,

                card1 BIGINT,
                amount DOUBLE PRECISION,
                product_cd VARCHAR(20),
                is_fraud INTEGER,

                cluster INTEGER,
                cluster_label VARCHAR(50),

                is_anomaly INTEGER,
                anomaly_score DOUBLE PRECISION,

                risk_score DOUBLE PRECISION,

                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
        )


# ============================================================
# INITIAL SEED
# ============================================================


def seed_if_empty():

    with engine.begin() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM merchant_scores")).scalar()

    if count and count > 0:
        logger.info("merchant_scores already seeded")
        return

    logger.info("Seeding merchant_scores...")

    csv_path = resolve_path(os.path.join("data", "merchant_scores_train.csv"))

    if not os.path.exists(csv_path):
        logger.info(
            "No merchant_scores_train.csv found at %s — skipping seed", csv_path
        )
        return

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower()

    cluster_fraud = df.groupby("cluster")["fraud_rate"].mean().sort_values()

    labels = [
        "Healthy",
        "Medium Risk",
        "High Risk",
        "Anomalous",
    ]

    label_map = {}

    for i, cluster_id in enumerate(cluster_fraud.index):
        label_map[cluster_id] = labels[i]

    df["cluster_label"] = df["cluster"].map(label_map)

    df.to_sql(
        "merchant_scores",
        engine,
        if_exists="append",
        index=False,
    )

    logger.info("merchant_scores seeded")


# ============================================================
# APP STARTUP
# ============================================================


@asynccontextmanager
async def lifespan(app: FastAPI):

    create_tables()
    seed_if_empty()

    yield


app = FastAPI(
    title="PulseFlow",
    lifespan=lifespan,
)


# ============================================================
# MERCHANT ENDPOINTS
# ============================================================


@app.get("/merchant/{card1}")
def get_merchant(card1: int):

    with engine.connect() as conn:
        result = conn.execute(
            text("""
            SELECT *
            FROM merchant_scores
            WHERE card1 = :card1
            """),
            {"card1": card1},
        ).fetchone()

    if result is None:
        return {"error": "merchant not found"}

    return dict(result._mapping)


@app.get("/merchants/high-risk")
def high_risk_merchants():

    with engine.connect() as conn:
        result = conn.execute(
            text("""
            SELECT
                card1,
                cluster_label,
                is_anomaly,
                fraud_rate,
                survival_30,
                survival_60,
                survival_90
            FROM merchant_scores
            WHERE is_anomaly = 1
               OR fraud_rate > 0.10
            ORDER BY fraud_rate DESC
            LIMIT 50
            """)
        ).fetchall()

    return [dict(r._mapping) for r in result]


@app.get("/merchants/stats")
def stats():

    with engine.connect() as conn:
        result = conn.execute(
            text("""
            SELECT
                COUNT(*) as total_merchants,
                SUM(is_anomaly) as total_anomalies,
                AVG(fraud_rate) as avg_fraud_rate,
                AVG(survival_90) as avg_survival_90
            FROM merchant_scores
            """)
        ).fetchone()

    return dict(result._mapping)


@app.get("/merchants/cluster-stats")
def cluster_stats():

    with engine.connect() as conn:
        result = conn.execute(
            text("""
            SELECT
                cluster_label,
                COUNT(*) as count,
                AVG(fraud_rate) as avg_fraud_rate
            FROM merchant_scores
            GROUP BY cluster_label
            ORDER BY avg_fraud_rate
            """)
        ).fetchall()

    return [dict(r._mapping) for r in result]


# ============================================================
# RISK SCORE
# ============================================================


@app.post("/merchant/score")
def score_merchant(data: dict):

    with engine.connect() as conn:
        result = conn.execute(
            text("""
            SELECT *
            FROM merchant_scores
            WHERE card1 = :card1
            """),
            {"card1": data.get("card1")},
        ).fetchone()

    if result is None:
        return {"error": "merchant not found"}

    m = dict(result._mapping)

    risk_score = (
        m["fraud_rate"] * 0.4 + m["is_anomaly"] * 0.3 + (1 - m["survival_90"]) * 0.3
    )

    recommendation = (
        "BLOCK" if risk_score > 0.30 else "REVIEW" if risk_score > 0.15 else "APPROVE"
    )

    return {
        "card1": m["card1"],
        "cluster_label": m["cluster_label"],
        "is_anomaly": m["is_anomaly"],
        "fraud_rate": m["fraud_rate"],
        "survival_90": m["survival_90"],
        "risk_score": round(risk_score, 4),
        "recommendation": recommendation,
    }


# ============================================================
# LIVE TRANSACTIONS
# ============================================================


@app.get("/transactions/recent")
def recent_transactions():

    with engine.connect() as conn:
        result = conn.execute(
            text("""
            SELECT *
            FROM transactions
            ORDER BY processed_at DESC
            LIMIT 100
            """)
        ).fetchall()

    return [dict(r._mapping) for r in result]


@app.get("/alerts/live")
def live_alerts():

    with engine.connect() as conn:
        result = conn.execute(
            text("""
            SELECT *
            FROM transactions
            WHERE is_anomaly = 1
               OR risk_score > 0.15
            ORDER BY processed_at DESC
            LIMIT 20
            """)
        ).fetchall()

    return [dict(r._mapping) for r in result]


# ============================================================
# DASHBOARD
# ============================================================


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    """Serve the static dashboard HTML. Try several likely locations and return the first that exists."""
    candidates = [
        resolve_path(os.path.join("services", "api", "dashboard.html")),
        resolve_path("dashboard.html"),
        os.path.join(os.path.dirname(__file__), "dashboard.html"),
        os.path.join(os.getcwd(), "dashboard.html"),
    ]

    for path in candidates:
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    return f.read()
        except Exception:
            logger.exception("Failed to open dashboard path %s", path)

    # nothing found — return a helpful error listing attempted paths
    logger.error("Dashboard not found. Tried: %s", candidates)
    msg = (
        "<h1>Dashboard not available</h1>"
        "<p>The dashboard file could not be found. Tried the following paths:</p>"
        "<ul>" + "".join(f"<li>{p}</li>" for p in candidates) + "</ul>"
    )
    return HTMLResponse(msg, status_code=500)


@app.get("/")
def root():
    return {
        "service": "PulseFlow",
        "status": "running",
    }
