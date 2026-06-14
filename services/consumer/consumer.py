import json
import logging
import os

import joblib
import numpy as np
import pandas as pd
from kafka import KafkaConsumer
from sqlalchemy import create_engine, text

# basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pulseflow.consumer")

# Project root (two levels up from this file)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def resolve_path(rel_path: str) -> str:
    base_name = os.path.basename(rel_path)
    candidates = []

    # same directory as this file
    candidates.append(os.path.join(os.path.dirname(__file__), base_name))

    # current working directory
    candidates.append(os.path.join(os.getcwd(), rel_path))
    candidates.append(os.path.join(os.getcwd(), base_name))

    # typical container location
    candidates.append(os.path.join("/app", rel_path))
    candidates.append(os.path.join("/app", base_name))

    # repository project path
    candidates.append(os.path.join(PROJECT_ROOT, rel_path))
    candidates.append(os.path.join(PROJECT_ROOT, base_name))

    # relative to this file
    candidates.append(os.path.join(os.path.dirname(__file__), rel_path))

    for p in candidates:
        p = os.path.abspath(p)
        if os.path.exists(p):
            return p

    # fallback
    return os.path.abspath(os.path.join(PROJECT_ROOT, rel_path))


# configure defaults for local dev
DATABASE_URL = os.getenv(
    "DATABASE_URL", f"sqlite:///{os.path.join(PROJECT_ROOT, 'pulseflow.db')}"
)
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")


def create_database_if_missing(
    db_url: str, max_retries: int = 10, retry_interval: int = 2
):
    """Ensure a Postgres database exists (retries while Postgres starts).

    Similar helper as in the API service so consumers can auto-create the DB when running in docker-compose.
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

FEATURE_COLUMNS = [
    "total_gmv",
    "avg_amount",
    "std_amount",
    "max_amount",
    "min_amount",
    "max_avg_ratio",
    "cv",
    "total_transactions",
    "fraud_rate",
    "pct_product_c",
    "pct_product_w",
    "avg_hour",
    "pct_credit",
    "pct_night_txn",
]

# models will be loaded lazily to reduce startup memory pressure
scaler = kmeans = iso = cox = None


def load_models():
    """Load model artifacts from the models directory (tries container /app and repo paths)."""
    global scaler, kmeans, iso, cox
    try:
        scaler_path = resolve_path(os.path.join("models", "scaler.pkl"))
        kmeans_path = resolve_path(os.path.join("models", "kmeans.pkl"))
        iso_path = resolve_path(os.path.join("models", "isolation_forest.pkl"))
        cox_path = resolve_path(os.path.join("models", "cox_model.pkl"))

        logger.info("Loading models from %s", os.path.dirname(scaler_path))
        scaler = joblib.load(scaler_path)
        kmeans = joblib.load(kmeans_path)
        iso = joblib.load(iso_path)
        cox = joblib.load(cox_path)

        logger.info(
            "Models loaded: scaler=%s kmeans=%s iso=%s cox=%s",
            scaler_path,
            kmeans_path,
            iso_path,
            cox_path,
        )
    except Exception as e:
        logger.exception("Failed loading model artifacts: %s", e)
        raise


CLUSTER_LABELS = {
    0: "Healthy",
    1: "High Risk",
    2: "Premium / Medium Risk",
    3: "Anomalous High Volume",
}


def create_transaction_table():
    """Create the transactions and merchant_scores tables if they don't exist."""
    with engine.begin() as conn:
        conn.execute(
            text(
                """
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
        """
            )
        )

        # ensure merchant_scores exists (same schema as API seed)
        conn.execute(
            text(
                """
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
        """
            )
        )


def build_initial_state(chunk_size: int = 100_000):
    """Return an empty dictionary initially.
    State for active merchants will be loaded from the database on-demand (lazily).
    """
    logger.info("Using lazy on-demand state loading for merchants")
    return {}


def load_merchant_from_db(card1: int) -> dict:
    """Load and reconstruct running merchant statistics from the database.
    
    This reconstructs the numeric running aggregates from stored averages, standard
    deviations, ratios, and percentages to enable incremental updates when new transactions arrive.
    """
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text(
                    """
                SELECT 
                    total_gmv, avg_amount, std_amount, max_amount, min_amount,
                    total_transactions, fraud_count, pct_product_c, pct_product_w,
                    avg_hour, pct_credit, pct_night_txn, t
                FROM merchant_scores
                WHERE card1 = :card1
                """
                ),
                {"card1": card1},
            ).fetchone()
            
            if result is None:
                return None
            
            row = result._mapping
            total_transactions = int(row["total_transactions"] or 0)
            if total_transactions <= 0:
                return None
            
            total_gmv = float(row["total_gmv"] or 0.0)
            avg_amount = float(row["avg_amount"] or 0.0)
            std_amount = float(row["std_amount"] or 0.0)
            
            # Reconstruct sum_sq_amounts: std^2 = mean(X^2) - mean(X)^2
            sum_sq_amounts = ((std_amount ** 2) + (avg_amount ** 2)) * total_transactions
            
            return {
                "sum_amounts": total_gmv,
                "sum_sq_amounts": sum_sq_amounts,
                "max_amount": float(row["max_amount"] or 0.0),
                "min_amount": float(row["min_amount"] or 0.0),
                "sum_hours": float(row["avg_hour"] or 0.0) * total_transactions,
                "fraud_count": int(row["fraud_count"] or 0),
                "total_transactions": total_transactions,
                "product_c_count": int(round((row["pct_product_c"] or 0.0) * total_transactions)),
                "product_w_count": int(round((row["pct_product_w"] or 0.0) * total_transactions)),
                "credit_count": int(round((row["pct_credit"] or 0.0) * total_transactions)),
                "night_count": int(round((row["pct_night_txn"] or 0.0) * total_transactions)),
                "first_txn": None, # Set dynamically when the first live transaction is processed
                "last_txn": None,
                "T_history": float(row["t"] or 0.0),
            }
    except Exception as e:
        logger.warning("Error loading merchant %d from DB: %s", card1, e)
    return None


def build_features(card1, s):
    """Build feature dictionary for a merchant state incrementally."""
    total_transactions = s.get("total_transactions", 0)

    total_gmv = float(s.get("sum_amounts", 0.0))
    avg_amount = float(total_gmv / total_transactions) if total_transactions else 0.0

    # Calculate standard deviation incrementally: std = sqrt(mean(X^2) - mean(X)^2)
    if total_transactions > 0:
        variance = (s.get("sum_sq_amounts", 0.0) / total_transactions) - (avg_amount ** 2)
        std_amount = float(np.sqrt(max(0.0, variance)))
    else:
        std_amount = 0.0

    max_amount = float(s.get("max_amount", 0.0))
    if max_amount == -float("inf"):
        max_amount = 0.0
    min_amount = float(s.get("min_amount", 0.0))
    if min_amount == float("inf"):
        min_amount = 0.0

    fraud_rate = (
        (s.get("fraud_count", 0) / total_transactions) if total_transactions else 0.0
    )

    pct_product_c = (
        (s.get("product_c_count", 0) / total_transactions)
        if total_transactions
        else 0.0
    )
    pct_product_w = (
        (s.get("product_w_count", 0) / total_transactions)
        if total_transactions
        else 0.0
    )
    pct_credit = (
        (s.get("credit_count", 0) / total_transactions) if total_transactions else 0.0
    )
    pct_night_txn = (
        (s.get("night_count", 0) / total_transactions) if total_transactions else 0.0
    )

    avg_hour = float(s.get("sum_hours", 0.0) / total_transactions) if total_transactions else 0.0

    max_avg_ratio = (max_amount / avg_amount) if avg_amount > 0 else 0.0
    cv = (std_amount / avg_amount) if avg_amount > 0 else 0.0

    return {
        "card1": card1,
        "total_gmv": total_gmv,
        "avg_amount": avg_amount,
        "std_amount": std_amount,
        "max_amount": max_amount,
        "min_amount": min_amount,
        "max_avg_ratio": max_avg_ratio,
        "cv": cv,
        "total_transactions": total_transactions,
        "fraud_count": s.get("fraud_count", 0),
        "fraud_rate": fraud_rate,
        "pct_product_c": pct_product_c,
        "pct_product_w": pct_product_w,
        "avg_hour": avg_hour,
        "pct_credit": pct_credit,
        "pct_night_txn": pct_night_txn,
    }


def main():
    import time

    # Load model artifacts at startup
    load_models()

    while True:
        try:
            create_transaction_table()
            merchant_state = build_initial_state()

            consumer = KafkaConsumer(
                "transactions",
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                value_deserializer=lambda x: json.loads(x.decode("utf-8")),
                auto_offset_reset="earliest",
            )

            logger.info(
                "Consumer started — listening for transactions on %s",
                KAFKA_BOOTSTRAP_SERVERS,
            )

            for message in consumer:
                try:
                    txn = message.value

                    card1 = (
                        int(txn.get("card1")) if txn.get("card1") is not None else None
                    )
                    if card1 is None:
                        continue

                    if card1 not in merchant_state:
                        # Load merchant features dynamically from DB
                        db_state = load_merchant_from_db(card1)
                        if db_state:
                            merchant_state[card1] = db_state
                        else:
                            merchant_state[card1] = {
                                "sum_amounts": 0.0,
                                "sum_sq_amounts": 0.0,
                                "max_amount": -float("inf"),
                                "min_amount": float("inf"),
                                "sum_hours": 0.0,
                                "fraud_count": 0,
                                "total_transactions": 0,
                                "product_c_count": 0,
                                "product_w_count": 0,
                                "credit_count": 0,
                                "night_count": 0,
                                "first_txn": txn.get("TransactionDT"),
                                "last_txn": txn.get("TransactionDT"),
                                "T_history": 0.0,
                            }

                    s = merchant_state[card1]

                    # Initialize lazy timestamps if we loaded from DB
                    if s.get("first_txn") is None:
                        txn_dt = txn.get("TransactionDT", 0.0)
                        s["first_txn"] = txn_dt - s["T_history"] * 86400
                        s["last_txn"] = txn_dt

                    amt = float(txn.get("TransactionAmt", 0.0))
                    s["sum_amounts"] += amt
                    s["sum_sq_amounts"] += amt * amt
                    s["max_amount"] = max(s["max_amount"], amt)
                    s["min_amount"] = min(s["min_amount"], amt)
                    s["total_transactions"] += 1
                    s["fraud_count"] += int(txn.get("isFraud", 0))

                    if txn.get("ProductCD") == "C":
                        s["product_c_count"] += 1
                    if txn.get("ProductCD") == "W":
                        s["product_w_count"] += 1

                    if str(txn.get("card6", "")).lower() == "credit":
                        s["credit_count"] += 1

                    try:
                        hour = int(float(txn.get("TransactionDT", 0)) // 3600) % 24
                    except Exception:
                        hour = 0

                    s["sum_hours"] += hour
                    if 0 <= hour <= 6:
                        s["night_count"] += 1

                    s["last_txn"] = txn.get("TransactionDT", s.get("last_txn"))

                    features = build_features(card1, s)

                    # select the columns expected by the scaler/model
                    X = pd.DataFrame([features])[FEATURE_COLUMNS]
                    X_scaled = scaler.transform(X)

                    cluster = int(kmeans.predict(X_scaled)[0])
                    cluster_label = CLUSTER_LABELS.get(cluster, "Unknown")

                    is_anomaly = int(iso.predict(X_scaled)[0] == -1)
                    anomaly_score = float(iso.score_samples(X_scaled)[0])

                    # Time since first transaction in days (fallback to 0 if missing)
                    T = (
                        (s.get("last_txn", 0) - s.get("first_txn", 0)) / 86400
                        if s.get("first_txn") is not None
                        else 0
                    )

                    cox_input = pd.DataFrame(
                        [
                            {
                                "fraud_rate": features["fraud_rate"],
                                "avg_amount": features["avg_amount"],
                                "cv": features["cv"],
                                "pct_product_c": features["pct_product_c"],
                                "pct_credit": features["pct_credit"],
                                "pct_night_txn": features["pct_night_txn"],
                                "total_transactions": features["total_transactions"],
                                "is_anomaly": is_anomaly,
                                "cluster": cluster,
                            }
                        ]
                    )

                    survival = cox.predict_survival_function(cox_input)

                    # Find indices closest to day 30/60/90 on the survival timeline
                    idx30 = int(np.abs(survival.index.values - 30).argmin())
                    idx60 = int(np.abs(survival.index.values - 60).argmin())
                    idx90 = int(np.abs(survival.index.values - 90).argmin())

                    survival_30 = float(survival.iloc[idx30].values[0])
                    survival_60 = float(survival.iloc[idx60].values[0])
                    survival_90 = float(survival.iloc[idx90].values[0])

                    risk_score = (
                        features["fraud_rate"] * 0.4
                        + is_anomaly * 0.3
                        + (1 - survival_90) * 0.3
                    )

                    with engine.begin() as conn:
                        conn.execute(
                            text(
                                """
                    INSERT INTO transactions(
                        transaction_id,
                        card1,
                        amount,
                        product_cd,
                        is_fraud,
                        cluster,
                        cluster_label,
                        is_anomaly,
                        anomaly_score,
                        risk_score
                    )
                    VALUES (
                        :transaction_id,
                        :card1,
                        :amount,
                        :product_cd,
                        :is_fraud,
                        :cluster,
                        :cluster_label,
                        :is_anomaly,
                        :anomaly_score,
                        :risk_score
                    )
                    ON CONFLICT DO NOTHING
                    """
                            ),
                            {
                                "transaction_id": int(txn.get("TransactionID", 0)),
                                "card1": card1,
                                "amount": amt,
                                "product_cd": txn.get("ProductCD"),
                                "is_fraud": int(txn.get("isFraud", 0)),
                                "cluster": cluster,
                                "cluster_label": cluster_label,
                                "is_anomaly": is_anomaly,
                                "anomaly_score": anomaly_score,
                                "risk_score": risk_score,
                            },
                        )

                        exists = conn.execute(
                            text(
                                """
                            SELECT 1
                            FROM merchant_scores
                            WHERE card1=:card1
                            """
                            ),
                            {"card1": card1},
                        ).fetchone()

                        if exists:
                            conn.execute(
                                text(
                                    """
                                UPDATE merchant_scores
                                SET
                                    total_gmv = :total_gmv,
                                    avg_amount = :avg_amount,
                                    std_amount = :std_amount,
                                    max_amount = :max_amount,
                                    min_amount = :min_amount,
                                    max_avg_ratio = :max_avg_ratio,
                                    cv = :cv,
                                    total_transactions = :total_transactions,
                                    fraud_count = :fraud_count,
                                    fraud_rate = :fraud_rate,
                                    pct_product_c = :pct_product_c,
                                    pct_product_w = :pct_product_w,
                                    avg_hour = :avg_hour,
                                    pct_credit = :pct_credit,
                                    pct_night_txn = :pct_night_txn,
                                    cluster = :cluster,
                                    cluster_label = :cluster_label,
                                    is_anomaly = :is_anomaly,
                                    anomaly_score = :anomaly_score,
                                    T = :T,
                                    survival_30 = :survival_30,
                                    survival_60 = :survival_60,
                                    survival_90 = :survival_90
                                WHERE card1 = :card1
                                """
                                ),
                                {
                                    **features,
                                    "card1": card1,
                                    "cluster": cluster,
                                    "cluster_label": cluster_label,
                                    "is_anomaly": is_anomaly,
                                    "anomaly_score": anomaly_score,
                                    "T": T,
                                    "survival_30": survival_30,
                                    "survival_60": survival_60,
                                    "survival_90": survival_90,
                                },
                            )
                        else:
                            conn.execute(
                                text(
                                    """
                                INSERT INTO merchant_scores(
                                    card1,
                                    total_gmv,
                                    avg_amount,
                                    std_amount,
                                    max_amount,
                                    min_amount,
                                    max_avg_ratio,
                                    cv,
                                    total_transactions,
                                    fraud_count,
                                    fraud_rate,
                                    pct_product_c,
                                    pct_product_w,
                                    avg_hour,
                                    pct_credit,
                                    pct_night_txn,
                                    cluster,
                                    cluster_label,
                                    is_anomaly,
                                    anomaly_score,
                                    T,
                                    survival_30,
                                    survival_60,
                                    survival_90
                                )
                                VALUES (
                                    :card1,
                                    :total_gmv,
                                    :avg_amount,
                                    :std_amount,
                                    :max_amount,
                                    :min_amount,
                                    :max_avg_ratio,
                                    :cv,
                                    :total_transactions,
                                    :fraud_count,
                                    :fraud_rate,
                                    :pct_product_c,
                                    :pct_product_w,
                                    :avg_hour,
                                    :pct_credit,
                                    :pct_night_txn,
                                    :cluster,
                                    :cluster_label,
                                    :is_anomaly,
                                    :anomaly_score,
                                    :T,
                                    :survival_30,
                                    :survival_60,
                                    :survival_90
                                )
                                """
                                ),
                                {
                                    **features,
                                    "card1": card1,
                                    "cluster": cluster,
                                    "cluster_label": cluster_label,
                                    "is_anomaly": is_anomaly,
                                    "anomaly_score": anomaly_score,
                                    "T": T,
                                    "survival_30": survival_30,
                                    "survival_60": survival_60,
                                    "survival_90": survival_90,
                                },
                            )

                    logger.info(
                        "processed txn=%s merchant=%s risk=%.3f",
                        txn.get("TransactionID"),
                        card1,
                        risk_score,
                    )
                except Exception:
                    logger.exception("Error while processing message — will continue")

        except Exception:
            logger.exception("Consumer failed — restarting in 5s")
            try:
                consumer.close()
            except Exception:
                pass
            time.sleep(5)


if __name__ == "__main__":
    main()
