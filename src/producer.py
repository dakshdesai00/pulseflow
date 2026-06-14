import json
import time

import pandas as pd
from kafka import KafkaProducer

TOPIC = "transactions"

producer = KafkaProducer(
    bootstrap_servers="localhost:9094",
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)

df = pd.read_csv("../data/test_transaction.csv")

for _, row in df.iterrows():
    payload = {
        "TransactionID": int(row["TransactionID"]),
        "TransactionDT": float(row["TransactionDT"]),
        "TransactionAmt": float(row["TransactionAmt"]),
        "ProductCD": str(row["ProductCD"]),
        "card1": int(row["card1"]),
        "card6": str(row["card6"]) if pd.notna(row["card6"]) else "",
        "isFraud": int(row["isFraud"]) if "isFraud" in row else 0,
    }

    producer.send(TOPIC, payload)
    print(f"sent txn={payload['TransactionID']} merchant={payload['card1']}")

    time.sleep(0.1)

producer.flush()
