"""Read the chapter 5 prediction file from the inference-datasets bucket.

    python book/ch05/show_predictions.py
"""

import io

from minio import Minio
import pandas as pd

client = Minio("localhost:9000", access_key="minio", secret_key="minio123", secure=False)
data = client.get_object("inference-datasets", "income_predictions.parquet").read()
frame = pd.read_parquet(io.BytesIO(data))
print("rows, columns:", frame.shape)
print("user_id in output:", "user_id" in frame.columns)
print(frame["Predicted_Income_Class"].describe().round(4).to_string())
