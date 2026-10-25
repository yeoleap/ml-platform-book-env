"""Step 2. Register the feature definitions and load the online store."""

from datetime import datetime, timezone
import importlib.util
import os

import pandas as pd
from feast import FeatureStore

import common

spec = importlib.util.spec_from_file_location(
    "features", os.path.join(common.FEATURE_REPO, "features.py")
)
definitions = importlib.util.module_from_spec(spec)
spec.loader.exec_module(definitions)

store = FeatureStore(repo_path=common.FEATURE_REPO)
store.apply([definitions.user, definitions.hours_source, definitions.hours,
             definitions.income_model_v1])
print("Registered:", [fv.name for fv in store.list_feature_views()],
      [fs.name for fs in store.list_feature_services()])

service = store.get_feature_service(common.FEATURE_SERVICE)
requests = pd.DataFrame({
    "user_id": [42],
    "event_timestamp": pd.to_datetime(["2025-05-22T00:00:00Z"]),
})
historical = store.get_historical_features(entity_df=requests, features=service).to_df()
print("Historical (2025-05-22):", historical["hours_per_week"].tolist())

store.materialize(
    datetime(2025, 4, 1, tzinfo=timezone.utc),
    datetime(2025, 7, 1, tzinfo=timezone.utc),
)
online = store.get_online_features(
    features=service, entity_rows=[{"user_id": 42}, {"user_id": 999}]
).to_dict()
print("Online latest:", online["hours_per_week"])
