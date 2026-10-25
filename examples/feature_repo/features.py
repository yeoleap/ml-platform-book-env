from datetime import timedelta

from feast import Entity, FeatureService, FeatureView, Field, FileSource, ValueType
from feast.types import Float64, String

user = Entity(name="user", join_keys=["user_id"], value_type=ValueType.INT64)

# No s3_endpoint_override here: the endpoint is taken from AWS_ENDPOINT_URL,
# so the same definition works from the laptop and inside the cluster.
hours_source = FileSource(
    name="hours_source",
    path="s3://feature-data-sets/hours.parquet",
    timestamp_field="event_timestamp",
)

hours = FeatureView(
    name="hours",
    entities=[user],
    ttl=timedelta(days=365),
    schema=[
        Field(name="hours_per_week", dtype=Float64),
        Field(name="workclass", dtype=String),
    ],
    source=hours_source,
)

income_model_v1 = FeatureService(name="income_model_v1", features=[hours])
