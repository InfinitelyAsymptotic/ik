from datetime import timedelta
from feast import Entity, FileSource, FeatureView, Field, ValueType

transaction = Entity(
    name="TransactionID",
    join_keys=["TransactionID"],
    value_type=ValueType.INT64,
    description="Transaction ID"
)

transaction_source = FileSource(
    path="s3://fraud-demo-data-yourname/raw/train_transaction.csv",  # updated by config
    event_timestamp_column="TransactionDT",
    created_timestamp_column=None
)

transaction_features = FeatureView(
    name="transaction_features",
    entities=["TransactionID"],
    ttl=timedelta(days=1),
    schema=[
        Field(name="TransactionAmt", dtype=ValueType.FLOAT),
        Field(name="ProductCD", dtype=ValueType.STRING),
        Field(name="card1", dtype=ValueType.INT64),
        Field(name="card2", dtype=ValueType.INT64),
    ],
    online=True,
    source=transaction_source,
    tags={"team": "fraud", "priority": "high"}
)