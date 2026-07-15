
import numpy as np
import pandas as pd


SELECTED_PREDICTORS = [
    "city_pop",
    "customer_age",
    "customer_merchant_distance_km",
    "log_transaction_amount",
    "transaction_hour",
    "transaction_day_of_week",
    "transaction_month",
    "gender",
    "state",
    "category",
    "merchant",
    "job"
]


REQUIRED_RAW_COLUMNS = [
    "trans_date_trans_time",
    "dob",
    "amt",
    "lat",
    "long",
    "merch_lat",
    "merch_long",
    "city_pop",
    "gender",
    "state",
    "category",
    "merchant",
    "job"
]


def prepare_model_predictors(raw_transactions):
    """
    Convert raw transaction records into the predictors expected by
    the FraudGuard ML preprocessing pipeline.

    Parameters
    ----------
    raw_transactions : pandas.DataFrame
        Raw transaction records containing the required input columns.

    Returns
    -------
    pandas.DataFrame
        Engineered predictors in the required model input order.
    """

    if not isinstance(raw_transactions, pd.DataFrame):
        raise TypeError(
            "raw_transactions must be a pandas DataFrame."
        )

    missing_columns = [
        column
        for column in REQUIRED_RAW_COLUMNS
        if column not in raw_transactions.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required raw columns: "
            + ", ".join(missing_columns)
        )

    engineered = raw_transactions.copy()

    engineered["trans_date_trans_time"] = pd.to_datetime(
        engineered["trans_date_trans_time"],
        errors="raise"
    )

    engineered["dob"] = pd.to_datetime(
        engineered["dob"],
        errors="raise"
    )

    transaction_time = engineered["trans_date_trans_time"]
    date_of_birth = engineered["dob"]

    # Transaction-time features
    engineered["transaction_hour"] = (
        transaction_time.dt.hour
    )

    engineered["transaction_day_of_week"] = (
        transaction_time.dt.dayofweek
    )

    engineered["transaction_month"] = (
        transaction_time.dt.month
    )

    # Customer age at the transaction date
    birthday_not_reached = (
        (transaction_time.dt.month < date_of_birth.dt.month)
        |
        (
            (transaction_time.dt.month == date_of_birth.dt.month)
            &
            (transaction_time.dt.day < date_of_birth.dt.day)
        )
    )

    engineered["customer_age"] = (
        transaction_time.dt.year
        - date_of_birth.dt.year
        - birthday_not_reached.astype(int)
    )

    # Customer-to-merchant Haversine distance
    customer_latitude = np.radians(
        engineered["lat"].astype(float)
    )

    customer_longitude = np.radians(
        engineered["long"].astype(float)
    )

    merchant_latitude = np.radians(
        engineered["merch_lat"].astype(float)
    )

    merchant_longitude = np.radians(
        engineered["merch_long"].astype(float)
    )

    latitude_difference = (
        merchant_latitude - customer_latitude
    )

    longitude_difference = (
        merchant_longitude - customer_longitude
    )

    haversine_value = (
        np.sin(latitude_difference / 2) ** 2
        +
        np.cos(customer_latitude)
        * np.cos(merchant_latitude)
        * np.sin(longitude_difference / 2) ** 2
    )

    haversine_value = np.clip(
        haversine_value,
        0.0,
        1.0
    )

    earth_radius_km = 6371.0

    engineered["customer_merchant_distance_km"] = (
        2
        * earth_radius_km
        * np.arcsin(np.sqrt(haversine_value))
    )

    # Logarithmic transaction-amount transformation
    if (engineered["amt"].astype(float) < 0).any():
        raise ValueError(
            "Transaction amount cannot be negative."
        )

    engineered["log_transaction_amount"] = np.log1p(
        engineered["amt"].astype(float)
    )

    model_predictors = engineered[
        SELECTED_PREDICTORS
    ].copy()

    if model_predictors.isna().any().any():
        invalid_columns = model_predictors.columns[
            model_predictors.isna().any()
        ].tolist()

        raise ValueError(
            "Missing or invalid engineered values in: "
            + ", ".join(invalid_columns)
        )

    return model_predictors
