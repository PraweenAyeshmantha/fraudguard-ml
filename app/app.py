
from datetime import date, datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from fraudguard_features import prepare_model_predictors


APP_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIRECTORY.parent

ARTIFACT_PATH = (
    PROJECT_ROOT
    / "models"
    / "fraudguard_ml_artifact.joblib"
)


@st.cache_resource
def load_fraudguard_artifact():
    return joblib.load(ARTIFACT_PATH)


st.set_page_config(
    page_title="FraudGuard ML",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ FraudGuard ML")
st.subheader("Credit Card Transaction Fraud Detection")

st.write(
    "Enter the transaction and customer details below to estimate "
    "the probability that the transaction is fraudulent."
)

try:
    artifact = load_fraudguard_artifact()

    preprocessor = artifact["preprocessor"]
    model = artifact["model"]
    classification_threshold = artifact[
        "classification_threshold"
    ]
    selected_predictors = artifact[
        "selected_predictors"
    ]

except Exception as error:
    st.error(
        "The FraudGuard model artefact could not be loaded."
    )
    st.exception(error)
    st.stop()


with st.form("fraud_prediction_form"):

    st.markdown("### Transaction details")

    transaction_date = st.date_input(
        "Transaction date",
        value=date.today()
    )

    transaction_time = st.time_input(
        "Transaction time",
        value=datetime.now().time().replace(
            second=0,
            microsecond=0
        )
    )

    transaction_amount = st.number_input(
        "Transaction amount",
        min_value=0.0,
        value=50.0,
        step=1.0
    )

    category = st.text_input(
        "Transaction category",
        value="shopping_net"
    )

    merchant = st.text_input(
        "Merchant",
        value="fraud_Test Merchant"
    )

    st.markdown("### Customer details")

    date_of_birth = st.date_input(
        "Customer date of birth",
        value=date(1990, 1, 1),
        min_value=date(1900, 1, 1),
        max_value=date.today()
    )

    gender = st.selectbox(
        "Gender",
        options=["M", "F"]
    )

    state = st.text_input(
        "State code",
        value="CA"
    )

    job = st.text_input(
        "Occupation",
        value="Software engineer"
    )

    city_population = st.number_input(
        "City population",
        min_value=0,
        value=100000,
        step=1000
    )

    st.markdown("### Geographical details")

    customer_latitude = st.number_input(
        "Customer latitude",
        min_value=-90.0,
        max_value=90.0,
        value=34.0522,
        format="%.6f"
    )

    customer_longitude = st.number_input(
        "Customer longitude",
        min_value=-180.0,
        max_value=180.0,
        value=-118.2437,
        format="%.6f"
    )

    merchant_latitude = st.number_input(
        "Merchant latitude",
        min_value=-90.0,
        max_value=90.0,
        value=34.0622,
        format="%.6f"
    )

    merchant_longitude = st.number_input(
        "Merchant longitude",
        min_value=-180.0,
        max_value=180.0,
        value=-118.2537,
        format="%.6f"
    )

    submitted = st.form_submit_button(
        "Analyse transaction"
    )


if submitted:

    try:
        transaction_datetime = datetime.combine(
            transaction_date,
            transaction_time
        )

        raw_transaction = pd.DataFrame(
            [
                {
                    "trans_date_trans_time": transaction_datetime,
                    "dob": date_of_birth,
                    "amt": transaction_amount,
                    "lat": customer_latitude,
                    "long": customer_longitude,
                    "merch_lat": merchant_latitude,
                    "merch_long": merchant_longitude,
                    "city_pop": city_population,
                    "gender": gender,
                    "state": state.strip(),
                    "category": category.strip(),
                    "merchant": merchant.strip(),
                    "job": job.strip()
                }
            ]
        )

        engineered_predictors = prepare_model_predictors(
            raw_transaction
        )

        engineered_predictors = engineered_predictors[
            selected_predictors
        ]

        processed_predictors = preprocessor.transform(
            engineered_predictors
        )

        fraud_probability = float(
            model.predict_proba(
                processed_predictors
            )[0, 1]
        )

        predicted_class = int(
            fraud_probability
            >= classification_threshold
        )

        st.markdown("## Prediction result")

        st.metric(
            "Estimated fraud probability",
            f"{fraud_probability:.2%}"
        )

        st.caption(
            "Classification threshold: "
            f"{classification_threshold:.4f}"
        )

        if predicted_class == 1:
            st.error(
                "Potential fraudulent transaction detected."
            )
        else:
            st.success(
                "Transaction classified as legitimate."
            )

        with st.expander(
            "View engineered model predictors"
        ):
            st.dataframe(
                engineered_predictors,
                use_container_width=True
            )

        st.info(
            "This result is a machine-learning risk estimate "
            "and should not be treated as a final banking decision."
        )

    except Exception as error:
        st.error(
            "The transaction could not be processed."
        )
        st.exception(error)
