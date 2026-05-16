from Core.Model import PatientModel
import streamlit as st
from Core.core import stream_text, load_models, append_rows_and_overwrite, connect_to_Google_drive
import time
import pandas as pd
import shap
import matplotlib.pyplot as plt

@st.fragment
def save_button_container(df):
    df["user"] = st.user.name
    with st.container(horizontal=True, horizontal_alignment="right"):
                if st.button("Save"):
                    append_rows_and_overwrite(df)

def diagnosis(patient: PatientModel):
    processor, model, explainer = load_models()

    with st.status("Running AI analysis...", expanded=True ) as status:

        st.write("Loading preprocessing pipeline...")
        time.sleep(0.5)

        input_data = patient.to_model_input()

        st.write("Transforming patient clinical data...")
        time.sleep(0.5)

        transformed_data = processor.pipeline_[
            "coding"
        ].transform(input_data)

        st.write("Generating prediction...")
        time.sleep(0.5)

        prediction = model.predict(
            transformed_data
        )[0]


        st.write("Computing confidence score...")
        time.sleep(0.5)

        probability = model.predict_proba(
            transformed_data
        )[0][1]

        affected_probability = probability * 100

        unaffected_probability = (
            1 - probability
        ) * 100

        status.update(
            label="Analysis completed successfully",
            state="complete"
        )

    with st.chat_message("assistant"):

        st.write(
            
                """
                Hello 👋
                The clinical analysis has been completed successfully.
                """
            
        )

        st.markdown("""
        ## Postoperative Recurrent Lymph Node Metastasis Prediction Report
        """)

        st.write(
                f"""
                A predictive analysis was generated for a patient
                aged {patient.age} years using the trained
                machine learning model.
                """
            
        )

        st.divider()

        if prediction == 0:

            st.write(
                    f"""
                    The patient is currently classified as
                    LOW RISK for postoperative recurrent
                    lymph node metastasis.
                    """
                
            )

            st.success(
                f"""
                Low Risk Prediction.
                - Confidence: {unaffected_probability:.2f}%
                """,
                icon="✅"
            )

        else:

            st.write(
                
                    f"""
                    The patient is currently classified as
                    HIGH RISK for postoperative recurrent
                    lymph node metastasis.
                    """
                
            )

            st.error(
                f"""
                High Risk Prediction

                - Confidence: {affected_probability:.2f}%
                """,
                icon="🚨"
            )

        st.divider()


        st.subheader("Prediction Probabilities")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                label="High-Risk Probability",
                value=f"{affected_probability:.2f}%"
            )

        with col2:

            st.metric(
                label="Low-Risk Probability",
                value=f"{unaffected_probability:.2f}%"
            )

        chart_data = pd.DataFrame({

            "Outcome": [
                "Low Risk",
                "High Risk"
            ],

            "Probability": [
                unaffected_probability,
                affected_probability
            ]
        })

        st.subheader("Prediction Confidence")

        st.bar_chart(
            data=chart_data,
            x="Outcome",
            y="Probability",
            width="stretch"
        )

        with st.expander(
            "Patient Clinical Summary"
        ):

            summary = pd.DataFrame({
            
                "Feature": [
                
                    "Age at Diagnosis",
                    "Sex",
                    "Alcoholism",
                    "Smoking",
                    "Family History of Cardiovascular Disease",
                    "Personal History of Other Cancers",
                    "Vascular Invasion",
                    "Multifocality",
                    "Extrathyroidal Extension",
                    "Tumour Stage (T)",
                    "Distant Metastasis (M)",
                    "Histology Type",
                    "AJCC8 Stage",
                    "ATA Recurrence Risk",
                    "Cumulative Iodine Dose",
                    "Levothyrox Dose"
                ],

                "Value": [
                    patient.age,
                    patient.sexe,
                    patient.alcool,
                    patient.tabac,
                    patient.mcv_fam,
                    patient.atcd_kc,
                    patient.invasion,
                    patient.multifocalite,
                    patient.ete,
                    patient.t_stage,
                    patient.meta,
                    patient.histology_type,
                    patient.ajcc8,
                    patient.risque_ata,
                    patient.dose_iode,
                    f"{patient.levothyrox:.1f} µg"
                ]
            })
            summary["Value"] = summary["Value"].astype(str)
            st.dataframe(
                summary,
                width="stretch",
                hide_index=True
            )
            save_button_container(summary)


        se = explainer(transformed_data)


        st.divider()

        st.subheader("Model Explainability (SHAP)")

        fig, ax = plt.subplots()

        shap.plots.waterfall(shap_values=se[0], max_display=25, show=False)

        st.pyplot(fig)
        

        st.caption(
            """
            This prediction was generated using a machine learning
            decision-support system and should assist—not replace—
            clinical judgement.
            """
        )



