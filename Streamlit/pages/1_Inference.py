
import streamlit as st
from Core.Model import PatientModel
from Core.Controller import diagnosis


def page():
    """
    Inference page 
    """
    st.set_page_config(
    page_title="Thyroid Cancer Prediction",
    layout="wide"
    )
    st.title("🤖 Explainable Computer-aided diangosis of postoperative recurrent lymph node metastasis of Thyroid cancer", text_alignment="left")
    st.divider()

    st.markdown("### 📥 Enter patient clinical information:")
    
    if "patient" not in st.session_state:
        st.session_state.patient = PatientModel()

    patient = st.session_state.patient

    col1, col2 = st.columns(2)

    with col1:

        patient.age = st.number_input(
            "Age at Diagnosis",
            min_value=18,
            max_value=100,
            value=patient.age,
            format="%d",
            step=1,
            help="Age of patient at diagnosis",
        )

        patient.sexe = st.selectbox(
            "Sex",
            ["Female", "Male"],
            index=["Female", "Male"].index(patient.sexe),
            help="Patient gender",
        )
        patient.alcool = st.selectbox(
            "Alcoholism",
            ["No", "Yes"],
            index=["No", "Yes"].index(patient.alcool),
            help="Alcoholic beverage consumption",
        )

        patient.tabac = st.selectbox(
            "Smoking",
            ["No", "Yes"],
            index=["No", "Yes"].index(patient.tabac),
            help="Tobacco Smoking"
        )

        patient.mcv_fam = st.selectbox(
            "Family history of cardiovascular disease",
            ["No", "Yes"],
            index=["No", "Yes"].index(patient.mcv_fam),
            help="Family history of cardiovascular disease"
        )
        patient.atcd_kc = st.selectbox(
            "Personal history of other cancers",
            ["No", "Yes"],
            index=["No", "Yes"].index(patient.atcd_kc),
            help="Personal history of other cancers"
        )
    
        patient.invasion = st.selectbox(
            "Vascular invasion",
            ["No", "Yes"],
            index=["No", "Yes"].index(patient.invasion),
            help="Vascular invasion"
        )
    
        patient.multifocalite = st.selectbox(
            "Multifocality",
            ["No", "Yes"],
            index=["No", "Yes"].index(patient.multifocalite),
            help="Multifocality"
        )

    with col2:

        patient.ete = st.selectbox(
            "Extrathyroidal extension",
            ["No", "Yes"],
            index=["No", "Yes"].index(patient.ete),
            help="Extension of the tumour beyond the thyroid capsule into surrounding tissues"
        )

        patient.t_stage = st.selectbox(
            "Tumour size (T)",
            ["T1", "T2", "T3", "T4"],
            index=["T1", "T2", "T3", "T4"].index(patient.t_stage),
            help="Primary tumour size and extent according to TNM classification"
        )

        patient.meta = st.selectbox(
            "Distance Metastasis (M)",
            ["M0", "M1"],
            index=["M0", "M1"].index(patient.meta),
            help="Presence or absence of distant metastasis"
        )

        patient.histology_type = st.selectbox(
            "Histology type",
            [
                "Papillary",
                "Medullary",
                "Anaplastic",
                "Follicular",
                "Poorly differentiated"
            ],
            index=[
                "Papillary",
                "Medullary",
                "Follicular",
                "Other"
            ].index(patient.histology_type),
            help="Histopathological subtype of thyroid carcinoma"
        )

        patient.ajcc8 = st.selectbox(
            "AJCC8",
            ["Stade I", "Stade II","Stade III","Stade IV"],
            index=["Stade I", "Stade II","Stade III","Stade IV"].index(patient.ajcc8),
            help="Cancer stage based on the AJCC 8th edition staging system"
        )

        patient.risque_ata = st.selectbox(#edit it
            "ATA Recurrence Risk",
            ["High", "Intermidate", "Low"],
            index=[
                "High",
                "Intermidate",
                "Low"
            ].index(patient.risque_ata),
            help="Risk category for recurrence according to American Thyroid Association guidelines"
        )

        patient.dose_iode = st.selectbox(
            "Cumulative Iodine Dose",
            ["0", "≤30", "30-150", ">150"],
            index=[
                "0",
                "≤30",
                "30-150",
                ">150"
            ].index(patient.dose_iode),
            help="Total cumulative radioactive iodine treatment dose received by the patient"
        )

        patient.levothyrox = st.slider(
            "Levothyrox Dose",
            min_value=0.0,
            max_value=300.0,
            value=patient.levothyrox,
            help="Daily levothyrox replacement or suppressive therapy dose in micrograms"
        )

    st.divider()
    is_clicked =st.button("Diagnosis", type="primary", width="stretch", icon="🩺", )
    st.divider()
    if(is_clicked):
        diagnosis(patient=patient)
    



if __name__ == "__main__":
    page()