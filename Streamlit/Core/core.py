
import streamlit as st
import time
import joblib
import shap
import io
import pandas as pd
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.http import MediaIoBaseUpload
from streamlit_gsheets import GSheetsConnection

def connect_to_Google_drive(readonly=True):
    # Authenticate using service account info from secrets
    creds = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes= ["https://www.googleapis.com/auth/drive.readonly"] if readonly else ['googleapis.com']
    )

    service = build("drive", "v3", credentials=creds)

    return service

def stream_text(text):

    for word in text.split(" "):
        yield word + " "
        time.sleep(0.03)


def append_rows_and_overwrite(dataToAdd, sheet="Sheet1"):
        conn = st.connection("gsheets", type=GSheetsConnection)
        df = conn.read(worksheet=sheet, ttl=0)
        df = pd.concat( [df,dataToAdd.set_index("Feature").T])
        conn.update(worksheet="Sheet1", data=df)
        st.toast(
             """
             Data saved successfully. Thank you for contributing to the continuous
            improvement of our services.""",
            icon="😍",
            duration="long"
        )
        conn.close()


def load_pickle(file_id):
    service = connect_to_Google_drive()
    request = service.files().get_media(fileId=file_id)
    # 3. Create an in-memory binary stream
    file_handle = io.BytesIO()
    downloader = MediaIoBaseDownload(file_handle, request)
    # 4. Stream the data from Google Drive
    done = False
    while done is False:
        status, done = downloader.next_chunk()

    # 5. Reset the stream pointer to the beginning
    file_handle.seek(0)
    return file_handle

@st.cache_resource
def load_models():

    model_pickle = load_pickle("1SEwIiH-9XixrPNJU6jGTxbbI9rTHHBAN")
    processor_pickle = load_pickle("1ST1_vr7hmU7thKfqyyhsmUnOrKxNhfNA")
    background_pickle = load_pickle("1BVtn8pNHl7HP-ZQvXMji3MYcqEt9ERAd")

    processor = joblib.load(processor_pickle)
    model = joblib.load(model_pickle)
    def predict(x):
        return model.predict(x)
    background = joblib.load(background_pickle)
    explainer = shap.KernelExplainer(predict, data=background, feature_names=background.columns)
    return processor, model, explainer