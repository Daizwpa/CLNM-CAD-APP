
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


def append_rows_and_overwrite(file_id, new_data_dict, service):
    # 1. Export Google Sheet as an Excel file binary stream
    request = service.files().export_media(
        fileId=file_id,
        mimeType='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
    
    file_stream = io.BytesIO()
    downloader = MediaIoBaseDownload(file_stream, request)
    
    done = False
    while not done:
        _, done = downloader.next_chunk()
    
    # 2. Read into pandas DataFrame
    file_stream.seek(0)
    df_existing = pd.read_excel(file_stream, engine='openpyxl')

    # 3. Append your new data rows
    df_new = pd.DataFrame(new_data_dict)
    df_updated = pd.concat([df_existing, df_new], ignore_index=True)

    # 4. Save the updated DataFrame back into memory buffer
    output_stream = io.BytesIO()
    with pd.ExcelWriter(output_stream, engine='openpyxl') as writer:
        df_updated.to_excel(writer, index=False)
    output_stream.seek(0)

    # 5. Overwrite the file on Google Drive
    media = MediaIoBaseUpload(
        output_stream, 
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 
        resumable=True
    )
    
    # NOTE: update() converts it back into a native Google Sheet automatically
    updated_file = service.files().update(
        fileId=file_id,
        media_body=media
    ).execute()
    
    print(f"Successfully updated Google Sheet ID: {file_id}")
    return updated_file

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