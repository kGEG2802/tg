from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import os
from oauth2client.service_account import ServiceAccountCredentials

scope = ["https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
drive_service = build("drive", "v3", credentials=creds)

def send_invoice_file(update, invoice_number):
    from google_sheet import sheet
    records = sheet.get_all_records()
    file_id = None
    for row in records:
        if str(row["Invoice №"]) == str(invoice_number):
            link = row["Link"]
            if "id=" in link:
                file_id = link.split("id=")[-1]
            elif "/d/" in link:
                file_id = link.split("/d/")[1].split("/")[0]
            break

    if not file_id:
        update.message.reply_text("❌ Посилання на файл не знайдено.")
        return

    request = drive_service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()

    fh.seek(0)
    update.message.reply_document(document=fh, filename=f"Invoice_{invoice_number}.pdf")
