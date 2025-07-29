import gspread
import os
from oauth2client.service_account import ServiceAccountCredentials

SPREADSHEET_NAME = "Invoice Table"
SHEET_NAME = "Sheet1"

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
client = gspread.authorize(creds)
sheet = client.open(SPREADSHEET_NAME).worksheet(SHEET_NAME)

def handle_invoice_request(invoice_number):
    records = sheet.get_all_records()
    for row in records:
        if str(row["Invoice №"]) == str(invoice_number):
            return f"📄 Invoice {invoice_number}:
Date: {row['Date']}
Total: {row['Total value']}
Status: {row['Status']}
Link: {row['Link']}"
    return "Не знайдено такого інвойсу."

def update_invoice_status(invoice_number):
    cell = sheet.find(str(invoice_number))
    if cell:
        status_cell = sheet.cell(cell.row, 5)
        sheet.update_cell(cell.row, 5, "Paid")
        return f"✅ Статус інвойсу {invoice_number} оновлено на Paid."
    return "Інвойс не знайдено."

def list_by_status(status):
    records = sheet.get_all_records()
    matching = [f"{row['Invoice №']} — {row['Total value']}" for row in records if row["Status"].lower() == status.lower()]
    if not matching:
        return "Немає інвойсів з таким статусом."
    return f"📋 Інвойси зі статусом {status}:
" + "\n".join(matching)
