from google_sheet import client

def log_query(user_id, message):
    try:
        log_sheet = client.open("Invoice Table").worksheet("Logs")
        log_sheet.append_row([str(user_id), message])
    except:
        pass
