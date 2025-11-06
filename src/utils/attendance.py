import csv
from datetime import datetime
import os

def mark_attendance(name, roll):
    date_today = datetime.now().strftime("%Y-%m-%d")
    time_now = datetime.now().strftime("%H:%M:%S")

    file_path = "Attendance.csv"

    # Agar file exist nahi hai → headers ke sath create karo
    file_exists = os.path.isfile(file_path)

    with open(file_path, "a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(["Name", "Roll Number", "Date", "Time"])

        # Check duplicate entry for today
        already_marked = False
        if file_exists:
            with open(file_path, "r") as read_file:
                for row in csv.reader(read_file):
                    if row[0] == name and row[2] == date_today:
                        already_marked = True
                        break

        if not already_marked:
            writer.writerow([name, roll, date_today, time_now])
            print(f"[✅ Attendance Marked] {name} ({roll}) at {time_now}")
        else:
            print("[ℹ️] Already marked today for:", name)
