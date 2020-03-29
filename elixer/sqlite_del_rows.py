"""
edit as needed,
rarely used;  simple remove rows
"""

import sqlite3

conn = sqlite3.connect("elixer_reports_20001.db")
cursor = conn.cursor()
SQL = "SELECT detectid FROM report where detectid < 2000114958;"
cursor.execute(SQL)
dets = cursor.fetchall()
cursor.close()
print(f"{len(dets)} detections to remove")
i = input("Continue with remove? (y/n)")
if len(i) > 0 and i.upper() != "Y":
    print("Cancelling.")
else:
    print("Removing ...")
    cursor = conn.cursor()
    SQL = "DELETE FROM report where detectid < 2000114958;"
    cursor.execute(SQL)
    conn.commit()
    cursor.close()

conn.close()
