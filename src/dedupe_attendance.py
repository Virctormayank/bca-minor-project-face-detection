import pandas as pd, os

f = "attendance/attendance.csv"
if not os.path.exists(f):
    print("No file found.")
    raise SystemExit

df = pd.read_csv(f)
print("Before:", len(df), "rows")

df2 = df.drop_duplicates(subset=["Name","Date"], keep="first")
backup = f.replace(".csv", "_backup.csv")
df.to_csv(backup, index=False)
df2.to_csv(f, index=False)

print("After:", len(df2), "rows")
print("✅ Dedupe done. Backup saved to:", backup)
