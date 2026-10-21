import pandas as pd
from ydata_profiling import ProfileReport 

df = pd.read_csv("student_performance.csv")
profile = ProfileReport(df, title="student Analysis", explorative=True)
profile.to_file("student.html")