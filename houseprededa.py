import pandas as pd
from ydata_profiling import ProfileReport 

df = pd.read_csv("housing1.csv")
profile = ProfileReport(df, title="House Price Dataset Analysis", explorative=True)
profile.to_file("house_report2.html")