import pandas as pd
from ydata_profiling import ProfileReport 

df = pd.read_csv("loan_approval_dataset.csv")
profile = ProfileReport(df, title="loan_approval_dataset", explorative=True)
profile.to_file("loan_approval.html")