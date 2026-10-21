import pandas as pd
from ydata_profiling import ProfileReport 

df = pd.read_csv("customer_churn_dataset-testing-master.csv")
profile = ProfileReport(df, title="customer_churn_dataset-testing-master", explorative=True)
profile.to_file("customer_churn_dataset-testing-master.html")