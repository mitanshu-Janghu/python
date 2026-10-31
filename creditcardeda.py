import pandas as pd
from data_profiling import ProfileReport

df = pd.read_csv("creditcard.csv")

profile = ProfileReport(
    df,
    title="credit card",
    explorative=True
)

profile.to_file("creditcard.html")