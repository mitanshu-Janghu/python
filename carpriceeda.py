import pandas as pd
from data_profiling import ProfileReport

df = pd.read_csv("car_price_prediction_.csv")

profile = ProfileReport(
    df,
    title="Car Price Dataset Analysis",
    explorative=True
)

profile.to_file("car_price_report.html")