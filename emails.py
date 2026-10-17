import pandas as pd
from ydata_profiling import ProfileReport

# Load dataset
df = pd.read_csv("emails.csv")

# Convert text column into NumPy array
x = df["text"].values

# Generate EDA report
profile = ProfileReport(
    df,
    title="Email Dataset Analysis",
    explorative=True
)
print(x)
# profile.to_file("emails.html")