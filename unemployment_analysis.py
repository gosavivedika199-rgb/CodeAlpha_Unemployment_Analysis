import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Graph Style
# -----------------------------
plt.style.use("ggplot")

# Create images folder if it doesn't exist
os.makedirs("images", exist_ok=True)

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("Unemployment in India.csv")

print("First 5 Records:\n")
print(df.head())

print("\nDataset Information:")
print(df.info())

# -----------------------------
# Data Cleaning
# -----------------------------
df.columns = df.columns.str.strip()

df = df.dropna()

df["Date"] = df["Date"].str.strip()
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

df = df.reset_index(drop=True)

print("\nDataset after Cleaning:\n")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# -----------------------------
# Exploratory Data Analysis
# -----------------------------
print("\n========== DATA ANALYSIS ==========")

print("\nNumber of States:")
print(df["Region"].nunique())

print("\nStates:")
print(df["Region"].unique())

print("\nArea Types:")
print(df["Area"].unique())

print("\nAverage Unemployment Rate:")
print(round(df["Estimated Unemployment Rate (%)"].mean(), 2))

print("\nHighest Unemployment Rate:")
print(df["Estimated Unemployment Rate (%)"].max())

print("\nLowest Unemployment Rate:")
print(df["Estimated Unemployment Rate (%)"].min())

# -----------------------------
# Graph 1
# Average Unemployment Rate by State
# -----------------------------
state_avg = (
    df.groupby("Region")["Estimated Unemployment Rate (%)"]
    .mean()
    .sort_values()
)

plt.figure(figsize=(12, 6))
state_avg.plot(kind="bar", color="steelblue")

plt.title("Average Unemployment Rate by State")
plt.xlabel("State")
plt.ylabel("Average Unemployment Rate (%)")
plt.xticks(rotation=90)

plt.tight_layout()
plt.savefig("images/state_unemployment.png")
plt.show()
plt.close()

# -----------------------------
# Graph 2
# Monthly Trend
# -----------------------------
monthly = df.groupby("Date")["Estimated Unemployment Rate (%)"].mean()

plt.figure(figsize=(12, 6))
monthly.plot(marker="o", color="green")

plt.title("Monthly Average Unemployment Rate")
plt.xlabel("Date")
plt.ylabel("Average Unemployment Rate (%)")
plt.grid(True)

plt.tight_layout()
plt.savefig("images/monthly_trend.png")
plt.show()
plt.close()

# -----------------------------
# Graph 3
# Correlation Heatmap
# -----------------------------
plt.figure(figsize=(8, 6))

sns.heatmap(
    df.select_dtypes(include="number").corr(),
    annot=True,
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.savefig("images/correlation_heatmap.png")
plt.show()
plt.close()

# -----------------------------
# Final Insights
# -----------------------------
print("\n========== PROJECT INSIGHTS ==========")

print(f"\nTotal Records: {len(df)}")

print(f"\nNumber of States: {df['Region'].nunique()}")

print(
    f"\nAverage Unemployment Rate: "
    f"{df['Estimated Unemployment Rate (%)'].mean():.2f}%"
)

print(
    f"\nHighest Unemployment Rate: "
    f"{df['Estimated Unemployment Rate (%)'].max():.2f}%"
)

print(
    f"\nLowest Unemployment Rate: "
    f"{df['Estimated Unemployment Rate (%)'].min():.2f}%"
)

highest_state = (
    df.groupby("Region")["Estimated Unemployment Rate (%)"]
    .mean()
    .idxmax()
)

lowest_state = (
    df.groupby("Region")["Estimated Unemployment Rate (%)"]
    .mean()
    .idxmin()
)

print(f"\nState with Highest Average Unemployment: {highest_state}")

print(f"State with Lowest Average Unemployment: {lowest_state}")

