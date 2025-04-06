import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st  # For interactive dashboard

# Set better visualization style
#sns.set_style("whitegrid")
#plt.style.use("ggplot")  # Modern color theme

# Load dataset
df = pd.read_csv("U.S._Chronic_Disease_Indicators.csv")
# Basic Info
print("\n shape of data")
print(df.shape)
# Display column names
print("\n Columns in Dataset:")
print(df.columns)
# Display dataset info
print("\nData types & non-null counts")
print(df.info())
print("\n Stats summary (mean, std, min, max")
print(df.describe())
# Display first few rows
print("\n First 5 rows of dataset:")
print(df.head())
# Check Missing Values
print("\n Check Missing Values",df.isnull().sum())

#Drop unnecessary columns
columns_to_drop = [
    "Response", "ResponseID", "StratificationCategory2", "Stratification2",
    "StratificationCategory3", "Stratification3", "StratificationCategoryID2",
    "StratificationID2", "StratificationCategoryID3", "StratificationID3",
    "DataValueFootnote", "DataValueFootnoteSymbol", "Geolocation"
]
df_cleaned = df.drop(columns=columns_to_drop, errors='ignore')

# Remove duplicate rows
df_cleaned = df_cleaned.drop_duplicates()

# Print new shape & remaining columns
print("\n New DataFrame Shape:", df_cleaned.shape)
print(" Remaining Columns:", df_cleaned.columns)

# Display dataset info after cleaning
print("\n Dataset Info After Cleaning:")
df_cleaned.info()

# Summary statistics after cleaning
print("\n Summary Statistics After Cleaning:")
print(df_cleaned.describe())

# Check missing values after cleaning
print("\n Missing Values After Cleaning:")
print(df_cleaned.isnull().sum())

# Drop rows with missing values
df_cleaned.dropna(inplace=True)

# Drop columns where more than 50% values are missing
df_cleaned = df_cleaned.dropna(axis=1, thresh=int(0.5 * len(df_cleaned)))

# Check missing values after cleaning
print("\n Missing Values After Cleaning:")
print(df_cleaned.isnull().sum())


# Convert categorical variables into numerical values
categorical_cols = df_cleaned.select_dtypes(include='object').columns
print("\n Categorical Columns:", categorical_cols)
df_cleaned = pd.get_dummies(df_cleaned, columns=categorical_cols, drop_first=True)

"""#  Optimized Histogram
plt.figure(figsize=(15, 8), dpi=100)  # High DPI for clarity
sampled_df = df_cleaned.sample(n=5000, random_state=42)

# Select numerical columns
num_cols = df_cleaned.select_dtypes(include=['float64', 'int64']).columns[:5]

# Create histograms
sampled_df[num_cols].hist(
    bins=30,
    color='skyblue',
    edgecolor='black',
    figsize=(15, 8),
    layout=(2, 3)  # 2 rows, 3 columns layout for better spacing
)

plt.suptitle("Optimized Histogram of Numerical Features", fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit title
plt.show()
"""
# Seaborn-based Histograms for Numerical Features
plt.figure(figsize=(15, 10), dpi=100)
sampled_df = df_cleaned.sample(n=5000, random_state=42)
num_cols = df_cleaned.select_dtypes(include=['float64', 'int64']).columns[:5]

for i, col in enumerate(num_cols, 1):
    plt.subplot(2, 3, i)
    sns.histplot(sampled_df[col], bins=30, kde=True, color='skyblue', edgecolor='black')
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
plt.suptitle("Seaborn Histograms of Numerical Features", fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

#  Improved Line Plot for Disease Trends
plt.figure(figsize=(16, 8)) 

sns.lineplot(
    x="YearStart", 
    y="DataValue", 
    hue="Topic", 
    data=df, 
    palette="coolwarm", 
    linewidth=2
)

plt.title("Chronic Disease Trend Over Time", fontsize=16, fontweight='bold')
plt.xlabel("Year")
plt.ylabel("Data Value")
plt.xticks(rotation=45)


plt.legend(
    title="Disease Type", 
    bbox_to_anchor=(1.25, 1), 
    loc='upper left',
    borderaxespad=0.5,
    fontsize='small',  
    title_fontsize='medium'
)

plt.tight_layout()  # Auto adjust for spacing
plt.show()
#State-wise Analysis - Diabetes Cases
"""plt.figure(figsize=(14, 7))  
top_states = (
    df[df["Topic"] == "Diabetes"]
    .groupby("LocationDesc")["DataValue"]
    .mean()
    .nlargest(10)
)

top_states.plot(
    kind='bar', 
    color='tomato', 
    edgecolor='black'
)

plt.title("Top 10 States with Highest Average Diabetes Cases", fontsize=16, fontweight='bold')
plt.xlabel("State", fontsize=12)
plt.ylabel("Average Data Value", fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()  # Adjusts spacing to prevent overlap
plt.show()
"""
# Seaborn Barplot - State-wise Diabetes Cases
plt.figure(figsize=(14, 7))
top_states = (
    df[df["Topic"] == "Diabetes"]
    .groupby("LocationDesc")["DataValue"]
    .mean()
    .nlargest(10)
).reset_index()

sns.barplot(x="LocationDesc", y="DataValue", data=top_states, palette="Reds_r", edgecolor='black')
plt.title("Top 10 States with Highest Average Diabetes Cases", fontsize=16, fontweight='bold')
plt.xlabel("State", fontsize=12)
plt.ylabel("Average Data Value", fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# Distribution of Chronic Disease Records by Year
plt.figure(figsize=(12, 6))
sns.countplot(
    x="YearStart",
    data=df,
    palette="viridis",
    order=sorted(df["YearStart"].unique())
)
plt.title("Distribution of Chronic Disease Records by Year", fontsize=16, fontweight='bold')
plt.xlabel("Year")
plt.ylabel("Number of Records")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


#  Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df_cleaned.corr(), annot=False, cmap="coolwarm", linewidths=0.5, cbar=True)
plt.title(" Correlation Heatmap", fontsize=16, fontweight='bold')
plt.show()

# Improved Boxplot for Outlier Detection
plt.figure(figsize=(15, 10))
for i, col in enumerate(num_cols[:6], 1):
    plt.subplot(2, 3, i)
    sns.boxplot(y=df_cleaned[col], color='cyan')
    plt.title(f"Outlier Detection - {col}")
plt.tight_layout()
plt.show()

# Interactive Dashboard using Streamlit
def main():
    st.title(" Chronic Disease Analysis Dashboard")
    
    if st.checkbox("Show Raw Data"):
        st.write(df_cleaned.head())
    
    st.subheader("Disease Trends Over Time")
    disease_options = df["Topic"].unique()
    selected_disease = st.selectbox("Select a Disease", disease_options)
    filtered_df = df[df["Topic"] == selected_disease]
    st.line_chart(filtered_df.set_index("YearStart")["DataValue"])
    
    st.subheader("State-wise Analysis")
    top_states = df[df["Topic"] == selected_disease].groupby("LocationDesc")["DataValue"].mean().nlargest(10)
    st.bar_chart(top_states)

if __name__ == "__main__":
    main()

print("\nEDA Completed! Data is now cleaned and visualized successfully.")
