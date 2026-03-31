# ============================================================
#   Air Quality Index (AQI) Predictor
#   College ML Project | Basic Level
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

print("=" * 55)
print("   AIR QUALITY INDEX PREDICTOR - ML PROJECT")
print("=" * 55)

# ============================================================
# STEP 1: CREATE DATASET (Simulated Indian City AQI Data)
# ============================================================
print("\n[STEP 1] Generating Dataset...")

np.random.seed(42)
n = 500

data = {
    "PM2_5":        np.random.uniform(10, 300, n),
    "PM10":         np.random.uniform(20, 400, n),
    "NO2":          np.random.uniform(5,  150, n),
    "SO2":          np.random.uniform(2,  80,  n),
    "CO":           np.random.uniform(0.1, 10, n),
    "Ozone":        np.random.uniform(10, 100, n),
    "Temperature":  np.random.uniform(10, 45,  n),
    "Humidity":     np.random.uniform(20, 95,  n),
    "Wind_Speed":   np.random.uniform(0,  20,  n),
}

# AQI formula (weighted sum + noise)
data["AQI"] = (
    0.35 * data["PM2_5"] +
    0.20 * data["PM10"]  +
    0.15 * data["NO2"]   +
    0.10 * data["SO2"]   +
    0.08 * data["CO"] * 10 +
    0.07 * data["Ozone"] +
    np.random.normal(0, 10, n)
).clip(0, 500).round(2)

df = pd.DataFrame(data)
df.to_csv("aqi_data.csv", index=False)
print(f"   Dataset created: {df.shape[0]} rows x {df.shape[1]} columns")
print(df.head(3))

# ============================================================
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
print("\n[STEP 2] Exploratory Data Analysis...")
print("\n--- Dataset Info ---")
print(df.describe().round(2))
print(f"\nMissing Values:\n{df.isnull().sum()}")

# AQI Category Distribution
def aqi_category(val):
    if val <= 50:   return "Good"
    elif val <= 100: return "Moderate"
    elif val <= 150: return "Unhealthy (Sensitive)"
    elif val <= 200: return "Unhealthy"
    elif val <= 300: return "Very Unhealthy"
    else:            return "Hazardous"

df["AQI_Category"] = df["AQI"].apply(aqi_category)
print(f"\nAQI Category Distribution:\n{df['AQI_Category'].value_counts()}")

# ============================================================
# STEP 3: VISUALIZATIONS
# ============================================================
print("\n[STEP 3] Generating Visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Air Quality Index - Exploratory Data Analysis", fontsize=16, fontweight="bold")

# Plot 1: AQI Distribution
axes[0,0].hist(df["AQI"], bins=30, color="steelblue", edgecolor="white")
axes[0,0].set_title("AQI Distribution")
axes[0,0].set_xlabel("AQI Value")
axes[0,0].set_ylabel("Frequency")

# Plot 2: AQI Category Pie Chart
cat_counts = df["AQI_Category"].value_counts()
colors = ["green","yellow","orange","red","purple","maroon"]
axes[0,1].pie(cat_counts, labels=cat_counts.index, autopct="%1.1f%%",
              colors=colors[:len(cat_counts)], startangle=140)
axes[0,1].set_title("AQI Category Distribution")

# Plot 3: PM2.5 vs AQI
axes[0,2].scatter(df["PM2_5"], df["AQI"], alpha=0.4, color="tomato")
axes[0,2].set_title("PM2.5 vs AQI")
axes[0,2].set_xlabel("PM2.5")
axes[0,2].set_ylabel("AQI")

# Plot 4: Correlation Heatmap
corr = df.drop(columns=["AQI_Category"]).corr()
sns.heatmap(corr, ax=axes[1,0], annot=True, fmt=".2f",
            cmap="coolwarm", linewidths=0.5, annot_kws={"size": 7})
axes[1,0].set_title("Feature Correlation Heatmap")

# Plot 5: Boxplot - AQI by Category
df_sorted = df.copy()
order = ["Good","Moderate","Unhealthy (Sensitive)","Unhealthy","Very Unhealthy","Hazardous"]
present = [c for c in order if c in df["AQI_Category"].unique()]
df_sorted["AQI_Category"] = pd.Categorical(df["AQI_Category"], categories=present, ordered=True)
df_sorted = df_sorted.sort_values("AQI_Category")
axes[1,1].boxplot(
    [df_sorted[df_sorted["AQI_Category"]==c]["AQI"].values for c in present],
    labels=[c[:8] for c in present]
)
axes[1,1].set_title("AQI by Category (Boxplot)")
axes[1,1].set_ylabel("AQI")
plt.setp(axes[1,1].xaxis.get_majorticklabels(), rotation=20, fontsize=8)

# Plot 6: Feature Importance Preview (Correlation with AQI)
feat_corr = df.drop(columns=["AQI_Category","AQI"]).corrwith(df["AQI"]).abs().sort_values(ascending=True)
axes[1,2].barh(feat_corr.index, feat_corr.values, color="mediumseagreen")
axes[1,2].set_title("Feature Correlation with AQI")
axes[1,2].set_xlabel("Correlation")

plt.tight_layout()
plt.savefig("eda_plots.png", dpi=150, bbox_inches="tight")
plt.show()
print("   EDA plots saved as eda_plots.png")

# ============================================================
# STEP 4: DATA PREPROCESSING
# ============================================================
print("\n[STEP 4] Data Preprocessing...")

X = df.drop(columns=["AQI", "AQI_Category"])
y = df["AQI"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"   Training samples : {X_train.shape[0]}")
print(f"   Testing  samples : {X_test.shape[0]}")

# ============================================================
# STEP 5: MODEL TRAINING (3 Models)
# ============================================================
print("\n[STEP 5] Training ML Models...")

models = {
    "Linear Regression":    LinearRegression(),
    "Decision Tree":        DecisionTreeRegressor(random_state=42),
    "Random Forest":        RandomForestRegressor(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train_sc, y_train)
    preds = model.predict(X_test_sc)
    results[name] = {
        "MAE":  round(mean_absolute_error(y_test, preds), 2),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, preds)), 2),
        "R2":   round(r2_score(y_test, preds), 4),
        "preds": preds
    }
    print(f"   ✅ {name} trained")

# ============================================================
# STEP 6: MODEL EVALUATION
# ============================================================
print("\n[STEP 6] Model Evaluation Results")
print("-" * 50)
print(f"{'Model':<25} {'MAE':>6} {'RMSE':>7} {'R2 Score':>10}")
print("-" * 50)
for name, res in results.items():
    print(f"{name:<25} {res['MAE']:>6} {res['RMSE']:>7} {res['R2']:>10}")
print("-" * 50)

best_model_name = max(results, key=lambda x: results[x]["R2"])
print(f"\n🏆 Best Model: {best_model_name} (R2 = {results[best_model_name]['R2']})")

# ============================================================
# STEP 7: RESULT VISUALIZATIONS
# ============================================================
print("\n[STEP 7] Generating Result Plots...")

fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
fig2.suptitle("Model Performance Comparison", fontsize=14, fontweight="bold")

model_names = list(results.keys())
short_names = ["Lin. Reg.", "Dec. Tree", "Rand. Forest"]

# Plot 1: R2 Score Comparison
r2_vals = [results[m]["R2"] for m in model_names]
bars = axes2[0].bar(short_names, r2_vals, color=["steelblue","orange","green"])
axes2[0].set_title("R² Score (Higher = Better)")
axes2[0].set_ylim(0, 1.1)
axes2[0].set_ylabel("R² Score")
for bar, val in zip(bars, r2_vals):
    axes2[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                  str(val), ha="center", fontweight="bold")

# Plot 2: MAE Comparison
mae_vals = [results[m]["MAE"] for m in model_names]
bars2 = axes2[1].bar(short_names, mae_vals, color=["steelblue","orange","green"])
axes2[1].set_title("MAE (Lower = Better)")
axes2[1].set_ylabel("Mean Absolute Error")
for bar, val in zip(bars2, mae_vals):
    axes2[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                  str(val), ha="center", fontweight="bold")

# Plot 3: Actual vs Predicted (Best Model)
best_preds = results[best_model_name]["preds"]
axes2[2].scatter(y_test, best_preds, alpha=0.4, color="tomato")
axes2[2].plot([y_test.min(), y_test.max()],
              [y_test.min(), y_test.max()], "k--", lw=2)
axes2[2].set_title(f"Actual vs Predicted\n({best_model_name})")
axes2[2].set_xlabel("Actual AQI")
axes2[2].set_ylabel("Predicted AQI")

plt.tight_layout()
plt.savefig("model_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("   Result plots saved as model_results.png")

# ============================================================
# STEP 8: LIVE PREDICTION DEMO
# ============================================================
print("\n[STEP 8] Live Prediction Demo")
print("-" * 40)

sample = pd.DataFrame([{
    "PM2_5": 120, "PM10": 180, "NO2": 80,
    "SO2": 30,    "CO": 5,     "Ozone": 60,
    "Temperature": 35, "Humidity": 70, "Wind_Speed": 5
}])

best = list(models.values())[list(models.keys()).index(best_model_name)]
sample_sc = scaler.transform(sample)
predicted_aqi = best.predict(sample_sc)[0]
category = aqi_category(predicted_aqi)

print(f"   Input  → PM2.5=120, PM10=180, NO2=80, Temp=35°C")
print(f"   Output → Predicted AQI : {predicted_aqi:.1f}")
print(f"            AQI Category  : {category}")

print("\n" + "=" * 55)
print("   PROJECT COMPLETE! ✅")
print("   Files Generated:")
print("   📊 eda_plots.png")
print("   📈 model_results.png")
print("   📄 aqi_data.csv")
print("=" * 55)
