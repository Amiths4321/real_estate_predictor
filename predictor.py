import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import joblib
import os

# ---- Load Real Data ----
df = pd.read_csv('housing.csv')

# ---- Clean Data ----
df = df.dropna()  # remove empty rows

# Convert text columns to numbers (e.g. city names)
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# ---- Define Features & Target ----
# Change 'price' to whatever your price column is named
target_column = 'price'
X = df.drop(columns=[target_column])
y = df[target_column]

# ---- Train/Test Split ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---- Train Random Forest ----
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("✅ Random Forest Model Trained!")

# ---- Evaluate ----
y_pred = model.predict(X_test)
print(f"\nMean Absolute Error : {mean_absolute_error(y_test, y_pred):,.2f}")
print(f"Accuracy (R2 Score) : {r2_score(y_test, y_pred):.2f}")

# ---- Feature Importance (What matters most?) ----
importance = pd.Series(model.feature_importances_, index=X.columns)
importance.sort_values().plot(kind='barh', figsize=(8,6), title='What Affects Price Most?')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()
print("📊 Feature importance chart saved!")

# ---- Save Model for reuse ----

# ---- Save Model for reuse ----


# Get the folder directory where this script itself lives
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_MODEL_PATH = os.path.join(SCRIPT_DIR, 'real_estate_model.pkl')

# Save it specifically into the same folder as the script
joblib.dump(model, TARGET_MODEL_PATH)
print(f"💾 Model safely saved as: {TARGET_MODEL_PATH}")