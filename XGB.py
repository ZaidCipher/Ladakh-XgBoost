import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import xgboost as xgb


def evaluate_predictions(model, X_data, y_true):
  """Evaluates model performance and prints metrics."""
  preds = model.predict(X_data)
  mae = mean_absolute_error(y_true, preds)
  rmse = np.sqrt(mean_squared_error(y_true, preds))
  print(f'Predicted PM2.5 (First Sample): {preds[0]:.2f} µg/m³')
  print(f'Mean Absolute Error (MAE) : {mae:.2f} µg/m³')
  print(f'Root Mean Squared Error (RMSE) : {rmse:.2f} µg/m³')
  return preds

df = pd.read_csv('log_9.csv')
df = df.dropna(axis=1, how='all')

df['PM2.5_ugm3'] = df['PM2.5_ugm3'].replace(0, np.nan).interpolate()

# ISO 8601 ORIGINAL TIME IN WEATHER STATION
df['Time'] = pd.to_datetime(df['Time'])
df['hour'] = df['Time'].dt.hour + (df['Time'].dt.minute / 60.0)

df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)

# Target: Predict PM2.5 at the next timestep (t+60)
df['target_PM2.5'] = df['PM2.5_ugm3'].shift(-60)

for lag in [1, 2, 3, 5]:
  df[f'PM2.5_lag_{lag}'] = df['PM2.5_ugm3'].shift(lag)
  df[f'Temp1_lag_{lag}'] = df['Temp1_C'].shift(lag)
  df[f'Humidity_lag_{lag}'] = df['Humidity_%'].shift(lag)

df['PM2.5_roll_mean_5'] = df['PM2.5_ugm3'].rolling(window=5).mean()
df['PM2.5_roll_std_5'] = df['PM2.5_ugm3'].rolling(window=5).std()

df = df.dropna()

feature_cols = [
    col for col in df.columns if col not in ['Time', 'hour', 'target_PM2.5']
]   

X = df[feature_cols]
y = df['target_PM2.5']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

model = xgb.XGBRegressor(
    n_estimators=150,
    learning_rate=0.03,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)
model.fit(X_train, y_train)


print('--- Test Set Performance ---')
preds = evaluate_predictions(model, X_test, y_test)

model_secondary = xgb.XGBRegressor(
    n_estimators=150,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=69,
)


model_secondary.fit(X_train, y_train)
model_secondary.save_model('xgboost_pm25_model.json')

plt.figure(figsize=(12, 5))
plt.plot(
    y_test.values[:200],
    label='Actual PM2.5 (Sensor)',
    color='#1f77b4',
    lw=1.5,
)
plt.plot(
    preds[:200],
    label='Predicted PM2.5 (XGBoost)',
    color='#ff7f0e',
    linestyle='--',
    lw=1.5,
)
plt.title(
    'XGBoost Short-Term Aerosol (PM2.5) Prediction (First 200 Steps)',
    fontsize=14,
    fontweight='bold',
)
plt.xlabel('Time Step (Minutes)', fontsize=12)
plt.ylabel('PM2.5 Concentration (µg/m³)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('pm25_xgboost_forecast.png', dpi=300)
plt.show()

plt.figure(figsize=(10, 6))
xgb.plot_importance(model, max_num_features=10, importance_type='weight')
plt.title('Top Microclimate Drivers of PM2.5 Predictions')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
plt.show()