import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import optuna
import xgboost as xgb

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)
from sklearn.model_selection import TimeSeriesSplit


df = pd.read_csv('log_9.csv')
df = df.dropna(axis=1, how='all')

df['PM2.5_ugm3'] = (
    df['PM2.5_ugm3']
    .replace(0, np.nan)
    .interpolate()
)

# Convert time
df['Time'] = pd.to_datetime(df['Time'])

# Cyclic time features
df['hour'] = (
    df['Time'].dt.hour +
    df['Time'].dt.minute / 60.0
)

df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)




# Predict PM2.5 one hour into the future
df['target_PM2.5'] = df['PM2.5_ugm3'].shift(-60)



for lag in [1, 2, 3, 5]:

    df[f'PM2.5_lag_{lag}'] = (
        df['PM2.5_ugm3'].shift(lag)
    )

    df[f'Temp1_lag_{lag}'] = (
        df['Temp1_C'].shift(lag)
    )

    df[f'Humidity_lag_{lag}'] = (
        df['Humidity_%'].shift(lag)
    )




df['PM2.5_roll_mean_5'] = (
    df['PM2.5_ugm3']
    .rolling(window=5)
    .mean()
)

df['PM2.5_roll_std_5'] = (
    df['PM2.5_ugm3']
    .rolling(window=5)
    .std()
)


# Remove NaNs caused by lags / rolling / target shift
df = df.dropna()


feature_cols = [
    col for col in df.columns
    if col not in ['Time', 'hour', 'target_PM2.5']
]

X = df[feature_cols]
y = df['target_PM2.5']




# IMPORTANT:
# shuffle=False because this is time-series data

split_index = int(len(X) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


def objective(trial):

    params = {

        # Number of trees
        'n_estimators': trial.suggest_int(
            'n_estimators',
            300,
            2000
        ),

        # Learning rate
        'learning_rate': trial.suggest_float(
            'learning_rate',
            0.005,
            0.1,
            log=True
        ),

        # Tree depth
        'max_depth': trial.suggest_int(
            'max_depth',
            2,
            12
        ),

        # Minimum child weight
        'min_child_weight': trial.suggest_int(
            'min_child_weight',
            1,
            15
        ),

        # Row subsampling
        'subsample': trial.suggest_float(
            'subsample',
            0.5,
            1.0
        ),

        # Feature subsampling
        'colsample_bytree': trial.suggest_float(
            'colsample_bytree',
            0.5,
            1.0
        ),

        # L1 regularization
        'reg_alpha': trial.suggest_float(
            'reg_alpha',
            1e-8,
            10.0,
            log=True
        ),

        # L2 regularization
        'reg_lambda': trial.suggest_float(
            'reg_lambda',
            1e-8,
            10.0,
            log=True
        ),

        'objective': 'reg:squarederror',

        'eval_metric': 'rmse',

        'random_state': 67,

        'n_jobs': -1
    }


    # Time-series cross validation
    tscv = TimeSeriesSplit(n_splits=3)

    scores = []


    for train_idx, val_idx in tscv.split(X_train):

        X_tr = X_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]

        y_tr = y_train.iloc[train_idx]
        y_val = y_train.iloc[val_idx]


        model = xgb.XGBRegressor(**params)


        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False
        )


        predictions = model.predict(X_val)


        # Optuna minimizes this
        mae = mean_absolute_error(
            y_val,
            predictions
        )

        scores.append(mae)


    return np.mean(scores)


# ============================================================
# 8. RUN OPTUNA
# ============================================================

study = optuna.create_study(
    direction='minimize',
    study_name='PM25_XGBoost_Optimization'
)

study.optimize(
    objective,
    n_trials=50,
    show_progress_bar=True
)


# ============================================================
# 9. BEST PARAMETERS
# ============================================================

print("\n==============================")
print("OPTUNA RESULTS")
print("==============================")

print(
    f"Best CV MAE: "
    f"{study.best_value:.4f}"
)

print("\nBest parameters:")

for key, value in study.best_params.items():
    print(f"{key}: {value}")


best_params = study.best_params.copy()

best_params.update({
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'random_state': 67,
    'n_jobs': -1
})


model = xgb.XGBRegressor(
    **best_params
)


model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    verbose=100
)



model.save_model(
    'pm25_xgboost_optuna.json'
)



preds = model.predict(X_test)




mae = mean_absolute_error(
    y_test,
    preds
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        preds
    )
)

mape = mean_absolute_percentage_error(
    y_test,
    preds
)

r2 = r2_score(
    y_test,
    preds
)

accuracy_pct = (
    1 - mape
) * 100


print("\n==============================")
print("FINAL MODEL PERFORMANCE")
print("==============================")

print(
    f"MAE  : {mae:.2f} µg/m³"
)

print(
    f"RMSE : {rmse:.2f} µg/m³"
)

print(
    f"R²   : {r2:.4f}"
)

print(
    f"MAPE : {mape * 100:.2f}%"
)

print(
    f"MAPE-based percentage : "
    f"{accuracy_pct:.2f}%"
)



plt.figure(figsize=(12, 5))

plt.plot(
    y_test.values,
    label='Actual PM2.5 (Sensor)',
    lw=1.5
)

plt.plot(
    preds,
    label='Predicted PM2.5 (XGBoost)',
    linestyle='--',
    lw=1.5
)

plt.title(
    'Optuna-Optimized XGBoost PM2.5 Prediction',
    fontsize=14,
    fontweight='bold'
)

plt.xlabel(
    'Time Step (Minutes)',
    fontsize=12
)

plt.ylabel(
    'PM2.5 Concentration (µg/m³)',
    fontsize=12
)

plt.legend(fontsize=11)

plt.grid(
    True,
    linestyle=':',
    alpha=0.6
)

plt.tight_layout()

plt.savefig(
    'pm25_xgboost_optuna_forecast.png',
    dpi=300
)

plt.show()




plt.figure(figsize=(10, 6))

xgb.plot_importance(
    model,
    max_num_features=10,
    importance_type='weight'
)

plt.title(
    'Top Microclimate Drivers of PM2.5 Predictions'
)

plt.tight_layout()

plt.savefig(
    'feature_importance_optuna.png',
    dpi=300
)

plt.show()