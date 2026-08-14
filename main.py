import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

weather_df = pd.read_csv("london_weather.csv")
weather_df.info()
weather_df.isna().sum()

weather_df["date"] = pd.to_datetime(weather_df["date"] , format="%Y%m%d")
weather_df["year"] = weather_df["date"].dt.year
weather_df["month"] = weather_df["date"].dt.month

weather_metrics = ['cloud_cover', 'sunshine', 'global_radiation', 'max_temp', 'mean_temp', 'min_temp', 'precipitation', 'pressure', 'snow_depth']
weather_per_month = weather_df.groupby(['year','month'], as_index=False)[weather_metrics].mean()

sns.lineplot(x="year", y="mean_temp", data=weather_per_month, errorbar=None)
plt.show()
sns.heatmap(weather_df.corr(), annot=True)
plt.show()

feature_sel = ['month', 'cloud_cover', 'sunshine', 'precipitation', 'pressure', 'global_radiation']
target_var = 'mean_temp'
weather_df = weather_df.dropna(subset=['mean_temp'])


X = weather_df[feature_sel]
y = weather_df[target_var]

X_train , X_test ,y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state= 40)

imputer = SimpleImputer(strategy="mean")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

for idx , depth in enumerate([1,2,10]):
    run_name = f"run_{idx}"
    with mlflow.start_run(run_name=run_name):
        lin_reg = LinearRegression().fit(X_train , y_train)
        dec_reg = DecisionTreeRegressor(random_state=40 , max_depth=depth).fit(X_train , y_train)
        ran_reg = RandomForestRegressor(random_state=40 , max_depth=depth).fit(X_train , y_train)

        mlflow.sklearn.log_model(lin_reg , "lin_reg")
        mlflow.sklearn.log_model(dec_reg, "dec_reg")
        mlflow.sklearn.log_model(ran_reg, "ran_reg")

        y_pred_lin_reg = lin_reg.predict(X_test)
        lin_reg_mse = mean_squared_error(y_test , y_pred_lin_reg)
        y_pred_dec_reg = dec_reg.predict(X_test)
        dec_reg_mse = mean_squared_error(y_test , y_pred_dec_reg)
        y_pred_ran_reg = ran_reg.predict(X_test)
        ran_reg_mse = mean_squared_error(y_test , y_pred_ran_reg)

        print("Linear Regression MSE:", lin_reg_mse)
        print("Decision Tree MSE:", dec_reg_mse)
        print("Random Forest MSE:", ran_reg_mse)

        print("\nLinear Regression Predictions:")
        print(y_pred_lin_reg)

        print("\nDecision Tree Predictions:")
        print(y_pred_dec_reg)

        print("\nRandom Forest Predictions:")
        print(y_pred_ran_reg)

        mlflow.log_param("max_depth", depth)
        mlflow.log_metric("mse_lr", lin_reg_mse)
        mlflow.log_metric("mse_tr", dec_reg_mse)
        mlflow.log_metric("mse_fr", ran_reg_mse)
        
experiment_results = mlflow.search_runs()
experiment_results
