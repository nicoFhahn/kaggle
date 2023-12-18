import json
import optuna
import pandas as pd
from src import visualization, preprocessing
from src.objective import objective_fun_xgb, objective_fun_catboost, objective_fun_lightgbm, objective_fun_random_forest
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import median_absolute_error
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
train = pd.concat([pd.read_csv("S3_25/train.csv"), pd.read_csv("S3_25/Mineral_Dataset_Supplementary_Info.csv").rename(columns={"Unnamed: 0": "id"})])
test = pd.read_csv("S3_25/test.csv")
#visualization.plot_distplots(train.drop(columns=["id", "Hardness"]))
#visualization.plot_boxplots(train.drop(columns=["id", "Hardness"]))
#visualization.plot_heatmap(train.drop(columns=["id", "Hardness"]))
train_clean, test_clean = preprocessing.remove_outliers(train.drop(columns=["id", "Hardness"]), test.drop(columns=["id"]), contamination=0.1, method="nearest", n_neighbors=20)
# high correlation between
# density total & allelectrons total
# allelectrons average & r_vdw_element average
# allelectrons average & r cov element average
# allelectrons average & density average
# val e average & el eng chi average
# atomicweight average & density average
# ionenergy average & el neg chi average
# ionenergy average & zaratio average

train = pd.concat([train[["id", "Hardness"]], train_clean], axis=1)
test = pd.concat([test[["id"]], test_clean], axis=1)

train, test = preprocessing.replace_categorical(train, test, "Hardness")
X_train, y_train = train.drop(columns=["id", "Hardness"]), train[["Hardness"]]

# X_train, y_train = preprocessing.create_synthetic_data(X_train, y_train, method="ADASYN")
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=.22, random_state=623)
X_valid = test.drop(columns="id")
#X_train, X_test, X_valid = preprocessing.scale_data(X_train, X_test, test.drop(columns="id"))
#X_train, X_test, X_valid = preprocessing.create_pca(X_train, X_test, X_valid, criterion = 0.95)
cat_study = optuna.create_study(direction="minimize")
cat_study.optimize(lambda trial: objective_fun_catboost(trial, 42, False, 8, "neg_median_absolute_error", "MedianAbsoluteError", X_train, y_train, "Regression"), n_trials=100, n_jobs=6)
with open("S3_25/params_cat.json", "w") as f:
    json.dump(cat_study.best_params, f)
gbm_study = optuna.create_study(direction="minimize")
gbm_study.optimize(lambda trial: objective_fun_lightgbm(trial, 42, False, 4, "neg_median_absolute_error", "mean_absolute_error", X_train, y_train, "Regression"), n_trials=100, n_jobs=6)
with open("S3_25/params_gbm.json", "w") as f:
    json.dump(gbm_study.best_params, f)

rf_study = optuna.create_study(direction="minimize")
rf_study.optimize(lambda trial: objective_fun_random_forest(trial, 42, False, 4, "neg_median_absolute_error", X_train, y_train, "Regression"), n_trials=100, n_jobs=6)
with open("S3_25/params_rf.json", "w") as f:
    json.dump(rf_study.best_params, f)
xgb_study = optuna.create_study(direction="minimize")
xgb_study.optimize(lambda trial: objective_fun_xgb(trial, 42, False, 4, "neg_median_absolute_error", "mae", X_train, y_train, "Regression"), n_trials=100, n_jobs=6)
with open("S3_25/params_xgb.json", "w") as f:
    json.dump(xgb_study.best_params, f)

with open("S3_25/params_cat.json", "r") as f:
    cat_params = json.load(f)
with open("S3_25/params_gbm.json", "r") as f:
    gbm_params = json.load(f)
with open("S3_25/params_rf.json", "r") as f:
    rf_params = json.load(f)
with open("S3_25/params_xgb.json", "r") as f:
    xgb_params = json.load(f)

cat_model = CatBoostRegressor(**cat_params)
gbm_model = LGBMRegressor(**gbm_params)
rf_model = RandomForestRegressor(**rf_params)
xgb_model = XGBRegressor(**xgb_params)
cat_model.fit(X_train, y_train)
gbm_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)
cat_pred = cat_model.predict(X_test)
gbm_pred = gbm_model.predict(X_test)
rf_pred = rf_model.predict(X_test)
xgb_pred = xgb_model.predict(X_test)
mae_cat = median_absolute_error(y_test, cat_pred)
mae_gbm = median_absolute_error(y_test, gbm_pred)
mae_rf = median_absolute_error(y_test, rf_pred)
mae_xgb = median_absolute_error(y_test, xgb_pred)
pd.concat([test[["id"]], pd.Series(cat_model.predict(X_valid))], axis=1).rename(columns={0:"Hardness"}).to_csv("S3_25/predictions_cat.csv", index=False)
pd.concat([test[["id"]], pd.Series(gbm_model.predict(X_valid))], axis=1).rename(columns={0:"Hardness"}).to_csv("S3_25/predictions_gbm.csv", index=False)
pd.concat([test[["id"]], pd.Series(rf_model.predict(X_valid))], axis=1).rename(columns={0:"Hardness"}).to_csv("S3_25/predictions_rf.csv", index=False)
pd.concat([test[["id"]], pd.Series(xgb_model.predict(X_valid))], axis=1).rename(columns={0:"Hardness"}).to_csv("S3_25/predictions_xgb.csv", index=False)
ensemble = VotingRegressor(estimators = [('lgb', gbm_model), ('xgb', xgb_model), ('CB', cat_model), ('rf', rf_model)],
                            weights = [0.15, 0.30, 0.20, 0.35]   #Adjust weighting since XGB performs better in local environment
                            )
ensemble.fit(X_train, y_train)
mae_ensemble = median_absolute_error(y_test, ensemble.predict(X_test))
pd.concat([test[["id"]], pd.Series(ensemble.predict(X_valid))], axis=1).rename(columns={0:"Hardness"}).to_csv("S3_25/predictions_ensemble.csv", index=False)
https://www.kaggle.com/code/xpehutta/xgboost-optuna-outliers-omitted
# todo replace null values function