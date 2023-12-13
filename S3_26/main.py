import jenkspy
import pandas as pd
import seaborn as sns
import xgboost as xgb
import numpy as np
import scipy.stats as stats
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from imblearn.over_sampling import SMOTE, ADASYN
from collections import Counter
from tqdm import tqdm

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
# load the training data
train = pd.read_csv("S3_26/train.csv")
df = train.copy()
# turn stage into category
train["Stage"] = train.Stage.astype("category")
# add dummy variables
train = pd.concat([train, pd.get_dummies(train.Drug, prefix="Drug")], axis=1).drop(columns="Drug")
train = pd.concat([train, pd.get_dummies(train.Sex, prefix="Sex")], axis=1).drop(columns="Sex")
train = pd.concat([train, pd.get_dummies(train.Ascites, prefix="Ascites")], axis=1).drop(columns="Ascites")
train = pd.concat([train, pd.get_dummies(train.Hepatomegaly, prefix="Hepatomegaly")], axis=1).drop(columns="Hepatomegaly")
train = pd.concat([train, pd.get_dummies(train.Spiders, prefix="Spiders")], axis=1).drop(columns="Spiders")
train = pd.concat([train, pd.get_dummies(train.Edema, prefix="Edema")], axis=1).drop(columns="Edema")
train = pd.concat([train, pd.get_dummies(train.Stage, prefix="Stage")], axis=1).drop(columns="Stage")
# encode status
le = LabelEncoder()
# split into X & y
X, y = train.drop(columns=["id", "Status"]), le.fit_transform(train.Status)
# split into train & test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=420)
# default models and proba function
models = [
    (BernoulliNB(), "proba"), (ExtraTreesClassifier(random_state=42), "proba"),
    (RandomForestClassifier(random_state=42), "proba"), (KNeighborsClassifier(), "proba"),
    (LinearDiscriminantAnalysis(), "proba"), (QuadraticDiscriminantAnalysis(), "proba"),
    (LinearSVC(multi_class="crammer_singer"), "proba_lr"), (LogisticRegression(multi_class="multinomial"), "proba"),
    (RidgeClassifier(), "proba_lr"), (xgb.XGBClassifier(objective='multi:softmax', seed=42), "proba")
]
# calculate log loss for default models
log_losses = [
    log_loss(y_test, models[i][0].fit(X_train, y_train).predict_proba(X_test))
    if models[i][1] == "proba" else
    log_loss(y_test, models[i][0].fit(X_train, y_train)._predict_proba_lr(X_test))
    for i in tqdm(range(len(models)))
]
# replace outliers with max value
for col in ["SGOT", "Tryglicerides", "Prothrombin"]:
    fit = LocalOutlierFactor(n_neighbors=20, contamination=0.1).fit_predict(X_train[[col]])
    temp_df = pd.concat([X_train[[col]].reset_index(drop=True), pd.Series(fit)], axis=1)
    min_value = temp_df[temp_df[0] != -1 ][col].min()
    max_value = temp_df[temp_df[0] != -1 ][col].max()
    X_train.loc[X_train[col] <= min_value, col] = min_value
    X_train.loc[X_train[col] >= max_value, col] = max_value
    X_test.loc[X_test[col] <= min_value, col] = min_value
    X_test.loc[X_test[col] >= max_value, col] = max_value
#jenks_dict = {
#    "Bilirubin": 3,
#    "Cholesterol": 2,
#    "Copper": 3,
#    "Alk_Phos": 3
#}
#for key, value in jenks_dict.items():
#    breaks = jenkspy.jenks_breaks(X_train[key], n_classes=value)
#    cat = pd.cut(X_train[key], bins=breaks).astype("interval")
#    X_train[key + "_cat"] = cat
#    cat = pd.cut(X_test[key], bins=breaks).astype("interval")
#    X_test[key + "_cat"] = cat
# calculate log losses again
log_losses_2 = [
    log_loss(y_test, models[i][0].fit(X_train, y_train).predict_proba(X_test))
    if models[i][1] == "proba" else
    log_loss(y_test, models[i][0].fit(X_train, y_train)._predict_proba_lr(X_test))
    for i in tqdm(range(len(models)))
]
# todo: cleaner columns names for categories
# fit models
# pca
# hyperparameter tuning
# xgboost

# hypertuning for extra trees
param_grid = {
    "n_estimators": np.arange(50, 700, 50),
    "max_depth": np.arange(1, 21),
    "min_samples_split": np.arange(1, 21)
}
base_estimator = ExtraTreesClassifier(random_state=42)
sh = HalvingRandomSearchCV(
    base_estimator, param_grid, cv=3,
    factor=2, verbose=10
).fit(X_train, y_train)
best_estimators = []
best_estimators.append((sh.best_estimator_, "proba"))
# hypertuning for random forest
param_grid = {
    "n_estimators": np.arange(50, 700, 50),
    "max_depth": np.arange(1, 21),
    "min_samples_split": np.arange(1, 21)
}
base_estimator = RandomForestClassifier(random_state=42)
sh = HalvingRandomSearchCV(
    base_estimator, param_grid, cv=3,
    factor=2, verbose=10
).fit(X_train, y_train)
best_estimators.append((sh.best_estimator_, "proba"))
# hypertuning for k neighbors
param_grid = {
    "n_neighbors": np.arange(2, 21, 1)
}
base_estimator = KNeighborsClassifier()
sh = HalvingRandomSearchCV(
    base_estimator, param_grid, cv=3,
    factor=2, verbose=10
).fit(X_train, y_train)
best_estimators.append((sh.best_estimator_, "proba"))
# hypertuning for xgboost
param_grid = {
    'max_depth': np.arange(1, 21),
    'learning_rate': stats.uniform(0.01, 0.1),
    'subsample': stats.uniform(0.5, 1),
    'n_estimators': np.arange(50, 700, 50),
}
base_estimator = xgb.XGBClassifier(objective='multi:softmax', seed=42)
sh = HalvingRandomSearchCV(
    base_estimator, param_grid, cv=3,
    factor=2, verbose=10
).fit(X_train, y_train)
best_estimators.append((sh.best_estimator_, "proba"))
# add the remaining estimators
best_estimators.extend(
    [(BernoulliNB(), "proba"), (LinearDiscriminantAnalysis(), "proba"),
    (QuadraticDiscriminantAnalysis(), "proba"), (LinearSVC(multi_class="crammer_singer"), "proba_lr"),
    (LogisticRegression(multi_class="multinomial"), "proba"), (RidgeClassifier(), "proba_lr")
])
# calculate all the log losses
log_losses_3 = [
    log_loss(y_test, best_estimators[i][0].fit(X_train, y_train).predict_proba(X_test))
    if best_estimators[i][1] == "proba" else
    log_loss(y_test, best_estimators[i][0].fit(X_train, y_train)._predict_proba_lr(X_test))
    for i in tqdm(range(len(best_estimators)))
]
print(log_losses_3)
# apply oversampling
X_resampled, y_resampled = SMOTE().fit_resample(X_train, y_train)
# hypertuning for extra trees
param_grid = {
    "n_estimators": np.arange(50, 700, 50),
    "max_depth": np.arange(1, 21),
    "min_samples_split": np.arange(1, 21)
}
base_estimator = ExtraTreesClassifier(random_state=42)
sh = HalvingRandomSearchCV(
    base_estimator, param_grid, cv=3,
    factor=2, verbose=10
).fit(X_resampled, y_resampled)
best_estimators_2 = []
best_estimators_2.append((sh.best_estimator_, "proba"))
# hypertuning for random forest
param_grid = {
    "n_estimators": np.arange(50, 700, 50),
    "max_depth": np.arange(1, 21),
    "min_samples_split": np.arange(1, 21)
}
base_estimator = RandomForestClassifier(random_state=42)
sh = HalvingRandomSearchCV(
    base_estimator, param_grid, cv=3,
    factor=2, verbose=10
).fit(X_resampled, y_resampled)
best_estimators_2.append((sh.best_estimator_, "proba"))
# hypertuning for k neighbors
param_grid = {
    "n_neighbors": np.arange(2, 21, 1)
}
base_estimator = KNeighborsClassifier()
sh = HalvingRandomSearchCV(
    base_estimator, param_grid, cv=3,
    factor=2, verbose=10
).fit(X_resampled, y_resampled)
best_estimators_2.append((sh.best_estimator_, "proba"))
# hypertuning for xgboost
param_grid = {
    'max_depth': np.arange(1, 21),
    'learning_rate': stats.uniform(0.01, 0.1),
    'subsample': stats.uniform(0.5, 1),
    'n_estimators': np.arange(50, 1050, 50),
}
base_estimator = xgb.XGBClassifier(objective='multi:softmax', seed=42)
sh = HalvingRandomSearchCV(
    base_estimator, param_grid, cv=3,
    factor=2, verbose=10
).fit(X_resampled, y_resampled)
best_estimators_2.append((sh.best_estimator_, "proba"))
# add the remaining estimators
best_estimators_2.extend(
    [(BernoulliNB(), "proba"), (LinearDiscriminantAnalysis(), "proba"),
    (QuadraticDiscriminantAnalysis(), "proba"), (LinearSVC(multi_class="crammer_singer"), "proba_lr"),
    (LogisticRegression(multi_class="multinomial"), "proba"), (RidgeClassifier(), "proba_lr")
])
log_losses_4 = [
    log_loss(y_test, best_estimators_2[i][0].fit(X_train, y_train).predict_proba(X_test))
    if best_estimators_2[i][1] == "proba" else
    log_loss(y_test, best_estimators_2[i][0].fit(X_train, y_train)._predict_proba_lr(X_test))
    for i in tqdm(range(len(best_estimators_2)))
]
X_resampled_2, y_resampled_2 = ADASYN().fit_resample(X, y)
# hypertuning for extra trees
param_grid = {
    "n_estimators": np.arange(50, 1050, 50),
    "max_depth": np.arange(1, 21),
    "min_samples_split": np.arange(1, 21)
}
base_estimator = ExtraTreesClassifier(random_state=42)
sh = HalvingRandomSearchCV(
    base_estimator, param_grid, cv=3,
    factor=2, verbose=10
).fit(X_resampled_2, y_resampled_2)
best_estimators_3 = []
best_estimators_3.append((sh.best_estimator_, "proba"))
# hypertuning for random forest
param_grid = {
    "n_estimators": np.arange(50, 1050, 50),
    "max_depth": np.arange(1, 21),
    "min_samples_split": np.arange(1, 21)
}
base_estimator = RandomForestClassifier(random_state=42)
sh = HalvingRandomSearchCV(
    base_estimator, param_grid, cv=3,
    factor=2, verbose=10
).fit(X_resampled_2, y_resampled_2)
best_estimators_3.append((sh.best_estimator_, "proba"))
# hypertuning for k neighbors
param_grid = {
    "n_neighbors": np.arange(2, 21, 50)
}
base_estimator = KNeighborsClassifier()
sh = HalvingRandomSearchCV(
    base_estimator, param_grid, cv=3,
    factor=2, verbose=10
).fit(X_resampled_2, y_resampled_2)
best_estimators_3.append((sh.best_estimator_, "proba"))
# hypertuning for xgboost
param_grid = {
    'max_depth': np.arange(1, 21),
    'learning_rate': stats.uniform(0.01, 0.1),
    'subsample': stats.uniform(0.5, 1),
    'n_estimators': np.arange(50, 1050, 50),
}
base_estimator = xgb.XGBClassifier(objective='multi:softmax', seed=42)
sh = HalvingRandomSearchCV(
    base_estimator, param_grid, cv=3,
    factor=2, verbose=10
).fit(X_resampled_2, y_resampled_2)
best_estimators_3.append((sh.best_estimator_, "proba"))
# add the remaining estimators
best_estimators_3.extend(
    [(BernoulliNB(), "proba"), (LinearDiscriminantAnalysis(), "proba"),
    (QuadraticDiscriminantAnalysis(), "proba"), (LinearSVC(multi_class="crammer_singer"), "proba_lr"),
    (LogisticRegression(multi_class="multinomial"), "proba"), (RidgeClassifier(), "proba_lr")
])
log_losses_5 = [
    log_loss(y_test, best_estimators_3[i][0].fit(X_train, y_train).predict_proba(X_test))
    if best_estimators_3[i][1] == "proba" else
    log_loss(y_test, best_estimators_3[i][0].fit(X_train, y_train)._predict_proba_lr(X_test))
    for i in tqdm(range(len(best_estimators_3)))
]

classifier = best_estimators_3[1][0]
classifier.fit(X_train, y_train)
test = pd.read_csv("S3_26/test.csv")
test["Stage"] = test.Stage.astype("category")
# add dummy variables
test = pd.concat([test, pd.get_dummies(test.Drug, prefix="Drug")], axis=1).drop(columns="Drug")
test = pd.concat([test, pd.get_dummies(test.Sex, prefix="Sex")], axis=1).drop(columns="Sex")
test = pd.concat([test, pd.get_dummies(test.Ascites, prefix="Ascites")], axis=1).drop(columns="Ascites")
test = pd.concat([test, pd.get_dummies(test.Hepatomegaly, prefix="Hepatomegaly")], axis=1).drop(columns="Hepatomegaly")
test = pd.concat([test, pd.get_dummies(test.Spiders, prefix="Spiders")], axis=1).drop(columns="Spiders")
test = pd.concat([test, pd.get_dummies(test.Edema, prefix="Edema")], axis=1).drop(columns="Edema")
test = pd.concat([test, pd.get_dummies(test.Stage, prefix="Stage")], axis=1).drop(columns="Stage")
probs = pd.DataFrame(classifier.predict_proba(test.drop(columns="id")))
probs.columns = ["Status_" + c for c in le.classes_]
df = pd.concat([test[["id"]], probs], axis=1)
df.to_csv("S3_26/out.csv", index=False)
# todo: pca
# data wrangling for test data as well
# decrease test / train split
# check if cat variable over normal improves it
# include original data
# rewrite as classes