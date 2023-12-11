import jenkspy
import pandas as pd
import seaborn as sns
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, IsolationForest
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, LocalOutlierFactor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, RidgeClassifier, RidgeClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from tqdm import tqdm

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
train = pd.read_csv("train.csv")
df = train.copy()
train = pd.concat([train, pd.get_dummies(train.Drug)], axis=1).drop(columns="Drug")
train = pd.concat([train, pd.get_dummies(train.Sex)], axis=1).drop(columns="Sex")
train = pd.concat([train, pd.get_dummies(train.Ascites)], axis=1).drop(columns="Ascites")
train = pd.concat([train, pd.get_dummies(train.Hepatomegaly)], axis=1).drop(columns="Hepatomegaly")
train = pd.concat([train, pd.get_dummies(train.Spiders)], axis=1).drop(columns="Spiders")
train = pd.concat([train, pd.get_dummies(train.Edema)], axis=1).drop(columns="Edema")
X, y = train.drop(columns=["id", "Status"]), train.Status
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=420)

models = [BernoulliNB(), ExtraTreesClassifier(), RandomForestClassifier(), KNeighborsClassifier(),
          LinearDiscriminantAnalysis(), QuadraticDiscriminantAnalysis(), LinearSVC(multi_class="crammer_singer"),
          LogisticRegression(multi_class="multinomial"), RidgeClassifier()]
model_1 = BernoulliNB().fit(X_train, y_train)
log_loss_1 = log_loss(y_test, model_1.predict_proba(X_test))
model_2 = ExtraTreesClassifier().fit(X_train, y_train)
log_loss_2 = log_loss(y_test, model_2.predict_proba(X_test))
model_3 = RandomForestClassifier().fit(X_train, y_train)
log_loss_3 = log_loss(y_test, model_3.predict_proba(X_test))
model_4 = KNeighborsClassifier().fit(X_train, y_train)
log_loss_4 = log_loss(y_test, model_4.predict_proba(X_test))
model_5 = LinearDiscriminantAnalysis().fit(X_train, y_train)
log_loss_5 = log_loss(y_test, model_5.predict_proba(X_test))
model_6 = QuadraticDiscriminantAnalysis().fit(X_train, y_train)
log_loss_6 = log_loss(y_test, model_6.predict_proba(X_test))
model_7 = LinearSVC(multi_class="crammer_singer").fit(X_train, y_train)
log_loss_7 = log_loss(y_test, model_7._predict_proba_lr(X_test))
model_8 = LogisticRegression(multi_class="multinomial").fit(X_train, y_train)
log_loss_8 = log_loss(y_test, model_8.predict_proba(X_test))
model_9 = LogisticRegressionCV(multi_class="multinomial").fit(X_train, y_train)
log_loss_9 = log_loss(y_test, model_9.predict_proba(X_test))
model_10 = RidgeClassifier().fit(X_train, y_train)
log_loss_10 = log_loss(y_test, model_10._predict_proba_lr(X_test))
model_11 = RidgeClassifierCV().fit(X_train, y_train)
log_loss_11 = log_loss(y_test, model_11._predict_proba_lr(X_test))

for col in ["SGOT", "Tryglicerides", "Prothrombin"]:
    fit = LocalOutlierFactor(n_neighbors=20, contamination=0.1).fit_predict(X_train[[col]])
    temp_df = pd.concat([X_train[[col]].reset_index(drop=True), pd.Series(fit)], axis=1)
    min_value = temp_df[temp_df[0] != -1 ][col].min()
    max_value = temp_df[temp_df[0] != -1 ][col].max()
    X_train.loc[X_train[col] <= min_value, col] = min_value
    X_train.loc[X_train[col] >= max_value, col] = max_value
    X_test.loc[X_test[col] <= min_value, col] = min_value
    X_test.loc[X_test[col] >= max_value, col] = max_value
jenks_dict = {
    "Bilirubin": 3,
    "Cholesterol": 2,
    "Copper": 3,
    "Alk_Phos": 3
}
for key, value in jenks_dict.items():
    breaks = jenkspy.jenks_breaks(X_train[key], n_classes=value)
    cat = pd.cut(X_train[key], bins=breaks)
    X_train[key + "_cat"] = cat
    cat = pd.cut(X_test[key], bins=breaks)
    X_test[key + "_cat"] = cat

X_train["Stage"] = X_train.Stage.astype("category")
X_train = X_train.copy()
X_train_2 = X_train[X_train.columns[:-4]]
fitted_models = [m.fit(X_train_2, y_train) for m in tqdm(models)]
X_train_3 = X_train.drop(columns=jenks_dict.keys())
X_train_3 = pd.concat([X_train_3, pd.get_dummies(X_train_3.Bilirubin_cat)], axis=1).drop(columns="Bilirubin_cat")
X_train_3 = pd.concat([X_train_3, pd.get_dummies(X_train_3.Cholesterol_cat)], axis=1).drop(columns="Cholesterol_cat")
X_train_3 = pd.concat([X_train_3, pd.get_dummies(X_train_3.Copper_cat)], axis=1).drop(columns="Copper_cat")
X_test_2 = X_test.copy()
X_test_2 = pd.concat([X_test_2, pd.get_dummies(X_test_2.Alk_Phos_cat)], axis=1).drop(columns="Alk_Phos_cat")
X_test_2 = pd.concat([X_test_2, pd.get_dummies(X_test_2.Bilirubin_cat)], axis=1).drop(columns="Bilirubin_cat")
X_test_2 = pd.concat([X_test_2, pd.get_dummies(X_test_2.Cholesterol_cat)], axis=1).drop(columns="Cholesterol_cat")
X_test_2 = pd.concat([X_test_2, pd.get_dummies(X_test_2.Copper_cat)], axis=1).drop(columns="Copper_cat")
X_test_2 = pd.concat([X_test_2, pd.get_dummies(X_test_2.Alk_Phos_cat)], axis=1).drop(columns="Alk_Phos_cat")
fitted_models = [m.fit(X_train_3, y_train) for m in tqdm(models)]
# todo: cleaner columns names for categories
# fit models
# pca
# hyperparameter tuning