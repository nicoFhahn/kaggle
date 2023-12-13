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


class multiclassifier():
    def __init__(self, file_path, full_data=True, random_state=420):
        if full_data:
            self.train = pd.concat([pd.read_csv(f"{file_path}/train.csv"), pd.read_csv(f"{file_path}/train.csv")])
        else:
            self.train = pd.read_csv(f"{file_path}/train.csv")
        self.test = pd.read_csv(f"{file_path}/test.csv")
        self.le = LabelEncoder()
        self.random_state = random_state
        self.base_models = [
            BernoulliNB(), ExtraTreesClassifier(random_state=random_state),
            RandomForestClassifier(random_state=random_state), KNeighborsClassifier(),
            LinearDiscriminantAnalysis(), QuadraticDiscriminantAnalysis(),
            LogisticRegression(multi_class="multinomial"), xgb.XGBClassifier(objective='multi:softmax', seed=random_state),
            RidgeClassifier()
        ]
        self.proba_functions = np.concatenate((np.repeat("proba", 8), np.repeat("proba_lr", 1)))

    def preprocess_data(self, test_size=.15, outlier_method="keep", synthetic_data=False):
        if outlier_method != "keep":
            cols = self.train.select_dtypes(include="number").columns[1:]
            for col in cols:
                fit = LocalOutlierFactor(n_neighbors=20, contamination=0.1).fit_predict(self.train[[col]])
                temp_df = pd.concat([self.train[[col]].reset_index(drop=True), pd.Series(fit)], axis=1)
                min_value = temp_df[temp_df[0] != -1][col].min()
                max_value = temp_df[temp_df[0] != -1][col].max()
                if outlier_method == "nearest":
                    self.train.loc[self.train[col] <= min_value, col] = min_value
                    self.train.loc[self.train[col] >= max_value, col] = max_value
                    self.test.loc[self.test[col] <= min_value, col] = min_value
                    self.test.loc[self.test[col] >= max_value, col] = max_value
                elif outlier_method == "drop":
                    self.train = self.train.drop(self.train[mc.train[col] < min_value].index)
                    self.train = self.train.drop(self.train[mc.train[col] > max_value].index)
        self.train["Stage"] = self.train["Stage"].astype("category")
        self.test["Stage"] = self.test["Stage"].astype("category")
        self.train = pd.concat([self.train, pd.get_dummies(self.train.Drug, prefix="Drug")], axis=1).drop(columns="Drug")
        self.train = pd.concat([self.train, pd.get_dummies(self.train.Sex, prefix="Sex")], axis=1).drop(columns="Sex")
        self.train = pd.concat([self.train, pd.get_dummies(self.train.Ascites, prefix="Ascites")], axis=1).drop(columns="Ascites")
        self.train = pd.concat([self.train, pd.get_dummies(self.train.Hepatomegaly, prefix="Hepatomegaly")], axis=1).drop(columns="Hepatomegaly")
        self.train = pd.concat([self.train, pd.get_dummies(self.train.Spiders, prefix="Spiders")], axis=1).drop(columns="Spiders")
        self.train = pd.concat([self.train, pd.get_dummies(self.train.Edema, prefix="Edema")], axis=1).drop(columns="Edema")
        self.train = pd.concat([self.train, pd.get_dummies(self.train.Stage, prefix="Stage")], axis=1).drop(columns="Stage")
        self.test = pd.concat([self.test, pd.get_dummies(self.test.Drug, prefix="Drug")], axis=1).drop(columns="Drug")
        self.test = pd.concat([self.test, pd.get_dummies(self.test.Sex, prefix="Sex")], axis=1).drop(columns="Sex")
        self.test = pd.concat([self.test, pd.get_dummies(self.test.Ascites, prefix="Ascites")], axis=1).drop(columns="Ascites")
        self.test = pd.concat([self.test, pd.get_dummies(self.test.Hepatomegaly, prefix="Hepatomegaly")], axis=1).drop(columns="Hepatomegaly")
        self.test = pd.concat([self.test, pd.get_dummies(self.test.Spiders, prefix="Spiders")], axis=1).drop(columns="Spiders")
        self.test = pd.concat([self.test, pd.get_dummies(self.test.Edema, prefix="Edema")], axis=1).drop(columns="Edema")
        self.test = pd.concat([self.test, pd.get_dummies(self.test.Stage, prefix="Stage")], axis=1).drop(columns="Stage")
        if synthetic_data:
            X, y = SMOTE().fit_resample(self.train.drop(columns=["id", "Status"]), self.le.fit_transform(self.train["Status"]))
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=self.random_state)
        else:
            X, y = self.train.drop(columns=["id"]), self.le.fit_transform(self.train["Status"])
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=self.random_state, stratify=y)
            self.X_train = self.X_train.drop(columns="Status")
            self.X_test = self.X_test.drop(columns="Status")
    def train_models(self, models):
        self.trained_base_models = [model.fit(self.X_train, self.y_train) for model in tqdm(models)]
    def predict_probabilities(self, is_base=True):
        if is_base:
            self.base_probabilities_train = [self.trained_base_models[i].predict_proba(self.X_train) if self.proba_functions[i] == "proba" else self.trained_base_models[i]._predict_proba_lr(self.X_train) for i in range(len(self.trained_base_models))]
            self.base_probabilities_test = [self.trained_base_models[i].predict_proba(self.X_test) if self.proba_functions[i] == "proba" else self.trained_base_models[i]._predict_proba_lr(self.X_test) for i in range(len(self.trained_base_models))]
        else:
            self.probabilities_train = [trained_model.predict_proba(self.X_train) for trained_model in self.trained_models]
            self.probabilities_test = [trained_model.predict_proba(self.X_test) for trained_model in self.trained_models]
    def calc_log_loss(self, is_base=True):
        if is_base:
            self.base_log_loss_train = [log_loss(self.y_train, prob) for prob in self.base_probabilities_train]
            self.base_log_loss_test = [log_loss(self.y_test, prob) for prob in self.base_probabilities_test]
        else:
            self.log_loss_train = [log_loss(self.y_train, prob) for prob in self.probabilities_train]
            self.log_loss_test = [log_loss(self.y_test, prob) for prob in self.probabilities_test]

    def tune_models(
            self,
            n_estimators=np.arange(50, 500, 50),
            max_depth=np.arange(1, 21),
            min_samples_split=np.arange(1, 21),
            n_neighbors=np.arange(1, 21),
            learning_rate=stats.uniform(0.01, 0.1),
            subsample=stats.uniform(0.5, 1),
            cv=3
    ):
        param_grid_tree = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split
        }
        param_grid_neighboours = {
            "n_neighbors": n_neighbors
        }
        param_grid_xg = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample
        }
        base_estimators = [
            RandomForestClassifier(random_state=self.random_state),
            ExtraTreesClassifier(random_state=self.random_state),
            KNeighborsClassifier(),
            xgb.XGBClassifier(objective="multi:softmax", seed=self.random_state)
        ]
        param_grids = np.concatenate((np.repeat(param_grid_tree, 2), [param_grid_neighboours], [param_grid_xg]))
        searches = [HalvingRandomSearchCV(base_estimators[i], param_grids[i], cv=cv, factor=3, verbose=10, scoring="neg_log_loss") for i in tqdm(range(4))]
        fits = [search.fit(self.X_train, self.y_train) for search in searches]
        self.trained_models = [fit.best_estimator_ for fit in fits]

    def fit_test_data(self, estimator):
        try:
            probs = pd.DataFrame(estimator.predict_proba(self.test.drop(columns="id")))
        except:
            probs = pd.DataFrame(estimator._predict_proba_lr(self.test.drop(columns="id")))
        probs.columns = ["Status_" + c for c in self.le.classes_]
        self.predictions = pd.concat([self.test[["id"]], probs], axis=1)



mc = multiclassifier(file_path="S3_26")
mc.preprocess_data(synthetic_data=True, test_size=.2, outlier_method="nearest")
mc.train_models(mc.base_models)
mc.predict_probabilities()
mc.calc_log_loss()
mc.fit_test_data(mc.base_models[0])
mc.tune_models()
mc.predict_probabilities(is_base=False)
mc.calc_log_loss(is_base=False)
mc.fit_test_data(mc.trained_models[0])
mc.predictions.to_csv("S3_26/predictions_random_forest_v1.csv", index=False)
mc.fit_test_data(mc.trained_models[1])
mc.predictions.to_csv("S3_26/predictions_extra_trees_v1.csv", index=False)
mc.fit_test_data(mc.trained_models[3])
mc.predictions.to_csv("S3_26/predictions_xgb_v1.csv", index=False)

mc2 = multiclassifier(file_path="S3_26")
mc2.preprocess_data(synthetic_data=False, outlier_method="nearest", test_size=.2)
mc2.train_models(mc2.base_models)
mc2.predict_probabilities()
mc2.calc_log_loss()
mc2.tune_models()
mc2.predict_probabilities(is_base=False)
mc2.calc_log_loss(is_base=False)
mc2.fit_test_data(mc2.trained_models[0])
mc2.predictions.to_csv("S3_26/predictions_random_forest_v2.csv", index=False)
mc2.fit_test_data(mc2.trained_models[1])
mc2.predictions.to_csv("S3_26/predictions_extra_trees_v2.csv", index=False)
mc2.fit_test_data(mc2.trained_models[3])
mc2.predictions.to_csv("S3_26/predictions_xgb_v2.csv", index=False)

# todo: besten extra trees exporten
# ohne extra data
# train size 80%

mc3 = multiclassifier(file_path="S3_26", full_data=False)
mc3.preprocess_data(synthetic_data=True, outlier_method="nearest", test_size=.2)
mc3.train_models(mc3.base_models)
mc3.predict_probabilities()
mc3.calc_log_loss()
mc3.tune_models(n_estimators=np.arange(50, 500, 50))
mc3.predict_probabilities(is_base=False)
mc3.calc_log_loss(is_base=False)
mc3.fit_test_data(mc3.trained_models[0])
mc3.predictions.to_csv("S3_26/predictions_random_forest_v3.csv", index=False)
mc3.fit_test_data(mc3.trained_models[1])
mc3.predictions.to_csv("S3_26/predictions_extra_trees_v3.csv", index=False)
mc3.fit_test_data(mc3.trained_models[3])
mc3.predictions.to_csv("S3_26/predictions_xgb_v3.csv", index=False)

mc4 = multiclassifier(file_path="S3_26", full_data=False)
mc4.preprocess_data(synthetic_data=False, outlier_method="nearest", test_size=.2)
mc4.train_models(mc4.base_models)
mc4.predict_probabilities()
mc4.calc_log_loss()
mc4.tune_models(n_estimators=np.arange(50, 500, 50))
mc4.predict_probabilities(is_base=False)
mc4.calc_log_loss(is_base=False)
mc4.fit_test_data(mc4.trained_models[0])
mc4.predictions.to_csv("S3_26/predictions_random_forest_v4.csv", index=False)
mc4.fit_test_data(mc4.trained_models[1])
mc4.predictions.to_csv("S3_26/predictions_extra_trees_v4.csv", index=False)
mc4.fit_test_data(mc4.trained_models[3])
mc4.predictions.to_csv("S3_26/predictions_xgb_v4.csv", index=False)