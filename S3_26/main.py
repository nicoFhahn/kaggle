import jenkspy
import pandas as pd
import seaborn as sns
import xgboost as xgb
import optuna
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
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.model_selection import cross_val_score, StratifiedKFold
from collections import Counter
from tqdm import tqdm
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class multiclassifier():
    def __init__(self, file_path, full_data=True, random_state=420, splits=8, use_gpu=False):
        if full_data:
            self.train = pd.concat([pd.read_csv(f'{file_path}/train.csv'), pd.read_csv(f'{file_path}/train.csv')])
        else:
            self.train = pd.read_csv(f'{file_path}/train.csv')
        self.test = pd.read_csv(f'{file_path}/test.csv')
        self.le = LabelEncoder()
        self.random_state = random_state
        self.splits = splits
        self.use_gpu = use_gpu
        self.base_models = [
            BernoulliNB(), ExtraTreesClassifier(random_state=random_state),
            RandomForestClassifier(random_state=random_state), KNeighborsClassifier(),
            LinearDiscriminantAnalysis(), QuadraticDiscriminantAnalysis(),
            LogisticRegression(multi_class='multinomial'), xgb.XGBClassifier(objective='multi:softmax', seed=random_state)
        ]
    def preprocess_data(self, test_size=.15, outlier_method='keep', synthetic_data=False, scaling_method="robust"):
        if scaling_method == "robust":
            scaler = RobustScaler()
        elif scaling_method == "standard":
            scaler = StandardScaler()
        if scaling_method != "none":
            cols = self.train.select_dtypes(include='number').columns[1:]
            scaler.fit(self.train[cols])
            transformed = scaler.transform(self.train[cols])
            self.train[cols] = transformed
            scaler.fit(self.test[cols])
            transformed = scaler.transform(self.test[cols])
            self.test[cols] = transformed
        if outlier_method != 'keep':
            cols = self.train.select_dtypes(include='number').columns[1:]
            for col in cols:
                fit = LocalOutlierFactor(n_neighbors=20, contamination=0.1).fit_predict(self.train[[col]])
                temp_df = pd.concat([self.train[[col]].reset_index(drop=True), pd.Series(fit)], axis=1)
                min_value = temp_df[temp_df[0] != -1][col].min()
                max_value = temp_df[temp_df[0] != -1][col].max()
                if outlier_method == 'nearest':
                    self.train.loc[self.train[col] <= min_value, col] = min_value
                    self.train.loc[self.train[col] >= max_value, col] = max_value
                    self.test.loc[self.test[col] <= min_value, col] = min_value
                    self.test.loc[self.test[col] >= max_value, col] = max_value
                elif outlier_method == 'drop':
                    self.train = self.train.drop(self.train[mc.train[col] < min_value].index)
                    self.train = self.train.drop(self.train[mc.train[col] > max_value].index)
        self.train['Stage'] = self.train['Stage'].astype('category')
        self.test['Stage'] = self.test['Stage'].astype('category')
        self.train = pd.concat([self.train, pd.get_dummies(self.train.Drug, prefix='Drug')], axis=1).drop(columns='Drug')
        self.train = pd.concat([self.train, pd.get_dummies(self.train.Sex, prefix='Sex')], axis=1).drop(columns='Sex')
        self.train = pd.concat([self.train, pd.get_dummies(self.train.Ascites, prefix='Ascites')], axis=1).drop(columns='Ascites')
        self.train = pd.concat([self.train, pd.get_dummies(self.train.Hepatomegaly, prefix='Hepatomegaly')], axis=1).drop(columns='Hepatomegaly')
        self.train = pd.concat([self.train, pd.get_dummies(self.train.Spiders, prefix='Spiders')], axis=1).drop(columns='Spiders')
        self.train = pd.concat([self.train, pd.get_dummies(self.train.Edema, prefix='Edema')], axis=1).drop(columns='Edema')
        self.train = pd.concat([self.train, pd.get_dummies(self.train.Stage, prefix='Stage')], axis=1).drop(columns='Stage')
        self.test = pd.concat([self.test, pd.get_dummies(self.test.Drug, prefix='Drug')], axis=1).drop(columns='Drug')
        self.test = pd.concat([self.test, pd.get_dummies(self.test.Sex, prefix='Sex')], axis=1).drop(columns='Sex')
        self.test = pd.concat([self.test, pd.get_dummies(self.test.Ascites, prefix='Ascites')], axis=1).drop(columns='Ascites')
        self.test = pd.concat([self.test, pd.get_dummies(self.test.Hepatomegaly, prefix='Hepatomegaly')], axis=1).drop(columns='Hepatomegaly')
        self.test = pd.concat([self.test, pd.get_dummies(self.test.Spiders, prefix='Spiders')], axis=1).drop(columns='Spiders')
        self.test = pd.concat([self.test, pd.get_dummies(self.test.Edema, prefix='Edema')], axis=1).drop(columns='Edema')
        self.test = pd.concat([self.test, pd.get_dummies(self.test.Stage, prefix='Stage')], axis=1).drop(columns='Stage')
        if synthetic_data:
            X, y = SMOTE().fit_resample(self.train.drop(columns=['id', 'Status']), self.le.fit_transform(self.train['Status']))
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=self.random_state)
        else:
            X, y = self.train.drop(columns=['id']), self.le.fit_transform(self.train['Status'])
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=self.random_state, stratify=y)
            self.X_train = self.X_train.drop(columns='Status')
            self.X_test = self.X_test.drop(columns='Status')
    def train_models(self, models):
        self.trained_base_models = [model.fit(self.X_train, self.y_train) for model in tqdm(models)]
    def calc_log_loss(self, estimator):
        probabilities_train = estimator.predict_proba(X_train)
        probabilities_test = estimator.predict_proba(X_test)
        log_loss_train = log_loss(self.y_train, probabilities_train)
        log_loss_test = log_loss(self.y_test, probabilities_test)
        return log_loss_train, log_loss_test
    def objective_fun_xgb(self, trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 1, 9),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 1.0),
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
            'subsample': trial.suggest_loguniform('subsample', 0.01, 1.0),
            'colsample_bytree': trial.suggest_loguniform('colsample_bytree', 0.01, 1.0),
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1.0),
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1.0),
            'eval_metric': 'mlogloss',
            'use_label_encoder': False,
            'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
            'seed': self.random_state
        }
        if self.use_gpu:
            params['device'] = 'cuda'
        if params['booster'] == 'gbtree' or params['booster'] == 'dart':
            params['max_depth'] = trial.suggest_int('max_depth', 1, 9)
            params['eta'] = trial.suggest_loguniform('eta', 1e-8, 1.0)
            params['gamma'] = trial.suggest_loguniform('gamma', 1e-8, 1.0)
            params['grow_policy'] = trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])
        if params['booster'] == 'dart':
            params['sample_type'] = trial.suggest_categorical('sample_type', ['uniform', 'weighted'])
            params['normalize_type'] = trial.suggest_categorical('normalize_type', ['tree', 'forest'])
            params['rate_drop'] = trial.suggest_loguniform('rate_drop', 1e-8, 1.0)
            params['skip_drop'] = trial.suggest_loguniform('skip_drop', 1e-8, 1.0)
        mod = xgb.XGBClassifier(**params)
        skf = StratifiedKFold(n_splits=self.splits, shuffle=True, random_state=self.random_state)
        cv = abs(cross_val_score(mod, self.X_train, self.y_train, cv=skf, scoring='neg_log_loss').mean())
        return cv
    def objective_fun_catboost(self, trial):
        params = {
            'iterations': trial.suggest_int('iterations', 50, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0),
            'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 1, 50),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 2, 50),
            'grow_policy': 'Lossguide',
            'random_state': self.random_state
        }
        if self.use_gpu:
            params['task_type'] = 'GPU'
        mod = CatBoostClassifier(**params)
        skf = StratifiedKFold(n_splits=self.splits, shuffle=True, random_state=self.random_state)
        cv = abs(cross_val_score(mod, self.X_train, self.y_train, cv=skf, scoring='neg_log_loss').mean())
        return cv
    def objective_fun_lightgbm(self, trial):
        params = {
            'n_estimators': trial.suggest_int('iterations', 50, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0),
            'min_child_weight': trial.suggest_float('min_child_weight', 0.5, 10),
            'subsample': trial.suggest_float('subsample', 0.01, 1),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 1),
            'objective': 'multiclass',
            'metric': 'multi_logloss',
            'random_state': self.random_state
        }
        if self.use_gpu:
            params['device'] = 'gpu'
        mod = LGBMClassifier(**params)
        skf = StratifiedKFold(n_splits=self.splits, shuffle=True, random_state=self.random_state)
        cv = abs(cross_val_score(mod, self.X_train, self.y_train, cv=skf, scoring='neg_log_loss').mean())
        return cv
    def objective_fun_rf(self, trial):
        params = {
            'n_estimators': trial.suggest_int('iterations', 50, 1000),
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss']),
            'max_depth': trial.suggest_int('max_depth', 1, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 50),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 50),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            'random_state': self.random_state
        }
        mod = RandomForestClassifier(**params)
        skf = StratifiedKFold(n_splits=self.splits, shuffle=True, random_state=self.random_state)
        cv = abs(cross_val_score(mod, self.X_train, self.y_train, cv=skf, scoring='neg_log_loss').mean())
        return cv
    def tune_model(self, objective, n_trials):
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        return study
    def fit_test_data(self, estimator):
        probs = pd.DataFrame(estimator.predict_proba(self.test.drop(columns='id')))
        probs.columns = ['Status_' + c for c in self.le.classes_]
        return pd.concat([self.test[['id']], probs], axis=1)



mc = multiclassifier(file_path='S3_26', use_gpu=False)
mc.preprocess_data(synthetic_data=True, test_size=.25, outlier_method='nearest', scaling_method="robust")
mc.train_models(mc.base_models)
exgb_study = mc.tune_model(mc.objective_fun_xgb, n_trials=50)
cat_study = mc.tune_model(mc.objective_fun_catboost, n_trials=50)
lgbm_study = mc.tune_model(mc.objective_fun_lightgbm, n_trials=50)
rf_study = mc.tune_model(mc.objective_fun_rf, n_trials=50)
# todo: include scaler
# from sklearn.preprocessing import MinMaxScaler

# https://www.kaggle.com/code/ashishkumarak/liver-cirrhosis-survival-prediction-multiclass
# https://optuna.org/
# https://practicaldatascience.co.uk/machine-learning/how-to-use-your-gpu-to-accelerate-xgboost-models

params_lgbm = {'iterations': 435, 'learning_rate': 0.17223516709365655, 'min_child_weight': 3.0876406237537144, 'subsample': 0.1839940317845411, 'colsample_bytree': 0.23777566321261087}
params_cat = {'iterations': 424, 'learning_rate': 0.328632739907193, 'l2_leaf_reg': 1, 'min_data_in_leaf': 34}
params_xgb = {'max_depth': 6, 'learning_rate': 0.2038999367930274, 'n_estimators': 331, 'min_child_weight': 2, 'gamma': 0.08190835902266243, 'subsample': 0.665389653995815, 'colsample_bytree': 0.322299220021032, 'reg_alpha': 5.9231902273760234e-08, 'reg_lambda': 0.10908267874899857, 'booster': 'gbtree', 'eta': 0.0038577922397016054, 'grow_policy': 'depthwise'}

X_train, y_train, X_test, y_test = mc.X_train, mc.y_train, mc.X_test, mc.y_test
lgbm = LGBMClassifier(**params_lgbm, random_state=420)
lgbm.fit(X_train, y_train)
cat = CatBoostClassifier(**params_cat, random_state=420, grow_policy='Lossguide')
cat.fit(X_train, y_train)
exgb = xgb.XGBClassifier(**params_xgb, random_state=420)
exgb.fit(X_train, y_train)
lgbm_prob = lgbm.predict_proba(X_test)
lgbm_ll = log_loss(y_test, lgbm_prob) # 0.25638605969931555
cat_prob = cat.predict_proba(X_test)
cat_ll = log_loss(y_test, cat_prob) # 0.15390307754220245
xgb_prob = exgb.predict_proba(X_test)
xgb_ll = log_loss(y_test, xgb_prob) # 0.15445409594605847

mc.fit_test_data(lgbm).to_csv("lgbm.csv", index=False)
mc.fit_test_data(cat).to_csv("cat.csv", index=False)
mc.fit_test_data(exgb).to_csv("xgb.csv", index=False)