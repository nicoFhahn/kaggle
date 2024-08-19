from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def objective_fun_xgb(trial, random_state, use_gpu, splits, scoring_cv, scoring_xgb, X_train, y_train, objective):
    '''
    objective function for training an xtreme gradient boosting model
    :param trial: A trial is a process of evaluating an objective function
    :param random_state: Random number seed.
    :param use_gpu: should the computation be done on the gpu?
    :param splits: number of splits/folds to use for cross validation
    :param scoring_cv: metric used to evaluate the cross-validation
    :param scoring_xgb: evaluation metrics for validation data
    :param X_train: Feature matrix
    :param y_train: Target matrix
    :param objective: Regression or classification?
    :return:
    '''
    params = {
        'max_depth': trial.suggest_int('max_depth', 1, 9),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 50, 600),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
        'subsample': trial.suggest_loguniform('subsample', 0.01, 1.0),
        'colsample_bytree': trial.suggest_loguniform('colsample_bytree', 0.01, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1.0),
        'eval_metric': scoring_xgb,
        'use_label_encoder': False,
        'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
        'seed': random_state
    }
    if use_gpu:
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
    if objective == 'Regression':
        mod = XGBRegressor(**params)
        skf = KFold(n_splits=splits, shuffle=True, random_state=random_state)
    else:
        mod = XGBClassifier(**params)
        skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)
    cv = abs(cross_val_score(mod, X_train, y_train, cv=skf, scoring=scoring_cv).mean())
    return cv

def objective_fun_catboost(trial, random_state, use_gpu, splits, scoring_cv, scoring_cat, X_train, y_train, objective):
    '''
    objective function for training a catboost model
    :param trial: A trial is a process of evaluating an objective function
    :param random_state: Random number seed.
    :param use_gpu: should the computation be done on the gpu?
    :param splits: number of splits/folds to use for cross validation
    :param scoring_cv: metric used to evaluate the cross-validation
    :param scoring_cat: evaluation metrics for validation data
    :param X_train: Feature matrix
    :param y_train: Target matrix
    :param objective: Regression or classification?
    :return:
    '''
    params = {
        'iterations': trial.suggest_int('iterations', 50, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0),
        'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 1, 50),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 2, 50),
        'grow_policy': trial.suggest_categorical('grow_policy', ['Depthwise', 'Lossguide']),
        'eval_metric': scoring_cat,
        'random_state': random_state
    }
    if use_gpu:
        params['task_type'] = 'GPU'
    if objective == 'Regression':
        mod = CatBoostRegressor(**params)
        skf = KFold(n_splits=splits, shuffle=True, random_state=random_state)
    else:
        mod = CatBoostClassifier(**params)
        skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)
    cv = abs(cross_val_score(mod, X_train, y_train, cv=skf, scoring=scoring_cv).mean())
    return cv

def objective_fun_lightgbm(trial, random_state, use_gpu, splits, scoring_cv, scoring_gbm, X_train, y_train, objective):
    '''
    objective function for training a light gbm model
    :param trial: A trial is a process of evaluating an objective function
    :param random_state: Random number seed.
    :param use_gpu: should the computation be done on the gpu?
    :param splits: number of splits/folds to use for cross validation
    :param scoring_cv: metric used to evaluate the cross-validation
    :param scoring_gbm: evaluation metrics for validation data
    :param X_train: Feature matrix
    :param y_train: Target matrix
    :param objective: Regression or classification?
    :return:
    '''
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0),
        'min_child_weight': trial.suggest_float('min_child_weight', 0.5, 10),
        'subsample': trial.suggest_float('subsample', 0.01, 1),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 1),
        'metric': scoring_gbm,
        'random_state': random_state
    }
    if use_gpu:
        params['device'] = 'gpu'
    if objective == 'Regression':
        params['objective'] = 'regression'
        mod = LGBMRegressor(**params)
        skf = KFold(n_splits=splits, shuffle=True, random_state=random_state)
    else:
        params['objective'] = 'multiclass'
        mod = LGBMClassifier(**params)
        skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)
    cv = abs(cross_val_score(mod, X_train, y_train, cv=skf, scoring=scoring_cv).mean())
    return cv

def objective_fun_random_forest(trial, random_state, splits, scoring_cv, X_train, y_train, objective):
    '''
    objective function for training a random forest model
    :param trial: A trial is a process of evaluating an objective function
    :param random_state: Random number seed.
    :param splits: number of splits/folds to use for cross validation
    :param scoring_cv: metric used to evaluate the cross-validation
    :param X_train: Feature matrix
    :param y_train: Target matrix
    :param objective: Regression or classification?
    :return:
    '''
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'max_depth': trial.suggest_int('max_depth', 1, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 50),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 50),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        'random_state': random_state
    }
    if objective == 'Regression':
        params['criterion'] = trial.suggest_categorical('criterion', ['squared_error', 'poisson', 'absolute_error', 'friedman_mse'])
        mod = RandomForestRegressor(**params)
        skf = KFold(n_splits=splits, shuffle=True, random_state=random_state)
    else:
        params['criterion'] = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss']),
        mod = RandomForestClassifier(**params)
        skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)
    cv = abs(cross_val_score(mod, X_train, y_train, cv=skf, scoring=scoring_cv).mean())
    return cv
