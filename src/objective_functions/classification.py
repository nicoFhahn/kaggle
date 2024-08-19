import optuna
import pandas as pd
import polars as pl
import numpy as np
from typing import Union, Callable
from optuna_integration.xgboost import XGBoostPruningCallback
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn import metrics
import warnings
warnings.simplefilter("ignore", UserWarning)

def hist_gradient_boosting(
        trial,
        X: Union[pl.DataFrame, pd.DataFrame],
        y: Union[pl.Series, pd.Series],
        random_state: int=9825,
        n_splits: int=5,
        scoring:str="f1"
):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0),
        'max_iter': trial.suggest_int('max_iter', 50, 1000),
        'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 2, 100),
        'max_depth': trial.suggest_int('max_depth', 2, 50),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 100),
        'l2_regularization': trial.suggest_float('l2_regularization', 0, 1.0),
        'max_features': trial.suggest_float('max_features', 0, 1),
        'max_bins': trial.suggest_int('max_bins', 2, 255),
        'random_state': random_state
    }
    mod = HistGradientBoostingClassifier(**params)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv = abs(cross_val_score(mod, X, y, cv=skf, scoring=scoring).mean())
    return cv

def random_forest(
        trial,
        X: Union[pl.DataFrame, pd.DataFrame],
        y: Union[pl.Series, pd.Series],
        random_state: int=9825,
        n_splits: int=5,
        scoring:str="f1"
):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss']),
        'max_depth': trial.suggest_int('max_depth', 1, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 50),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 50),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        'random_state': random_state
    }
    mod = RandomForestClassifier(**params)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv = abs(cross_val_score(mod, X, y, cv=skf, scoring=scoring).mean())
    return cv

def cat_boost(
        trial,
        X: Union[pl.DataFrame, pd.DataFrame],
        y: Union[pl.Series, pd.Series],
        random_state: int=9825,
        n_splits: int=5,
        scoring:str="f1",
):
    params = {
        'iterations': trial.suggest_int('iterations', 50, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0),
        'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 1, 50),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 2, 50),
        'grow_policy': 'Lossguide',
        'random_state': random_state,
        'silent': True,
        'depth': trial.suggest_int('depth', 1, 16),
        'random_strength': trial.suggest_float('random_strength', 0, 1)
    }
    mod = CatBoostClassifier(**params)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv = abs(cross_val_score(mod, X, y, cv=skf, scoring=scoring).mean())
    return cv

def lgbm(
        trial,
        X: Union[pl.DataFrame, pd.DataFrame],
        y: Union[pl.Series, pd.Series],
        random_state: int=9825,
        n_splits: int=5,
        scoring:str="f1"
):
    params = {
        'n_estimators': trial.suggest_int('iterations', 50, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0),
        'min_child_weight': trial.suggest_float('min_child_weight', 0.5, 10),
        'subsample': trial.suggest_float('subsample', 0.01, 1),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 1),
        'random_state': random_state,
        'verbose': -1
    }
    mod = LGBMClassifier(**params)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv = abs(cross_val_score(mod, X, y, cv=skf, scoring=scoring).mean())
    return cv


def xgb(
        trial,
        X: Union[pl.DataFrame, pd.DataFrame],
        y: Union[pl.Series, pd.Series],
        random_state: int = 9825,
        n_splits: int = 5,
        scoring:str="f1",
        xgb_scoring: str = "logloss",
        xgb_objective: str = "binary:logistic",
        score_function: Callable = metrics.f1_score,
        do_pruning: bool = False
):
    params = {
        'max_depth': trial.suggest_int('max_depth', 1, 9),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.01, 1.0, log=True),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 1.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        'eval_metric': xgb_scoring,
        'use_label_encoder': False,
        'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
        'random_state': random_state,
        'objective': xgb_objective
    }
    if params['booster'] == 'gbtree' or params['booster'] == 'dart':
        params['max_depth'] = trial.suggest_int('max_depth', 1, 9)
        params['eta'] = trial.suggest_float('eta', 1e-8, 1.0, log=True)
        params['gamma'] = trial.suggest_float('gamma', 1e-8, 1.0, log=True)
        params['grow_policy'] = trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])
    if params['booster'] == 'dart':
        params['sample_type'] = trial.suggest_categorical('sample_type', ['uniform', 'weighted'])
        params['normalize_type'] = trial.suggest_categorical('normalize_type', ['tree', 'forest'])
        params['rate_drop'] = trial.suggest_float('rate_drop', 1e-8, 1.0, log=True)
        params['skip_drop'] = trial.suggest_float('skip_drop', 1e-8, 1.0, log=True)
    if do_pruning:
        mod = XGBClassifier(
            **params,
            early_stopping_rounds=100,
            callbacks=[XGBoostPruningCallback(trial=trial, observation_key=f"validation_0-{xgb_scoring}")],
            verbosity=0
        )
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        cv_scores = []
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
            y = pd.Series(y)
        elif isinstance(X, pl.DataFrame):
            X = X.to_pandas()
            y = y.to_pandas()

        for train_idx, val_idx in skf.split(X, y):
            X_train_cv = X.iloc[train_idx]
            X_val_cv = X.iloc[val_idx]
            y_train_cv = y.iloc[train_idx]
            y_val_cv = y.iloc[val_idx]
            mod.fit(
                X_train_cv, y_train_cv,
                eval_set=[(X_val_cv, y_val_cv)],
                verbose=False
            )
            # Report intermediate objective value
            if xgb_objective in ["multi:softprob"]:
                val_pred = mod.predict_proba(X_val_cv)
                loss = score_function(y_val_cv, val_pred)
            else:
                val_pred = mod.predict(X_val_cv)
                loss = score_function(y_val_cv, val_pred)

            # Report and prune without repeating the step
            trial.report(loss, step=len(cv_scores))
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            cv_scores.append(loss)
        cv = np.mean(cv_scores)
    else:
        mod = XGBClassifier(**params)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        cv = abs(cross_val_score(mod, X, y, cv=skf, scoring=scoring).mean())
    return cv