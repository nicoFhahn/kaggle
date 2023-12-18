import pandas as pd
import numpy as np
import scipy.stats as stats
import itertools
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE, SMOTEN, SMOTENC, BorderlineSMOTE, ADASYN
from imblearn.combine import SMOTETomek
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, MinMaxScaler
from sklearn.impute import KNNImputer, SimpleImputer

def remove_outliers(train_df, test_df, method, n_neighbors, contamination):
    org_cols_train, org_cols_test = train_df.columns, test_df.columns
    cols = train_df.select_dtypes(include="number").columns
    train_remind, test_remind = train_df.drop(columns=cols), test_df.drop(columns=cols)
    fits = [LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination).fit_predict(train_df[[col]]) for col in cols]
    temp_dfs = [pd.concat([train_df[[cols[i]]].reset_index(drop=True), pd.Series(fits[i])], axis=1) for i in range(len(fits))]
    min_values = [temp_df[temp_df[0]==1].iloc[:,0].min() for temp_df in temp_dfs]
    max_values = [temp_df[temp_df[0]==1].iloc[:,0].max() for temp_df in temp_dfs]
    if method == 'nearest':
        new_cols = [replace_values(train_df[[cols[i]]], test_df[[cols[i]]], min_values[i], max_values[i]) for i in range(len(fits))]
        train_df, test_df = pd.concat([new_col[0] for new_col in new_cols], axis=1), pd.concat([new_col[1] for new_col in new_cols], axis=1)
    elif method == 'remove':
        indices = [outlier_indices(train_df[[cols[i]]], min_values[i], max_values[i]) for i in range(len(fits))]
        ind0, ind1 =  list(itertools.chain.from_iterable([ind[0] for ind in indices])), list(itertools.chain.from_iterable([ind[1] for ind in indices]))
        train_df = train_df.drop(list(itertools.chain.from_iterable([ind0, ind1])))
    train_df = pd.concat([train_df, train_remind], axis=1)
    test_df = pd.concat([test_df, test_remind], axis=1)
    return train_df[org_cols_train], test_df[org_cols_test]

def replace_values(series1, series2, min_value, max_value):
    series1[series1.iloc[:, 0] < min_value] = min_value
    series2[series2.iloc[:, 0] < min_value] = min_value
    series1[series1.iloc[:, 0] > max_value] = max_value
    series2[series2.iloc[:, 0] > max_value] = max_value
    return series1, series2

def outlier_indices(series1, min_value, max_value):
    ind0=series1[series1.iloc[:,0] < min_value].index
    ind1=series1[series1.iloc[:,0] > max_value].index
    return ind0, ind1

def replace_categorical(train, test, target):
    y = train[[target]]
    train_df = train.drop(columns=target)
    train_cat = train_df.select_dtypes(["object", "category"])
    if train_cat.shape[1] == 0:
        return train, test
    test_cat = test.select_dtypes(["object", "category"])
    train_df = train_df.drop(train_cat.columns, axis=1)
    test = test.drop(test_cat.columns, axis=1)
    train_cat = pd.get_dummies(train_cat, prefix=train_cat.columns)
    test_cat = pd.get_dummies(test_cat, prefix=test_cat.columns)
    train = pd.concat([train_df, train_cat, y], axis=1)
    test = pd.concat([test, test_cat], axis=1)
    return train, test

def create_synthetic_data(X, y, method, random_state=42):
    if method == "SMOTE":
        X_fit,y_fit = SMOTE(random_state=random_state).fit_resample(X, y)
    elif method == "ADASYN":
        X_fit,y_fit = ADASYN(random_state=random_state).fit_resample(X, y)
    elif method == "SMOTENC":
        cols = list(X.columns)
        cat_cols = X.select_dtypes(include=["category", "boolean"]).columns
        categorical_features = [cols.index(col) for col in cat_cols]
        X_fit,y_fit = SMOTENC(random_state=random_state, categorical_features=categorical_features).fit_resample(X, y)
    elif method == "BorderlineSMOTE":
        X_fit,y_fit = BorderlineSMOTE(random_state=random_state).fit_resample(X, y)
    elif method == "SMOTEN":
        X_fit,y_fit = SMOTEN(random_state=random_state).fit_resample(X, y)
    elif method == "SMOTETomek":
        X_fit,y_fit = SMOTETomek(random_state=random_state).fit_resample(X, y)
    return X_fit, y_fit

def create_pca(X_train, X_test, X_valid, criterion):
    pca = PCA()
    pca.fit(X_train)
    include = np.cumsum(pca.explained_variance_ratio_) <= criterion
    train_pca = pd.DataFrame(pca.transform(X_train)).iloc[:,include]
    test_pca = pd.DataFrame(pca.transform(X_test)).iloc[:,include]
    valid_pca = pd.DataFrame(pca.transform(X_valid)).iloc[:,include]
    return train_pca, test_pca, valid_pca

def scale_data(X_train, X_test, X_valid):
    cols = X_train.select_dtypes(include="number").columns
    is_normal = [stats.shapiro(X_train[col])[1] >= 0.05 for col in cols]
    is_skewed = [stats.skewtest(X_train[col])[1] < 0.05 for col in np.array(cols)[~np.array(is_normal)]]
    normal_cols = cols[is_normal]
    skewed_cols = np.array(cols)[~np.array(is_normal)][is_skewed]
    reminder = np.array(cols)[~np.array(is_normal)][~np.array(is_skewed)]
    if len(normal_cols) > 0:
        scaler = StandardScaler()
        scaler.fit(X_train[normal_cols])
        X_train_transformed = scaler.transform(X_train[normal_cols])
        X_test_transformed = scaler.transform(X_test[normal_cols])
        X_valid_transformed = scaler.transform(X_valid[normal_cols])
        X_train[normal_cols] = X_train_transformed
        X_test[normal_cols] = X_test_transformed
        X_valid[normal_cols] = X_valid_transformed
    if len(skewed_cols) > 0:
        scaler = PowerTransformer(method='yeo-johnson')
        scaler.fit(X_train[skewed_cols])
        X_train_transformed = scaler.transform(X_train[skewed_cols])
        X_test_transformed = scaler.transform(X_test[skewed_cols])
        X_valid_transformed = scaler.transform(X_valid[skewed_cols])
        X_train[skewed_cols] = X_train_transformed
        X_test[skewed_cols] = X_test_transformed
        X_valid[skewed_cols] = X_valid_transformed
    if len(reminder) > 0:
        scaler = MinMaxScaler()
        scaler.fit(X_train[reminder])
        X_train_transformed = scaler.transform(X_train[reminder])
        X_test_transformed = scaler.transform(X_test[reminder])
        X_valid_transformed = scaler.transform(X_valid[reminder])
        X_train[reminder] = X_train_transformed
        X_test[reminder] = X_test_transformed
        X_valid[reminder] = X_valid_transformed
    return X_train, X_test, X_valid

def impute_na(df):
    numerical_columns = df.select_dtypes(include=['number']).columns
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    knn_imputer = KNNImputer()
    df[numerical_columns] = knn_imputer.fit_transform(df[numerical_columns])
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_columns] = categorical_imputer.fit_transform(df[categorical_columns])
    return df