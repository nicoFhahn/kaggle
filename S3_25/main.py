import pandas as pd
from src import visualization, preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
le = LabelEncoder()
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
train = pd.read_csv("S3_26/train.csv")
test = pd.read_csv("S3_26/test.csv")
train["Stage"] = train["Stage"].astype("category")
test["Stage"] = test["Stage"].astype("category")

# visualization.plot_distplots(train.drop(columns=["id", "Hardness"]))
train_clean, test_clean = preprocessing.remove_outliers(train.drop(columns=["id", "Status"]), test.drop(columns=["id"]), contamination=0.1, method="nearest", n_neighbors=20)

train = pd.concat([train[["id", "Status"]], train_clean], axis=1)
test = pd.concat([test[["id"]], test_clean], axis=1)

train, test = preprocessing.replace_categorical(train, test, "Status")
X_train, y_train = train.drop(columns=["id", "Status"]), le.fit_transform(train[["Status"]])
# visualization.plot_distplots(train.drop(columns=["id", "Hardness"]))

X_train, y_train = preprocessing.create_synthetic_data(X_train, y_train, method="SMOTE")
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=.25, random_state=42, stratify=y_train)
X_train, X_test, X_valid = preprocessing.scale_data(X_train, X_test, test.drop(columns="id"))
X_train, X_test, X_valid = preprocessing.create_pca(X_train, X_test, X_valid, criterion = 0.95)

