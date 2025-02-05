import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

df = pd.read_excel('Top PPR RB Seasons Since 2020.xlsx')

#print(df.head())

encode_columns = ['Conference', 'College', 'Team']
df[encode_columns] = df[encode_columns].apply(lambda col: col.astype('category').cat.codes)

#print(df.head())

X = df.drop(columns=['PPR Points', 'Player', 'Best Year', 'Repeats'])
y = df['PPR Points']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.3, random_state = 42)

cat_features_indices = [X.columns.get_loc(col) for col in encode_columns]

model = CatBoostRegressor(
    iterations = 1000,
    depth = 6,
    learning_rate = 0.03,
    l2_leaf_reg = 3,
    loss_function = 'MAE',
    verbose = 100,
    early_stopping_rounds = 50
)

for col in X_train.columns:
    train_mean = np.mean(X_train[col])
    val_mean = np.mean(X_val[col])
    print(f"{col}: Train Mean = {train_mean:.2f}, Val Mean = {val_mean:.2f}")

model.fit(X_train, y_train, eval_set = (X_val, y_val), cat_features = cat_features_indices)

feature_importance = model.get_feature_importance(prettified = True)
print("Feature Importance:\n", feature_importance)

y_pred = model.predict(X_test)
print(y_pred)

print(X_train.describe())
print(X_val.describe())

mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f}")
