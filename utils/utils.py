from sklearn.base import BaseEstimator, TransformerMixin
import datetime
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer


def evaluate_models(X, y, test_size=0.2, cv=5, random_state=42):
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_squared_error , r2_score
    from sklearn.linear_model import LinearRegression, ElasticNet
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import SVR
    from xgboost import XGBRegressor
    import pandas as pd


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    models = {
        "Linear Regression": LinearRegression(),
        "Elastic Net": ElasticNet(random_state=random_state),
        "Random Forest": RandomForestRegressor(random_state=random_state),
        "XGBoost": XGBRegressor(random_state=random_state, verbosity=0),
        "KNN Regressor": KNeighborsRegressor(),
        "SVR" : SVR(kernel='rbf', C=1.0, epsilon=0.1)
    }

    results = []

    for name, model in models.items():
        neg_mse_scores = cross_val_score(
            model, X_train, y_train, 
            scoring='neg_root_mean_squared_error', 
            cv=cv
        )
        rmse_scores = -neg_mse_scores

        r2_scores = cross_val_score(
            model, X_train, y_train, 
            scoring='r2', 
            cv=cv
        )

        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        train_rmse = mean_squared_error(y_train, y_pred_train)
        train_r2 = r2_score(y_train, y_pred_train)

        test_rmse = mean_squared_error(y_test, y_pred_test)
        test_r2 = r2_score(y_test, y_pred_test)

        results.append({
            "Model": name,
            "Train RMSE": train_rmse,
            "Train R2": train_r2,
            "Test RMSE": test_rmse,
            "Test R2": test_r2,
            "CV RMSE Mean": rmse_scores.mean(),
            "CV RMSE Std": rmse_scores.std(),
            "CV R2 Mean": r2_scores.mean(),
            "CV R2 Std": r2_scores.std()
        })

    return pd.DataFrame(results).sort_values(by="Test RMSE")


def coordinate_imputer(df, column, map_lat, map_long):
    df[f"{column}Latitude"] = df[column].map(map_lat)
    df[f"{column}Longitude"] = df[column].map(map_long)
    return df

def convert_to_day_of_year(row, year_column, month_column, day_column):
    import datetime
    date = datetime.date(row[year_column], row[month_column], row[day_column])
    return date.timetuple().tm_yday

def knn_impute_numeric(df, numeric_columns, n_neighbors=5):
    imputer = KNNImputer(n_neighbors=n_neighbors)
    
    # Skopiuj dataframe, żeby nie nadpisywać oryginału
    df_copy = df.copy()
    
    # Imputacja - tylko dla wskazanych kolumn
    df_copy[numeric_columns] = imputer.fit_transform(df_copy[numeric_columns])
    
    return df_copy