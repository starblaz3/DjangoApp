import pandas as pd
import numpy as np
import os.path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Evaluate the models
def evaluate_model(y_true, y_pred, model_name, mode):
    if mode == '1':
        e = mean_absolute_error(y_true, y_pred)
        return (f"{model_name} - Mean Absolute Error (MAE): {e:.2f}")
    elif mode == '2':
        e = mean_squared_error(y_true, y_pred)
        return(f"{model_name} - Mean Squared Error (MSE): {e:.2f}")
    else:
        e = np.sqrt(mean_squared_error(y_true, y_pred))
        return(f"{model_name} - Root Mean Squared Error (RMSE): {e:.2f}")

# Initialize and train Linear Regression model
def lr(mode,X_train, X_test, y_train, y_test):
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)

    evalString=evaluate_model(y_test, y_pred_lr, "Linear Regression",mode)
    return lr_model,evalString

# Initialize and train Random Forest model
def rf(mode,X_train, X_test, y_train, y_test):
    
    rf_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf_grid_search = GridSearchCV(RandomForestRegressor(), rf_param_grid, cv=5, scoring='neg_mean_squared_error')
    rf_grid_search.fit(X_train, y_train)
    rf_best_model = rf_grid_search.best_estimator_
    y_pred_rf = rf_best_model.predict(X_test)

    evalString=evaluate_model(y_test, y_pred_rf, "Random Forest",mode)
    return rf_best_model,evalString

# Initialize and train Support Vector Regression (SVR) model
def svr(mode,X_train, X_test, y_train, y_test):
    svr_param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'poly', 'rbf']
    }
    svr_grid_search = GridSearchCV(SVR(), svr_param_grid, cv=5, scoring='neg_mean_squared_error')
    svr_grid_search.fit(X_train, y_train)
    svr_best_model = svr_grid_search.best_estimator_
    y_pred_svr = svr_best_model.predict(X_test)

    evalString=evaluate_model(y_test, y_pred_svr, "SVR",mode)
    return svr_best_model,evalString

# Initialize and train K-nearest neighbor (KNN) model
def knn(mode,X_train, X_test, y_train, y_test):

    knn_param_grid = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance']
    }
    knn_grid_search = GridSearchCV(KNeighborsRegressor(), knn_param_grid, cv=5, scoring='neg_mean_squared_error')
    knn_grid_search.fit(X_train, y_train)
    knn_best_model = knn_grid_search.best_estimator_
    y_pred_knn = knn_best_model.predict(X_test)

    evaluate_model(y_test, y_pred_knn, "KNN",mode)
    return knn_best_model

# Initialize and train Gradient Boosting model
def gb(mode,X_train, X_test, y_train, y_test):
    gb_param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5]
    }
    gb_grid_search = GridSearchCV(GradientBoostingRegressor(), gb_param_grid, cv=5, scoring='neg_mean_squared_error')
    gb_grid_search.fit(X_train, y_train)
    gb_best_model = gb_grid_search.best_estimator_
    y_pred_gb = gb_best_model.predict(X_test)

    evaluate_model(y_test, y_pred_gb, "Gradient Boosting",mode)
    return gb_best_model

def api(selected_model,mode,path):    
    # Load and preprocess the data
    data = pd.read_csv(os.path.join(path,'data.csv'))  # Replace 'your_data.csv' with your file path
    X = data['air_temperature'].values.reshape(-1, 1)
    y = data['water_temperature'].values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    if selected_model == '1':
        model,evalString = lr(mode,X_train, X_test, y_train, y_test)
    elif selected_model == '2':
        model,evalString = rf(mode,X_train, X_test, y_train, y_test)
    elif selected_model == '3':
        model,evalString = svr(mode,X_train, X_test, y_train, y_test)
    elif selected_model == '4':
        model = knn(mode,X_train, X_test, y_train, y_test)
    else:
        model = gb(mode,X_train, X_test, y_train, y_test)
    y_pred_all = model.predict(X)

    # Create a DataFrame to store the predictions
    predictions_df = pd.DataFrame({'AirTemperature': X.flatten(), 'water_temperature': y_pred_all})    
    # Save the predictions to a CSV file
    predictions_df.to_csv(os.path.join(path,'predictions.csv'), index=False)
    return evalString