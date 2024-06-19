import pandas as pd
import numpy as np
import multiprocessing as mp
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge
from pytorch_tabnet.tab_model import TabNetRegressor
#from catboost import CatBoostRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split


models = {
    "SVR_Linear": SVR(kernel='linear'),
    "SVR_RBF": SVR(kernel='rbf'),
    "SVR_Polynomial": SVR(kernel='poly'),
    "XGBoost": XGBRegressor(n_estimators=50, max_depth=6, eta=0.1),
    "Random_Forest": RandomForestRegressor(n_estimators=25, n_jobs=mp.cpu_count()),
    "MLP_256_128": MLPRegressor(hidden_layer_sizes=(256, 128)),
    "MLP_256_128_64": MLPRegressor(hidden_layer_sizes=(256, 128, 64), alpha=0.001),
    "MLP_256": MLPRegressor(hidden_layer_sizes=(256,), alpha=0.001),
    "Gradient_Boosting": GradientBoostingRegressor(n_estimators=25),
    "Decision_Tree": DecisionTreeRegressor(max_depth=5),
    "KNN_5": KNeighborsRegressor(n_neighbors=5, n_jobs=mp.cpu_count()),
    "KNN_10": KNeighborsRegressor(n_neighbors=10, n_jobs=mp.cpu_count()),
    "KNN_20": KNeighborsRegressor(n_neighbors=20, n_jobs=mp.cpu_count()),
    "Linear Regression": LinearRegression(n_jobs=mp.cpu_count()),
    "Ridge": Ridge(),
    "TabNet_GLU_2": TabNetRegressor(n_d=12, n_a=12, n_steps=6, n_shared=2, verbose=0),    
    "TabNet_GLU_5": TabNetRegressor(n_d=4, n_a=4, n_steps=3, n_shared=5, verbose=0),
    "AdaBoost": AdaBoostRegressor(n_estimators=25),
    "Bagging": BaggingRegressor(n_estimators=50, n_jobs=mp.cpu_count()),
    "ExtraTrees": ExtraTreesRegressor(n_estimators=75, n_jobs=mp.cpu_count()),
    "HistGradientBoosting": HistGradientBoostingRegressor(),
   # "CatBoost": CatBoostRegressor(iterations=50, task_type="CPU", thread_count=mp.cpu_count()),
}


def train_and_predict(model_name, model, x_train, y_train, x_test):
    if model_name == 'TabNet_GLU_2' or model_name == 'TabNet_GLU_5':
        model.fit(x_train, y_train.reshape(-1,1),patience=10, max_epochs = 30)
    else:
        model.fit(x_train, y_train)
    return model.predict(x_test)

def train_models(dataset):

    if dataset == 'law_school_edited':
        data = pd.read_csv('law_school_edited.csv')
        X = data.drop('y', axis=1)
        y = data.y
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
        protected = X_test['race'].values

    elif dataset == 'insurance_edited':
        data = pd.read_csv('insurance_edited.csv')
        X = data.drop(['charges'], axis=1)
        y = data.charges
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
        protected = X_test['sex'].values
        X_train = X_train.drop(['sex'], axis=1)
        X_test = X_test.drop(['sex'], axis=1)

    elif dataset == 'crime_edited':
        data = pd.read_csv('crime_edited.csv')
        X = data.drop('ViolentCrimesPerPop', axis=1)
        y = data.ViolentCrimesPerPop

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
        protected = np.where(X_test.racepctblack >= 0.5, 0, 1)

    else:
        return 'Dataset not found'

    c = 0
    results_df = pd.DataFrame()
    print(f'Models training begins {dataset}')
    for model_name, model in models.items():

        pred = train_and_predict(model_name, model, X_train.values, y_train.values, X_test.values)
        results_df[model_name] = pred
        print(f'    {model_name} completed. ')
        c+=1
    print(f'Models training completed {dataset}')

    results_df['protected'] = protected
    results_df['y_test'] = y_test.values
    return results_df
