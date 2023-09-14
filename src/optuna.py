import lightgbm as lgb
import numpy as np
from sklearn.metrics import accuracy_score

# Objective 함수 정의
def objective(trial, X_train, y_train, X_test, y_test):
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.1, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.1, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
    }

    model = lgb.train(params, lgb.Dataset(X_train, label=y_train), 100)
    y_pred = np.rint(model.predict(X_test))
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy