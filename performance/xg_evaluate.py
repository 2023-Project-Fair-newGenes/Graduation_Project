import logging
from vef import VCFDataset, Classifier

import xgboost as xg
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
import numpy as np
import optuna
from sklearn.metrics import roc_curve, auc

FORMAT = '%(levelname)-7s %(asctime)-15s %(name)-15s %(message)s'
logging.basicConfig(level='INFO', format=FORMAT)
def model_evaluation(X, y, clf):
    logger = logging.getLogger('logger')
    logger.info('NA12878 ch11 데이터셋을 train과 test로 split해 테스트 진행 ')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    # 하이퍼파라미터 설정 전
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    logger.info('하이퍼 파라미터 설정 전 accuracy_score, classification_report')
    logger.info("[accuracy_score] : {}".format(accuracy_score(y_test, y_pred)))
    logger.info("[classification_report] \n{}".format(classification_report(y_test, y_pred)))

    # 최적의 하이퍼파라미터 값 출력
    best_params = {'n_estimators':200, 'learning_rate':0.1, 'max_depth': 3}
    print("Best Hyperparameters:", best_params)

    # 최적의 하이퍼파라미터로 모델 훈련
    dtrain = xg.DMatrix(data=X_train, label=y_train)
    dtest = xg.DMatrix(data=X_test, label=y_test)

    best_model = xg.train(best_params, dtrain, 100)

    # 테스트 데이터로 모델 평가
    y_p = np.rint(best_model.predict(dtest))
    logger.info('하이퍼 파라미터 설정 후 accuracy_score, classification_report')
    logger.info("[accuracy_score] : {}".format(accuracy_score(y_test, y_p)))
    logger.info("[classification_report] \n{}".format(classification_report(y_test, y_p)))

    # ROC curve로 시각화
    y_probs = clf.predict_proba(X_test)[:, 1]

    pre_tuned_fpr, pre_tuned_tpr, thresholds = roc_curve(y_test, y_probs)
    roc_auc_pre = auc(pre_tuned_fpr, pre_tuned_tpr)
    best_tuned_fpr, best_tuned_tpr, thresholds = roc_curve(y_test, y_p)
    roc_auc_best = auc(best_tuned_fpr, best_tuned_tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(pre_tuned_fpr, pre_tuned_tpr, color='blue', lw=2,
             label='Pre-tuned ROC curve (area = %0.2f)'.format(roc_auc_pre))
    plt.plot(best_tuned_fpr, best_tuned_tpr, color='darkorange', lw=2,
             label='Best-tuned ROC curve (area = %0.2f)'.format(roc_auc_best))

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    # plt.show()
    save_path = 'model_performance_XG.png'
    plt.savefig(save_path)

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

    model = xg.train(params, xg.Dataset(X_train, label=y_train), 100)
    y_pred = np.rint(model.predict(X_test))
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy
