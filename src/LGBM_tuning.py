from src import objective

import lightgbm as lgb
import optuna
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, auc, accuracy_score, classification_report
from sklearn.model_selection import train_test_split

import logging

FORMAT = '%(levelname)-7s %(asctime)-15s %(name)-15s %(message)s'
logging.basicConfig(level='INFO', format=FORMAT)

def LGBM_tuning(clf, X, y):
    logger = logging.getLogger('Test')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    # 하이퍼파라미터 설정 전
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    logger.info('하이퍼 파라미터 설정 전 accuracy_score, classification_report')
    logger.info("[accuracy_score] : {}".format(accuracy_score(y_test, y_pred)))
    logger.info("[classification_report] \n{}".format(classification_report(y_test, y_pred)))

    # Optuna로 하이퍼파라미터 최적화
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=100)

    # 최적의 하이퍼파라미터 값 출력
    best_params = study.best_params
    print("Best Hyperparameters:", best_params)

    # 최적의 하이퍼파라미터로 모델 훈련
    best_model = lgb.train(best_params, lgb.Dataset(X_train, label=y_train), 100)

    # 테스트 데이터로 모델 평가
    y_p = np.rint(best_model.predict(X_test))
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
    save_path = r'../plt_results/model_performance_LGBM_pre&best.png'
    plt.savefig(save_path)