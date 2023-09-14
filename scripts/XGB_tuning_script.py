"""
Train ch11 dataset, test ch20 dataset
"""
from src import VCFDataset, Classifier
import xgboost as xg

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, auc, accuracy_score, classification_report
from sklearn.model_selection import train_test_split

import logging

FORMAT = '%(levelname)-7s %(asctime)-15s %(name)-15s %(message)s'
logging.basicConfig(level='INFO', format=FORMAT)

def main():
    logger = logging.getLogger('Test')
    logger.info('-----NA12878 ch11 데이터셋을 train 70%과 test 30%으로 split해 테스트 진행-----')

    vcf_hap = "../dataset/NA12878_chr11.indel.vcf.gz.happy.vcf.gz"
    vcf_tgt = "../dataset/NA12878_chr11.indel.vcf.gz"
    mode = "INDEL"
    n_trees = 150
    kind = "GB"

    # 데이터셋 준비
    logger.info('-----데이터 셋 준비-----')
    dataset = VCFDataset(vcf_hap, vcf_tgt, mode)
    X, y = dataset.get_dataset('*')

    clf = Classifier(dataset.features, n_trees, kind)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    # 하이퍼파라미터 설정 전
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    logger.info('하이퍼 파라미터 설정 전 accuracy_score, classification_report')
    logger.info("[accuracy_score] : {}".format(accuracy_score(y_test, y_pred)))
    logger.info("[classification_report] \n{}".format(classification_report(y_test, y_pred)))

    # 최적의 하이퍼파라미터 값 출력
    best_params = {'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 3}
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
    save_path = r'../plt_results/model_performance_XGB_pre&best.png'
    plt.savefig(save_path)

if __name__ == '__main__':
    main()
