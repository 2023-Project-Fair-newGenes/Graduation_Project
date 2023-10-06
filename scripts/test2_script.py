"""
Train ch11 dataset, test ch20 dataset
"""
from src import VCFDataset, Classifier
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.model_selection import train_test_split

import logging

FORMAT = '%(levelname)-7s %(asctime)-15s %(name)-15s %(message)s'
logging.basicConfig(level='INFO', format=FORMAT)

def main():
    logger = logging.getLogger('Test')
    logger.info('-----NA12878 ch11 데이터셋 train, ch20 데이터 셋 test 진행-----')

    mode = "INDEL"
    n_trees = 150

    # train 데이터셋 준비
    vcf_hap_train = "../dataset/NA12878_chr11.indel.vcf.gz.happy.vcf.gz"
    vcf_tgt_test = "../dataset/NA12878_chr11.indel.vcf.gz"

    logger.info('-----train 데이터 셋 준비-----')
    dataset_train = VCFDataset(vcf_hap_train, vcf_tgt_test, mode)
    X_train, y_train = dataset_train.get_dataset('*')

    # test 데이터셋 준비
    vcf_hap_test = "../dataset/NA12878_chr20.indel.vcf.gz.happy.vcf.gz"
    vcf_tgt_test = "../dataset/NA12878_chr20.indel.vcf.gz"

    logger.info('-----test 데이터 셋 준비-----')
    dataset_test = VCFDataset(vcf_hap_test, vcf_tgt_test, mode)
    X_test, y_test = dataset_test.get_dataset('*')

    # 모델 초기화
    #rf_model = Classifier(dataset_train.features, n_trees, 'RF')
    lgbm_model = Classifier(dataset_train.features, n_trees, 'LGBM')
    svm_model = Classifier(dataset_train.features, n_trees, 'SVM')
    xgb_model = Classifier(dataset_train.features, n_trees, 'XG')

    logger.info('-----train 진행-----')
    # 각 모델 학습
    #rf_model.fit(X_train, y_train)
    lgbm_model.fit(X_train, y_train)
    svm_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)

    logger.info('-----test 진행-----')
    # 각 모델 예측 확률
    #rf_probs = rf_model.predict_proba(X_test)[:, 1]
    lgbm_probs = lgbm_model.predict_proba(X_test)[:, 1]
    svm_probs = svm_model.predict_proba(X_test)[:, 1]
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]

    logger.info('-----ROC, AUC 계산-----')
    # AUC 계산
    #rf_auc = roc_auc_score(y_test, rf_probs)
    lgbm_auc = roc_auc_score(y_test, lgbm_probs)
    svm_auc = roc_auc_score(y_test, svm_probs)
    xgb_auc = roc_auc_score(y_test, xgb_probs)

    # ROC 곡선 생성
    #rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
    lgbm_fpr, lgbm_tpr, _ = roc_curve(y_test, lgbm_probs)
    svm_fpr, svm_tpr, _ = roc_curve(y_test, svm_probs)
    xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_probs)

    # ROC 곡선 시각화
    plt.figure(figsize=(10, 7))
    #plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {rf_auc:.2f})')
    plt.plot(lgbm_fpr, lgbm_tpr, label=f'LightGBM (AUC = {lgbm_auc:.3f})')
    plt.plot(svm_fpr, svm_tpr, label=f'SVM (AUC = {svm_auc:.3f})')
    plt.plot(xgb_fpr, xgb_tpr, label=f'XGBoost (AUC = {xgb_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')  # diagonal dotted line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve\nch11 train, ch20 test')
    plt.legend(loc='lower right')
    save_path = r'../plt_results/performance_by_model_test2.png'
    plt.savefig(save_path)

    logger.info('-----TEST 완료-----')

if __name__ == '__main__':
    main()