#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2018 Chuanyi Zhang <chuanyi5@illinois.edu>
#
# Distributed under terms of the MIT license.

"""
Train and save VEF classifiers
"""
import argparse
from vef import VCFDataset, Classifier
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.model_selection import train_test_split



def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='''\
Train a filter
-------------------------
Example of use

python vef_clf.py --happy path/to/NA12878.vcf.happy.vcf --target path/to/NA12878.vcf --mode SNP --num_trees 150 --kind RF
            ''')
    requiredNamed = parser.add_argument_group("required named arguments")
    requiredNamed.add_argument("--happy", help="hap.py annoted target VCF file", required=True)
    requiredNamed.add_argument("--target", help="target pipeline VCF file", required=True)
    requiredNamed.add_argument("--mode", help="mode, SNP or INDEL", required=True,
            choices=["SNP", "INDEL"])

    optional = parser.add_argument_group("optional arguments")
    optional.add_argument("-n", "--num_trees", help="number of trees, default = 150", type=int, default=150)

    args = parser.parse_args()
    vcf_hap = args.happy
    vcf_tgt = args.target
    mode = args.mode
    n_trees = args.num_trees

    dataset = VCFDataset(vcf_hap, vcf_tgt, mode)
    X, y = dataset.get_dataset('*')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 모델 초기화
    rf_model = Classifier(dataset.features, n_trees, 'RF')
    lgbm_model = Classifier(dataset.features, n_trees, 'LGBM')
    svm_model = Classifier(dataset.features, n_trees, 'SVM')
    xgb_model = Classifier(dataset.features, n_trees, 'XG')

    # 각 모델 학습
    rf_model.fit(X_train, y_train)
    lgbm_model.fit(X_train, y_train)
    svm_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)

    # 각 모델 예측 확률
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    lgbm_probs = lgbm_model.predict_proba(X_test)[:, 1]
    svm_probs = svm_model.predict_proba(X_test)[:, 1]
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]

    # AUC 계산
    rf_auc = roc_auc_score(y_test, rf_probs)
    lgbm_auc = roc_auc_score(y_test, lgbm_probs)
    svm_auc = roc_auc_score(y_test, svm_probs)
    xgb_auc = roc_auc_score(y_test, xgb_probs)

    # ROC 곡선 생성
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
    lgbm_fpr, lgbm_tpr, _ = roc_curve(y_test, lgbm_probs)
    svm_fpr, svm_tpr, _ = roc_curve(y_test, svm_probs)
    xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_probs)

    # ROC 곡선 시각화
    plt.figure(figsize=(10, 7))
    plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {rf_auc:.2f})')
    plt.plot(lgbm_fpr, lgbm_tpr, label=f'LightGBM (AUC = {lgbm_auc:.2f})')
    plt.plot(svm_fpr, svm_tpr, label=f'SVM (AUC = {svm_auc:.2f})')
    plt.plot(xgb_fpr, xgb_tpr, label=f'XGBoost (AUC = {xgb_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # diagonal dotted line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    save_path = r'../plt/performance_by_model.png'
    plt.savefig(save_path)


if __name__ == '__main__':
    main()