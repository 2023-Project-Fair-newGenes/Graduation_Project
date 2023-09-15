from lightgbm import plot_importance
import matplotlib.pyplot as plt
from src import VCFDataset, Classifier
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from xgboost import XGBClassifier

import logging

FORMAT = '%(levelname)-7s %(asctime)-15s %(name)-15s %(message)s'
logging.basicConfig(level='INFO', format=FORMAT)

def main():
    logger = logging.getLogger('TEST')

    vcf_hap = "../dataset/NA12878_chr11.indel.vcf.gz.happy.vcf.gz"
    vcf_tgt = "../dataset/NA12878_chr11.indel.vcf.gz"
    mode = "INDEL"
    n_trees = 150
    kind = "LGBM"

    # 데이터셋 준비
    logger.info('-----데이터 셋 준비-----')
    dataset = VCFDataset(vcf_hap, vcf_tgt, mode)
    X, y = dataset.get_dataset('*')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    lgbm_model = lgb.LGBMClassifier(n_estimators=n_trees, force_row_wise=True, num_leaves = 96, learning_rate = 0.024733289023679998,
                                          feature_fraction = 0.8439020417557227, bagging_fraction = 0.21552726628147978, min_child_samples = 86)

    lgbm_model.fit(X_train, y_train)

    print(dataset.features)

    ax = lgb.plot_importance(lgbm_model, max_num_features=8, importance_type='split')
    ax.set(title=f'Feature Importance (split)',
           xlabel='Feature Importance',
           ylabel='Features')
    ax.figure.savefig('../plt_results/LGBM_feature_importance.png', dpi=300)

    plt.clf()
    xgb = XGBClassifier(n_estimators=200, objective='binary:logistic', max_depth=3, learning_rate=0.1)
    xgb.fit(X_train, y_train)

    plt.barh(dataset.features, xgb.feature_importances_)
    plt.savefig('../plt_results/XGB_feature_importance.png')

if __name__ == '__main__':
    main()