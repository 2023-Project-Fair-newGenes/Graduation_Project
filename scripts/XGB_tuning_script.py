"""
Train ch11 dataset, test ch20 dataset
"""
from src import VCFDataset, Classifier, XGB_tuning

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

    XGB_tuning(clf, X, y)

if __name__ == '__main__':
    main()
