import sys
sys.path.append('..')
# print(sys.path)

import argparse
from vef import VCFDataset, Classifier
# from ./performance/ import *
from performance.performance_evaluate import *
def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='''\
Evaluate model performance
-------------------------
Example of use

python performance_evaluate.py --happy path/to/NA12878.vcf.happy.vcf --target path/to/NA12878.vcf --mode SNP --num_trees 150 --kind RF
            ''')
    requiredNamed = parser.add_argument_group("required named arguments")
    requiredNamed.add_argument("--happy", help="hap.py annoted target VCF file", required=True)
    requiredNamed.add_argument("--target", help="target pipeline VCF file", required=True)
    requiredNamed.add_argument("--mode", help="mode, SNP or INDEL", required=True,
            choices=["SNP", "INDEL"])

    optional = parser.add_argument_group("optional arguments")
    optional.add_argument("-n", "--num_trees", help="number of trees, default = 150", type=int, default=150)
    optional.add_argument("--kind", choices=["SVM", "SUPPORTVECTOR"], type=str, default="SVM",
            help="..")

    args = parser.parse_args()
    vcf_hap = args.happy
    vcf_tgt = args.target
    mode = args.mode
    n_trees = args.num_trees
    kind = args.kind

    dataset = VCFDataset(vcf_hap, vcf_tgt, mode)
    X, y = dataset.get_dataset('*')

    clf = Classifier(dataset.features, n_trees, kind)

    model_evaluation(X, y, clf)

if __name__ == '__main__':
    main()