
import argparse
from performance.feature_selection import VCFDataset, Classifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='''\
Evaluate model performance
-------------------------
Example of use

python evaluate.py --happy path/to/NA12878.vcf.happy.vcf --target path/to/NA12878.vcf --mode SNP --num_trees 150 --kind RF
            ''')
    requiredNamed = parser.add_argument_group("required named arguments")
    requiredNamed.add_argument("--happy", help="hap.py annoted target VCF file", required=True)
    requiredNamed.add_argument("--target", help="target pipeline VCF file", required=True)
    requiredNamed.add_argument("--mode", help="mode, SNP or INDEL", required=True,
            choices=["SNP", "INDEL"])

    optional = parser.add_argument_group("optional arguments")
    optional.add_argument("-n", "--num_trees", help="number of trees, default = 150", type=int, default=150)
    optional.add_argument("--kind", choices=["LGBM", "LightGBM"], type=str, default="RF",
            help="..")
    optional.add_argument("--performance",
                          choices=["ALL"],
                          type=str,
                          help="Model Performance Assessment, available values: ALL")

    args = parser.parse_args()
    vcf_hap = args.happy
    vcf_tgt = args.target
    mode = args.mode
    n_trees = args.num_trees
    kind = args.kind

    # 임계값의 비율 범위 설정 (0부터 10까지 0.01 간격)
    nan_threshold = np.linspace(0.01, 0.1, 11)
    accuracies = []
    print(nan_threshold)
    for num in nan_threshold:
        dataset = VCFDataset(vcf_hap, vcf_tgt, mode, num)
        X, y = dataset.get_dataset('*')

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        clf = Classifier(dataset.features, n_trees, kind)

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

    # 시각화
    plt.figure(figsize=(10, 6))
    plt.plot(nan_threshold, accuracies, marker='o')
    plt.title("Model Performance vs. Variance")
    plt.xlabel("Variance")
    plt.ylabel("Accuracy")
    plt.grid(True)
    save_path = r'../plt/model_performance.png'
    plt.savefig(save_path)

if __name__ == '__main__':
    main()
