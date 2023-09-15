from src.feature_selection import VCFDataset, Classifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def main():

    vcf_hap = "../dataset/NA12878_chr11.indel.vcf.gz.happy.vcf.gz"
    vcf_tgt = "../dataset/NA12878_chr11.indel.vcf.gz"
    mode = "INDEL"
    n_trees = 150
    kind = "LGBM"

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

    print(accuracies)

    # 시각화
    # plt.figure(figsize=(10, 6))
    # plt.plot(nan_threshold, accuracies, marker='o')
    # plt.title("Model Performance vs. Variance")
    # plt.xlabel("Variance")
    # plt.ylabel("Accuracy")
    # plt.grid(True)
    # save_path = r'../plt_results/model_performance_missingvalues.png'
    # plt.savefig(save_path)


if __name__ == '__main__':
    main()