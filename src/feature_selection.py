#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
vef.core
~~~~~~~~

This module implements the core features of VEF.

:copyright: © 2018 by Chuanyi Zhang.
:license: MIT, see LICENSE for more details.
"""

import allel
import time
import logging
import csv
import os
import binascii
import gzip
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import joblib
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import lightgbm as lgb
import pandas as pd
from xgboost import XGBClassifier

FORMAT = '%(levelname)-7s %(asctime)-15s %(name)-15s %(message)s'
logging.basicConfig(level='INFO', format=FORMAT)


class _VCFExtract:
    """Extract data from VCF file."""

    def __init__(self, filepath):
        self.filepath = filepath
        self.fields, self.samples, self.header, _ = allel.iter_vcf_chunks(filepath, fields='*')
        self.features = None
        self.variants = None
        self.logger = logging.getLogger(self.__class__.__name__)

    def fetch_data(self, mode, features=None):
        """Fetch data from VCF file.

        :param str mode: kind of variants training on: 'SNP', 'INDEL' or 'BOTH'.
        :param list features:
        :returns union: data (Numpy.array), features (list), vartype_index (Numpy.array).
        """
        VAR_PREFIX = 'variants/'
        if features is None:
            fields = [(VAR_PREFIX + k) for k in
                      list(self.header.infos.keys()) + ['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL']]
            data = allel.read_vcf(self.filepath, fields='*')
            self.features = [ftr for ftr in fields if np.issubdtype(data[ftr].dtype, np.number)]
        else:
            self.features = features
            for i in list(self.features):
                if i not in self.fields:
                    self.logger.error("Error: {} field not in {}, we have {}".format(i, self.filepath, self.fields))
                    exit(-1)
            data = allel.read_vcf(self.filepath, fields='*')
        if mode.upper() == 'SNP':
            is_vartype = data[VAR_PREFIX + 'is_snp']
        elif mode.upper() == 'INDEL':
            is_vartype = np.logical_not(data[VAR_PREFIX + 'is_snp'])
        elif mode.upper() == 'BOTH':
            is_vartype = np.repeat(True, data[VAR_PREFIX + 'is_snp'].shape[0])
        else:
            self.logger.warning("No such mode {}, using mode SNP.".format(mode))
            is_vartype = np.repeat(True, data[VAR_PREFIX + 'is_snp'].shape[0])
        vartype_index = np.where(is_vartype)[0]
        annotes = [data[ftr][is_vartype] for ftr in self.features]
        annotes = np.vstack([c if c.ndim == 1 else c[:, 0] for c in annotes])
        return annotes.transpose(), self.features, vartype_index

    @staticmethod
    def mend_nan(features, axis=1):
        mean = np.nanmean(features, axis=0)
        nan_idx = np.where(np.isnan(features))
        features[nan_idx] = mean[nan_idx[1]]


class VCFDataset_FS:
    """Compare VCF files and prepare features/labels for training.

    :param str hap_filepath: filepath to happy output VCF.
    :param str specimen_filepath: filepath to specimen VCF.
    :param str mode: kind of variants training on: 'SNP' or 'INDEL'.
    """

    def __init__(self, hap_filepath, specimen_filepath, mode, num):
        self.hap_vcf = _VCFExtract(hap_filepath)
        self.specimen_vcf = _VCFExtract(specimen_filepath)
        self.dataset = {}
        self.contigs = []
        self.features = []
        self.logger = logging.getLogger(self.__class__.__name__)
        self._compare(mode, num)

    @staticmethod
    def _extract_factory(truth_idx, vartype):
        """
        Create function that check if this variant is vartype
        :param truth_idx: the index of 'TRUTH' in list of samples in outcome of hap.py
        :param str vartype: variant type, SNP or INDEL

        :returns function: the function that check if vartype
        """

        def inner(arr):
            if arr[truth_idx] == vartype and arr[1 - truth_idx] == vartype:
                return 1
            elif arr[truth_idx] == 'NOCALL' and arr[1 - truth_idx] == vartype:
                return 0
            else:
                return -1

        return inner

    def _compare(self, mode, num):
        self.logger.info("Start extracting label from {}".format(os.path.abspath(self.hap_vcf.filepath)))
        VAR_PREFIX = 'variants/'
        mode_list = ['SNP', 'INDEL']
        data = allel.read_vcf(self.hap_vcf.filepath,
                              fields=['calldata/BVT', 'variants/Regions', 'variants/POS', 'variants/CHROM'])
        conf_mask = data['variants/Regions'].astype(bool)
        self.logger.info("Total variants(hap.py): {}, in high-conf region variants: {}".format(conf_mask.shape[0],
                                                                                               int(sum(conf_mask))))
        if mode.upper() in mode_list:
            extract_target_vartype = self._extract_factory(np.where(self.hap_vcf.samples == 'TRUTH')[0][0],
                                                           mode.upper())
        else:
            self.logger.warning("Warning: mode {} not exist. Using SNP mode.".format(mode))
            extract_target_vartype = self._extract_factory(np.where(self.hap_vcf.samples == 'TRUTH')[0][0], 'SNP')

        vartype = np.apply_along_axis(extract_target_vartype, 1, data['calldata/BVT'][conf_mask, :])
        label_list = np.vstack((data['variants/POS'][conf_mask], vartype))
        idx = (vartype != -1)
        label_list = label_list[:, idx]
        chrom_list = data['variants/CHROM'][conf_mask][idx]
        chroms = np.unique(chrom_list)
        label_list = {chrom: label_list[:, np.where(chrom_list == chrom)[0]] for chrom in chroms}
        for key in label_list:
            _, idx, cnt = np.unique(label_list[key][0, :], return_index=True, return_counts=True)
            label_list[key] = label_list[key][:, idx[cnt <= 1]]
        self.logger.info("Finish extracting label from file")
        data = allel.read_vcf(self.specimen_vcf.filepath, fields='*')

        self.logger.info("Start extracting variants from {}".format(os.path.abspath(self.specimen_vcf.filepath)))
        # feature selection
        num_var = np.shape(data[VAR_PREFIX + 'REF'])[0]
        self.features = [(VAR_PREFIX + k) for k in list(self.specimen_vcf.header.infos.keys()) + ['QUAL']]
        # 주어진 데이터의 타입이 숫자 데이터 타입인지 확인
        self.features = [ftr for ftr in self.features if np.issubdtype(data[ftr].dtype, np.number)]
        self.logger.info("결측치가 {}% 이상인 경우 feature 제외하고 feature selection 진행 ".format(num))
        # self.features = [ftr for ftr in self.features if np.sum(np.isnan(data[ftr])) < num * num_var]
        self.features = [ftr for ftr in self.features if np.nanvar(data[ftr]) >= num]
        features_avoid = [VAR_PREFIX + 'VQSLOD']
        for ftr in features_avoid:
            if ftr in self.features:
                self.features.remove(ftr)
        self.features.sort()

        # merge features, with CHROM, POS
        annotes = [data[ftr] for ftr in [VAR_PREFIX + 'POS'] + self.features]
        annotes = np.vstack([c if c.ndim == 1 else c[:, 0] for c in annotes])
        chrom_list = data[VAR_PREFIX + 'CHROM']
        self.contigs = list(label_list)
        annotes_chrom = {ch: annotes[:, np.where(chrom_list == ch)[0]] for ch in self.contigs}

        variant_count = 0
        positive_count = 0
        for ch in self.contigs:
            if ch not in label_list:
                continue
            else:
                annotes_idx = np.where(np.isin(annotes_chrom[ch][0], label_list[ch][0]))[0]
                label_idx = np.where(np.isin(label_list[ch][0], annotes_chrom[ch][0]))[0]
                self.dataset[ch] = np.vstack(
                    (annotes_chrom[ch][1:, annotes_idx], label_list[ch][1, label_idx])).transpose()
                self.logger.debug("CHROM:{}, {} {}".format(ch, np.shape(annotes_chrom[ch][1:, annotes_idx]),
                                                           np.shape(label_list[ch][1, label_idx])))
                idx = ~np.any(np.isnan(self.dataset[ch]), axis=1)
                self.dataset[ch] = {'X': self.dataset[ch][idx, :-1], 'y': self.dataset[ch][idx, -1]}
                variant_count += self.dataset[ch]['y'].shape[0]
                positive_count += sum(self.dataset[ch]['y'])
        # np.savez_compressed(dataset_file, infos=features, chrom_data=dataset)
        self.logger.info("Finish extracting variants from file, #variants={}, #positive_sample={}, #features={}".format(
            variant_count, int(positive_count), len(self.features)))
        return self

    def get_dataset(self, contigs):
        """
        Return features and labels of request contigs. If contigs is '*', return all data.

        :param list contigs: contigs requested.
        :returns union: features: X, labels: y.
        """
        if '*' in list(contigs):
            contig_list = self.contigs
        else:
            contig_list = []
            if type(contigs) is not list:
                contigs = [contigs]
            for ctg in contigs:
                if ctg not in self.contigs:
                    self.logger.warning("Requested contig {} not exist.".format(ctg))
                contig_list.append(ctg)
        X = np.vstack([self.dataset[ctg]['X'] for ctg in contig_list])
        y = np.hstack([self.dataset[ctg]['y'] for ctg in contig_list])
        return X, y

    def save(self, output_filepath):
        np.savez_compressed(output_filepath, dataset=self.dataset)
        self.logger.info("Dataset saved to file {}".format(os.path.abspath(output_filepath)))

    @staticmethod
    def load(dataset_filepath):
        return np.load(dataset_filepath)


class Classifier_FS:
    """Ensemble classifier."""

    def __init__(self, features, n_trees=150, kind="SVM"):
        self.kind = kind

        if kind.upper() == "RF" or kind.upper() == "RANDOMFOREST":
            self.kind = "RF"
            self.clf = RandomForestClassifier(criterion='gini', max_depth=20, n_estimators=n_trees)
        elif kind.upper() == "AB" or kind.upper() == "ADABOOST":
            self.kind = "AB"
            self.clf = AdaBoostClassifier(n_estimators=n_trees)
        elif kind.upper() == "GB" or kind.upper() == "GRADIENTBOOST":
            self.kind = "GB"
            self.clf = GradientBoostingClassifier(n_estimators=n_trees)
        elif kind.upper() == "SVM" or kind.upper() == "SUPPORTVECTOR":
            self.kind = "SVM"
            self.clf = SVC(C=1000, gamma=0.0001, kernel='rbf')
        elif kind.upper() == "LGBM" or kind.upper() == "LIGHTGBM":
            self.kind = "LGBM"
            self.clf = lgb.LGBMClassifier(n_estimators=n_trees, force_row_wise=True, num_leaves=96,
                                          learning_rate=0.024733289023679998,
                                          feature_fraction=0.8439020417557227, bagging_fraction=0.21552726628147978,
                                          min_child_samples=86)
        elif kind.upper() == "XG" or kind.upper() == "XGBOOST":
            self.kind = "XG"
            self.clf = XGBClassifier(objective='binary:logistic', n_estimator = 100, learning_rate = 0.2, max_depth = 3, colsample_bytree = 0.9, min_child_weight = 5, subsample = 0.9)
        else:
            print("model is " + kind)
            logger = logging.getLogger(self.__class__.__name__)
            logger.error("No such type of classifier exist.")
        self.features = features

    def fit(self, X, y, sample_weight=None):
        logger = logging.getLogger(self.__class__.__name__)
        logger.info("Begin training model")
        t0 = time.time()
        self.clf.fit(X, y, sample_weight=sample_weight)
        logger.info("Training a".format())
        # logger.debug("Importance: {}".format(self.clf.feature_importances_))
        t1 = time.time()
        logger.info("Finish training model")
        logger.info("Elapsed time {:.3f}s".format(t1 - t0))

    def gridsearch(self, X, y, k_fold=5, n_jobs=2):

        logger = logging.getLogger(self.__class__.__name__)
        logger.info("Begin grid search")
        t0 = time.time()
        kfold = KFold(n_splits=k_fold, shuffle=True)
        if self.kind == "RF":
            parameters = {
                'n_estimators': list(range(50, 251, 10)),
            }
        elif self.kind == "GB":
            parameters = {
                'n_estimators': np.arange(50, 251, 10),
                'learning_rate': np.logspace(-5, 0, 10),
            }
        elif self.kind == "AB":
            parameters = {
                'n_estimators': np.arange(50, 251, 10),
                'learning_rate': np.logspace(-4, 0, 10),
            }
        elif self.kind == "SVM":
            parameters = {'C': [0.1, 1, 10, 100, 1000],
                          'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                          'kernel': ['rbf']
                          }
            logger.info(f"Kind: {self.kind}, {self.clf}")
            grid = GridSearchCV(self.clf, parameters, refit=True, verbose=3)
            grid.fit(X, y)
            print(grid.best_params_)
            t1 = time.time()
            logger.info("Finish training model")
            logger.info("Elapsed time {:.3f}s".format(t1 - t0))
            return

        logger.info(f"Kind: {self.kind}, {self.clf}")
        self.clf = GridSearchCV(self.clf, parameters, scoring='f1', n_jobs=n_jobs, cv=kfold, refit=True)
        self.clf.fit(X, y)
        print(self.clf.cv_results_, '\n', self.clf.best_params_)
        logger.debug("Grid_scores: {}".format(self.clf.cv_results_))
        t1 = time.time()
        logger.info("Finish training model")
        logger.info("Elapsed time {:.3f}s".format(t1 - t0))

    def save(self, output_filepath):
        logger = logging.getLogger(self.__class__.__name__)
        joblib.dump(self, output_filepath)
        logger.info("Classifier saved at {}".format(os.path.abspath(output_filepath)))

    def predict(self, *args, **kwargs):
        return self.clf.predict(*args, **kwargs)

    def predict_proba(self, *args, **kwargs):
        return self.clf.predict_proba(*args, **kwargs)

    def predict_log_proba(self, *args, **kwargs):
        return self.clf.predict_log_proba(*args, **kwargs)

    @staticmethod
    def load(classifier_path):
        return joblib.load(classifier_path)


class VCFApply(_VCFExtract):
    """Apply the pre-trained classifier on a VCF file.

    :params str filepath: filepath to unfiltered VCF file.
    :params Classifier classifier: pre-trained classifier.
    :params str mode: kind of variants training on: 'SNP' or 'INDEL'.
    """

    def __init__(self, filepath, classifier: Classifier_FS, mode):
        super().__init__(filepath)
        self.classifier = classifier
        self.vartype = mode.upper
        self.data, _, self.vartype_index = self.fetch_data("BOTH", self.classifier.features)  # temp
        self.mend_nan(self.data)
        self.predict_y = None
        self.logger = logging.getLogger(self.__class__.__name__)

        # check features
        this_feature = set(self.features)
        clf_feature = set(self.classifier.features)
        if this_feature != clf_feature:
            if len(clf_feature - this_feature) == 0:
                pass
            self.logger.warning(
                "Features not match! Missing features: {}, excessive features: {}".format(this_feature - clf_feature,
                                                                                          clf_feature - this_feature))

    def apply(self):
        self.predict_y = self.classifier.predict(self.data)
        if self.classifier.kind in ["SVM", "SUPPORTVECTOR", "LGBM", "LightGBM"]:
            probabilities = self.classifier.predict_proba(self.data)
            self.predict_y_log_proba = np.log(probabilities)
        # self.predict_y_log_proba = self.classifier.predict_log_proba(self.data)

    def _is_gzip(self, file):
        with open(file, 'rb') as f:
            return binascii.hexlify(f.read(2)) == b'1f8b'

    def write_filtered(self, output_filepath):
        """
        Write filtered VCF file, SNPs only and change the FILTER field to PASS or VEF_FILTERED

        :params output_filepath: output filepath.
        """

        if np.sum(self.predict_y) == 0:
            self.logger.error("No passed variants.")
            return
        chunk_size = 10000
        is_gzip = self._is_gzip(self.filepath)
        self.logger.info("Start output filtered result to file {}".format(os.path.abspath(output_filepath)))
        t0 = time.time()
        chunk = []
        if is_gzip:
            infile = gzip.open(self.filepath, 'rt')
        else:
            infile = open(self.filepath, 'r')
        with infile, open(output_filepath, 'w') as outfile:
            # iterate headers
            if self.vartype == 'SNP':
                is_vartype = lambda x, y: len(x) == 1 and len(y) == 1
            else:
                is_vartype = lambda x, y: not (len(x) == 1 and len(y) == 1)

            filter_written = False
            for line in infile:
                if line.startswith("##"):
                    if 'FILTER' in line and not filter_written:
                        chunk.append('##FILTER=<ID=VEF_FILTERED,Description="VEF filtered">\n')
                        chunk.append(
                            '##INFO=<ID=VEF,Number=1,Type=Float,Description="Log Probability of being true variants according to VEF">\n')
                        filter_written = True
                    chunk.append(line)
                elif line.startswith("#"):
                    fields = line[1:].split()
                    chunk.append(line)
                    outfile.write(''.join(chunk))
                    chunk = []
                    break
            idx_FILTER = fields.index("FILTER")
            idx_REF = fields.index("REF")
            idx_ALT = fields.index("ALT")
            idx_INFO = fields.index("INFO")
            vcf_reader = csv.reader(infile, delimiter='\t')
            for num_row, row in enumerate(vcf_reader):
                if self.predict_y[num_row]:
                    row[idx_FILTER] = "PASS"
                    row[idx_INFO] += (";VEF={:.4e}".format(self.predict_y_log_proba[num_row, 1]))
                else:
                    row[idx_FILTER] = "VEF_FILTERED"
                    row[idx_INFO] += (";VEF={:.4e}".format(self.predict_y_log_proba[num_row, 1]))
                chunk.append(row)
                if num_row % chunk_size == 0 and num_row > 0:
                    outfile.write('\n'.join(['\t'.join(var) for var in chunk]) + '\n')
                    chunk = []
            outfile.write('\n'.join(['\t'.join(var) for var in chunk]))
        t1 = time.time()
        self.logger.info("Finish output filtered result, time elapsed {:.3f}s".format(t1 - t0))
        self.logger.info("Finish output filtered result, time elapsed {:.3f}s".format(t1 - t0))
