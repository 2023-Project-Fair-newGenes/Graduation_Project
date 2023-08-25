import allel
import time
import logging
import csv
import os
import binascii
import gzip
import numpy as np
import joblib

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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
            fields = [(VAR_PREFIX + k) for k in list(self.header.infos.keys()) + ['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL']]
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



class VCFData:
    """Compare VCF files and prepare features/labels for training.

    :param str hap_filepath: filepath to happy output VCF.
    :param str specimen_filepath: filepath to specimen VCF.
    :param str mode: kind of variants training on: 'SNP' or 'INDEL'.
    """

    def __init__(self, hap_filepath, specimen_filepath, mode):
        self.hap_vcf = _VCFExtract(hap_filepath)
        self.specimen_vcf = _VCFExtract(specimen_filepath)
        self.dataset = {}
        self.contigs = []
        self.features = []
        self.logger = logging.getLogger(self.__class__.__name__)
        self._EDA(mode)

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

    def _EDA(self,mode):
        self.logger.info("Start extracting label from {}".format(os.path.abspath(self.hap_vcf.filepath)))
        VAR_PREFIX = 'variants/'
        mode_list = ['SNP', 'INDEL']

        # hap VCF 파일에서 필요한 필드를 읽어옴, data에 저장
        data = allel.read_vcf(self.hap_vcf.filepath,
                              fields=['calldata/BVT', 'variants/Regions', 'variants/POS', 'variants/CHROM'])

        # data['variants/Regions']에서 읽어온 데이터를 부울(Boolean) 형태로 변환하여 conf_mask 변수에 할당
        # 변이 호출 결과에서 고신뢰도 영역에 해당하는 변이들을 표시
        # True : 고신뢰도 영역에 해당하는 변이를 나타냄 // False : 고신뢰도 영역에 해당하지 않는 변이를 나타냄
        conf_mask = data['variants/Regions'].astype(bool)

        self.logger.info(
            "Total variants(hap.py): {}, in high-conf region variants: {}".format(conf_mask.shape[0], int(sum(conf_mask))))

        if mode.upper() in mode_list:
            extract_target_vartype = self._extract_factory(np.where(self.hap_vcf.samples == 'TRUTH')[0][0], mode.upper())
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
        # self.features = [ftr for ftr in self.features if np.issubdtype(data[ftr].dtype, np.number)]
        # self.features = [ftr for ftr in self.features if np.sum(np.isnan(data[ftr])) < 0.01 * num_var]
        # self.features = [ftr for ftr in self.features if np.nanvar(data[ftr]) >= 0.1]

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

        for ch in self.contigs:
            if ch not in label_list:
                continue
            else:
                annotes_idx = np.where(np.isin(annotes_chrom[ch][0], label_list[ch][0]))[0]
                label_idx = np.where(np.isin(label_list[ch][0], annotes_chrom[ch][0]))[0]
                self.dataset[ch] = np.vstack(
                    (annotes_chrom[ch][1:, annotes_idx], label_list[ch][1, label_idx])).transpose()

                data_array = np.vstack((annotes_chrom[ch][1:, annotes_idx], label_list[ch][1, label_idx])).transpose()

                # 데이터프레임으로 변환
                columns = self.features + ['Label']
                df = pd.DataFrame(data_array, columns=columns)

                pd.set_option('display.max_columns', None)
                print(df.head())


                # TODO : 여기서 시각화 코드 찍어보기

                #AN
                AN_counts = df['variants/AN'].value_counts()
                print("AN_counts : ",AN_counts)


                # Count 플롯 시각화
                plt.clf()
                sns.set(style="darkgrid")
                sns.countplot(x=df['variants/AN'], data=df)
                plt.title('Count Plot of variants/AN')
                plt.show()



                # MLEAF
                MLEAF_counts = df['variants/MLEAF'].value_counts()
                print("MLEAF_counts : ", MLEAF_counts)

                # count 플롯 시각화
                plt.clf()
                sns.set(style="darkgrid")
                sns.countplot(x=df['variants/MLEAF'], data=df)
                plt.title('Count Plot of variants/MLEAF')
                plt.show()


                # HaplotypeScore
                HaplotypeScore_counts = df['variants/HaplotypeScore'].value_counts()
                print("HaplotypeScore_counts : ", HaplotypeScore_counts)


                # InbreedingCoeff

                InbreedingCoeff_counts = df['variants/InbreedingCoeff'].value_counts()
                print("InbreedingCoeff_counts : ", InbreedingCoeff_counts)


                # MQRankSum

                MQRankSum_counts = df['variants/MQRankSum'].value_counts()
                print("MQRankSum_counts : ", MQRankSum_counts)

                # 분산도 시각화
                plt.clf()
                plt.scatter(df['variants/MQRankSum'], df['Label'])
                plt.title('Scatter Plot of variants/MQRankSum')
                plt.show()




                # ReadPosRankSum

                ReadPosRankSum_counts = df['variants/ReadPosRankSum'].value_counts()
                print("ReadPosRankSum_counts : ", ReadPosRankSum_counts)

                # 분산도 시각화
                plt.clf()
                plt.scatter(df['variants/ReadPosRankSum'], df['Label'])
                plt.title('Scatter Plot of variants/ReadPosRankSum')
                plt.show()


                # DS

                DS_counts = df['variants/DS'].value_counts()
                print("DS_counts : ", DS_counts)

                plt.clf()
                sns.set(style="darkgrid")
                sns.countplot(x=df['variants/DS'], data=df)
                plt.title('Count Plot of variants/DS')
                plt.show()


                # ExcessHet

                ExcessHet_counts = df['variants/ExcessHet'].value_counts()
                print("ExcessHet_counts : ", ExcessHet_counts)

                # 분산도 시각화
                plt.clf()
                sns.set(style="darkgrid")
                sns.countplot(x=df['variants/ExcessHet'], data=df)
                plt.title('Count Plot of variants/ExcessHet')
                plt.show()



                # FS

                FS_counts = df['variants/FS'].value_counts()
                print("FS_counts : ", FS_counts)


                # 분산도 시각화

                plt.clf()
                plt.scatter(df['variants/FS'], df['Label'])
                plt.title('Scatter Plot of variants/FS')
                plt.show()

        return self