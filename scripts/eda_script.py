from src import VCFData
def main():

    vcf_hap = "../dataset/NA12878_chr11.indel.vcf.gz.happy.vcf.gz"
    vcf_tgt = "../dataset/NA12878_chr11.indel.vcf.gz"
    mode = "INDEL"

    data = VCFData(vcf_hap, vcf_tgt, mode)

if __name__ == '__main__':
    main()