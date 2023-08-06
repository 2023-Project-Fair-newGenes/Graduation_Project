import argparse
from performance import *

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='''\
EDA
-------------------------
Example of use

python evaluate.py --happy path/to/NA12878.vcf.happy.vcf --target path/to/NA12878.vcf --mode SNP
            ''')
    requiredNamed = parser.add_argument_group("required named arguments")
    requiredNamed.add_argument("--happy", help="hap.py annoted target VCF file", required=True)
    requiredNamed.add_argument("--target", help="target pipeline VCF file", required=True)
    requiredNamed.add_argument("--mode", help="mode, SNP or INDEL", required=True,
            choices=["SNP", "INDEL"])

    args = parser.parse_args()
    vcf_hap = args.happy
    vcf_tgt = args.target
    mode = args.mode

    data = VCFData(vcf_hap, vcf_tgt, mode)

if __name__ == '__main__':
    main()
