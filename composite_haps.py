#! /usr/bin/env python2
"""
This script takes in a VCF file containing the genotypes of N individuals
individuals and generates N composite _haplotypes_ (N/2 diploid individuals) to
break up latent IBD using the method described by Browning and Browning 2010.
"""

import sys
import argparse
import pysam

import recmap


def main():
    """
    do thing.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("vcf_fn", metavar="vars.vcf[.gz]", type=str,
                        help="VCF file containing genotypes for individuals")
    parser.add_argument("rec_fn", metavar="recmap.tab", type=str,
                        help="Genetic distances in tab file.")
    args = parser.parse_args()

#   rmap = recmap.RecMap()
#   with open(args.rec_fn, 'r') as rec_in:
#       rmap.read_tab(rec_in)

    with pysam.VariantFile(args.vcf_fn, 'r') as vcf_in:
        for rec in vcf_in.fetch():
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())

