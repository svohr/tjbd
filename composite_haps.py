#! /usr/bin/env python2
"""
This script takes in a VCF file containing the genotypes of N individuals
individuals and generates N composite _haplotypes_ (N/2 diploid individuals) to
break up latent IBD using the method described by Browning and Browning 2010.
"""

import sys
import argparse
import random
import pysam

import recmap


def init_composites(nsamps, segsize, start_gpos):
    """
    Generate the random offsets for each composite individual.

    Args:
        nsamps: Number of starting individuals.
        segsize: Size of segments to use from each individual.
        start_gpos: The first genetic position we are using.
    Returns:
        compo_src: starting individual for each composite.
        compo_off: starting offset for each composite.
    """
    if nsamps % 2 == 1:
        nsamps -= 1
    compo_src = range(0, nsamps, 2)
    compo_off = [random.uniform(0.001, segsize - 0.001) + start_gpos
                 for _ in xrange(nsamps / 2)]
    return compo_src, compo_off


def update_composites(compo_src, compo_off, segsize, gpos):
    """
    Updates the current lists of source individuals and offsets for each
    composite individual.

    Args:
        compo_src: list of current source individual for each composite.
        compo_off: list of next switch position for each composite.
        segsize: max size of segment for a single source individual.
        gpos: current genetic position.
    Returns:
        compo_src: updated list of source individuals
        compo_off: updated list of next switch positions
    """
    for i, _ in enumerate(compo_src):
        while compo_off[i] < gpos:
            compo_off[i] += segsize
            compo_src[i] = (compo_src[i] + 1) % (len(compo_src) * 2)
    return compo_src, compo_off


def write_composites(out, vcf_rec, gpos, compo_src):
    """
    Writes a tab-delimited table entry for this position including the
    genotypes for each composite individual. Fields are 1) chromosome ID,
    2) position, 3) genetic position, 4-) base for each composite haplotype.

    e.g.,
    chr1    10000   0.91345 A   T   A   A   T   A

    Args:
        out: destination to write entry.
        vcf_rec: A VCF VariantRecord,
        compo_src: The current source individual for each composite individual.
    Returns:
        nothing.
    """
    out.write("%s\t%d\t%f" % (vcf_rec.chrom, vcf_rec.pos, gpos))
    for _, src in enumerate(compo_src):
        out.write("\t%s\t%s" % tuple([vcf_rec.alleles[gt] for gt in vcf_rec.samples[src]["GT"]]))
    out.write("\n")
    return


def main():
    """
    do thing.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("vcf_fn", metavar="vars.vcf[.gz]", type=str,
                        help="VCF file containing genotypes for individuals")
    parser.add_argument("rec_fn", metavar="recmap.tab", type=str,
                        help="Genetic distances in tab file.")
    parser.add_argument("-s", "--size", dest="segsize", metavar="cM",
                        type=float, default=0.2,
                        help="Size of segments to use (in centiMorgans)")
    args = parser.parse_args()

    rmap = recmap.RecMap()
    with open(args.rec_fn, 'r') as rec_in:
        rmap.read_tab(rec_in)

    with pysam.VariantFile(args.vcf_fn, 'r') as vcf_in:
        cmp_src = None
        for rec in vcf_in.fetch():
            _ = rec.alts # this avoids a segfault
            gpos = rmap.position(rec.chrom, rec.pos)
            if cmp_src is None:
                cmp_src, cmp_off = init_composites(len(vcf_in.header.samples),
                                                   args.segsize, gpos)
            else:
                cmp_src, cmp_off = update_composites(cmp_src, cmp_off,
                                                     args.segsize, gpos)
            write_composites(sys.stdout, rec, gpos, cmp_src)

    return 0


if __name__ == "__main__":
    sys.exit(main())
