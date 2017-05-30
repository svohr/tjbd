#! /usr/bin/env python2
"""
This script simulates low-coverage obsevations of a historic individual
including segments of IBD shared with a high-coverage present day individual
and uses the forward backward algorithm implemented in ibd_hmm to detect
the simulated IBD segments.

Sam Vohr (svohr@soe.ucsc.edu)
Mon May 29 15:37:59 PDT 2017
"""

import sys
import random
import argparse
import itertools

import numpy
import pysam
import intervaltree


def pick_ibd_segments():
    ibd = intervaltree.IntervalTree()
    ibd[10000000:30000000] = 1
    return ibd


def match_found(hi_geno, lo_obs):
    """
    Returns True if the low-coverage observation matches one of the alleles
    in the high-coverage genotype, False otherwise.
    """
    if lo_obs != hi_geno[0]:
        return lo_obs == hi_geno[1]
    return True


def simulate_hmm_input(vcf_in, ibd_segs, lo_indv, hi_indv, coverage):
    """
    Simulates low-coverage sequencing from a recent historical ancestor (low_i)
    of a present-day individul (hi_indv) by sampling SNP entries from a VCF
    file. IBD segments are simulated by taking observations from the
    high-coverage individual. Returns a vector of chromosome positions and
    a vector of booleans indicate whether the historic individual shared one
    of the present-day individual's alleles.

    Args:
        vcf_in: VCF file input for the chromosome of interest.
        ibd_segs: Chromosome regions where IBD should be simulated.
        lo_indv: ID of the low-coverage, historic individual.
        hi_indv: ID of the high-coverage, present day individual.
        coverage: depth of sequencing in the historic individual.
    Returns:
        positions: A vector of chromosome positions where an observation
                   was made.
        match: A vector of booleans indicating whether the observe historic
               base matched one of the present-day bases.
    """
    obs_pos = list()
    obs_state = list()

    for rec in vcf_in:
        if numpy.random.poisson(coverage) != 1:
            continue # position was not observed or observed too many times.
        _ = rec.alts # this avoids a segfault ¯\_(ツ)_/¯
        hi_samp = rec.samples[hi_indv]
        lo_samp = rec.samples[lo_indv]
        hi_geno = hi_samp['GT']
        lo_geno = random.choice(lo_samp['GT'])
        if ibd_segs.overlaps(rec.pos):
            if random.choice(True, False):
                # observing from IBD segment.
                lo_geno = hi_geno[0]
            # else: observing from other chromosome.
        obs_pos.append(rec.pos)
        obs_state.append(match_found(hi_geno, lo_geno))
    return numpy.array(obs_pos), numpy.array(obs_state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("vcf_fn", metavar="vars.vcf[.gz]", type=str,
                        help="VCF file containing genotypes for individuals")
    args = parser.parse_args()

    ibd_segs = pick_ibd_segments()
    with pysam.VariantFile(args.vcf_fn, 'r') as vcf_in:
        pos, obs = simulate_hmm_input(vcf_in, ibd_segs, "HG00097", "HG00099", 0.01)
        for p, o in itertools.izip(pos, obs):
            print p, "match" if o else "no-match", "IBD" if ibd_segs.overlaps(p) else "no-IBD"
    return 0


if __name__ == "__main__":
    sys.exit(main())
