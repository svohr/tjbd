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

import numpy
import pysam


def match_found(hi_geno, lo_obs):
    """
    Returns True if the low-coverage observation matches one of the alleles
    in the high-coverage genotype, False otherwise.
    """
    if lo_obs != hi_geno[0]:
        return lo_obs == hi_geno[1]
    return True


def simulate_hmm_input(vcf_in, ibd_segs, lo_indv, hi_indv, coverage, err)
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
        err: rate at which erroneous bases occur in historic observations.
    Returns:
        positions: A vector of chromosome positions where an observation
                   was made
        match: A vector of booleans indicating whether the observe historic
               base matched one of the present-day bases.
    """
    obs_pos = list()
    obs_state = list()

    for rec in vcf_in:
        if numpy.random.poisson(coverage) != 1:
            continue # position was not observed or observed too many times.
        obs_pos.append(rec.pos)
        hi_geno = rec.samples[hi_indv]['GT']
        lo_geno = random.choice(rec.samples[lo_indv]['GT'])
        if ibd_segs.overlaps(rec.pos):
            if random.choice(True, False):
                # observing from IBD segment.
                lo_geno = hi_geno[0]
            # else: observing from other chromosome.
        obs_state.append(match_found(hi_geno, lo_geno))
    return numpy.array(obs_pos), numpy.array(obs_state)


def main():
    return 0


if __name__ == "__main__":
    sys.exit(main())
