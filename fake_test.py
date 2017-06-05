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

import freqs
import recmap
import ibd_hmm


def pick_ibd_segments():
    ibd = intervaltree.IntervalTree()
    ibd[19000000:25000000] = 1
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
    obs_hi_geno = list()
    obs_lo_geno = list()

    for rec in vcf_in:
        if numpy.random.poisson(coverage) != 1:
            continue # position was not observed or observed too many times.
        _ = rec.alts # this avoids a segfault
        hi_samp = rec.samples[hi_indv]
        lo_samp = rec.samples[lo_indv]
        hi_geno = hi_samp['GT']
        lo_geno = random.choice(lo_samp['GT'])
        if ibd_segs.overlaps(rec.pos):
            if random.choice([True, False]):
                # observing from IBD segment.
                lo_geno = hi_geno[0]
            # else: observing from other chromosome.
        obs_pos.append(rec.pos)
        obs_hi_geno.append(hi_geno)
        obs_lo_geno.append(lo_geno)
    return numpy.array(obs_pos), obs_hi_geno, obs_lo_geno


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("vcf_fn", metavar="vars.vcf[.gz]", type=str,
                        help="VCF file containing genotypes for individuals")
    parser.add_argument("frq_fn", metavar="frqs.count", type=str,
                        help="Allele counts from VCFTools")
    parser.add_argument("rec_fn", metavar="recmap.tab", type=str,
                        help="Genetic distances in tab file.")
    args = parser.parse_args()

    ibd_segs = pick_ibd_segments()

    frqs = freqs.AlleleFreqs()
    with open(args.frq_fn, 'r') as frq_in:
        frqs.read_vcf_counts(frq_in)

    rmap = recmap.RecMap()
    with open(args.rec_fn, 'r') as rec_in:
        rmap.read_tab(rec_in)

    with pysam.VariantFile(args.vcf_fn, 'r') as vcf_in:
        pos, obs_hi, obs_lo = simulate_hmm_input(vcf_in, ibd_segs, "HG00097", "HG00099", 0.05)
        obs = [match_found(h, l) for h, l in itertools.izip(obs_hi, obs_lo)]
        lo_freq = numpy.array([frqs.frequency("21", p, b) for p, b in itertools.izip(pos, obs_lo)])

        ibd_trs, noibd_trs = ibd_hmm.state_trans(rmap, 5, "21", pos)
        probs = ibd_hmm.forward_backward(5, obs, lo_freq, ibd_trs, noibd_trs)
        lprobs = ibd_hmm.forward_backward_log_prob(5, obs, lo_freq, ibd_trs, noibd_trs)
        for p, o, f, i, n, prob, l in itertools.izip(pos, obs, lo_freq, ibd_trs, noibd_trs, probs, lprobs):
            print p, "match" if o else "no-match", "IBD" if ibd_segs.overlaps(p) else "no-IBD", f, i, n, prob, l

    return 0


if __name__ == "__main__":
    sys.exit(main())
