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
    """
    Chooses intervals of a chromosome to be simulated as Identitical by
    descent.

    Args: None
    Returns: An interval tree containing IBD segments.
    """
    ibd = intervaltree.IntervalTree()
    ibd[19000000:25000000] = 1
    return ibd


def match_found(hi_geno, lo_obs):
    """
    Returns True if the low-coverage observation matches one of the alleles
    in the high-coverage genotype, False otherwise.
    """
    return sum(h == lo_obs for h in hi_geno)


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
        lo_chrm = random.choice([0, 1])
        lo_geno = lo_samp['GT'][lo_chrm]
        if ibd_segs.overlaps(rec.pos) and lo_chrm == 0:
            # observing from IBD segment.
            lo_geno = hi_geno[0]
            # else: observing from other chromosome.
        obs_pos.append(rec.pos)
        obs_hi_geno.append(hi_geno)
        obs_lo_geno.append(lo_geno)
    return numpy.array(obs_pos), obs_hi_geno, obs_lo_geno


def find_matches(obs_hi, obs_lo):
    """
    Compares the high-coverage genotypes with the low-coverage observations
    and produces a list of bools indicating whether a match was found.

    Args:
        obs_hi: List of high-coverage genotypes stored as 2-ples of strings.
        obs_lo: List of low-coverage observations (single base (string))
    Returns:
        A list of booleans indicating wheather a match was found or not.
    """
    return [match_found(h, l) for h, l in itertools.izip(obs_hi, obs_lo)]


def get_frequencies(frqs, chrm, pos, obs):
    """
    Finds the allele frequencies for the bases observed at the given positions.

    Args:
        chrm: a chromosome ID
        pos: a list of chromosome positions (int)
        obs: a list of single base observations (string)
    Returns:
        a numpy vector of allele frequencies
    """
    return numpy.array([frqs.frequency(chrm, p, b)
                        for p, b in itertools.izip(pos, obs)])


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
    ngen = 5
    chrom = "21"

    frqs = freqs.AlleleFreqs()
    with open(args.frq_fn, 'r') as frq_in:
        frqs.read_vcf_counts(frq_in)

    rmap = recmap.RecMap()
    with open(args.rec_fn, 'r') as rec_in:
        rmap.read_tab(rec_in)


    for i in xrange(10):
        with pysam.VariantFile(args.vcf_fn, 'r') as vcf_in:
            hi_indv, lo_indv = random.sample(vcf_in.header.samples, 2)
            pos, obs_hi, obs_lo = simulate_hmm_input(vcf_in,
                                                     ibd_segs,
                                                     lo_indv,
                                                     hi_indv,
                                                     0.05)
            obs = find_matches(obs_hi, obs_lo)
            lo_freq = get_frequencies(frqs, chrom, pos, obs_lo)
            ibd_trs, noibd_trs = ibd_hmm.state_trans(rmap, ngen, chrom, pos)

            probs = ibd_hmm.forward_backward(ngen, obs, lo_freq, ibd_trs, noibd_trs)
            lprobs = ibd_hmm.forward_backward_log_prob(ngen, obs, lo_freq, ibd_trs, noibd_trs)
            for p, o, f, i, n, prob, l in itertools.izip(pos, obs, lo_freq, ibd_trs, noibd_trs, probs, lprobs):
                print "%s_%s" % (hi_indv, lo_indv), chrom, p, ngen, o, "IBD" if ibd_segs.overlaps(p) else "no-IBD", f, i, n, prob, l
    return 0


if __name__ == "__main__":
    sys.exit(main())
