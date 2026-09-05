"""
freqs.py

This file contains functions for reading allele count output from VCF tools.
"""

import sys


class AlleleFreqs(object):
    """
    Class for storing and retrieving allele frequencies from VCFTools output
    using the --count option.

    Attributes:
        freqs: A dictionary that associates a chromosome, position and allele
               with the frequency of that allele through the _key method.
        total: The total number of chromosomes sampled when estimating allele
               frequency (assumed to be the same across all sites.)
    """
    def __init__(self, count_in=None):
        """
        Initialize a new AlleleFreqs table.

        Args:
            count_in: a file object from which a table of allele counts is read.
        Returns: nothing
        """
        self.freqs = dict()
        self.total = None
        if count_in is not None:
            self.read_vcf_counts(count_in)

    def read_vcf_counts(self, count_in):
        """
        Reads in an allele counts table (tab-delimited) produced using
        vcftools --counts and stores the frequency of each allele it describes.
        Also stores the total number of chromosomes surveyed.

        Args:
            count_in: a file object from which a table of allele counts is read.
        Returns: nothing
        """
        count_in.readline() # remove the header line
        for line in count_in:
            items = line.split('\t')
            chrm = items[0]
            pos = int(items[1])
            total = int(items[3])
            for count_pair in items[4:]:
                base, _, count = count_pair.rpartition(':')
                self.freqs[_key(chrm, pos, base)] = float(count) / total
            if self.total is None:
                self.total = total

    def frequency(self, chrm, pos, base):
        """
        Retrieves an allele frequency from the stored table. If a variant is
        not found in the table, a pseudocount frequency is returned based on
        the number of chromosomes surveyed.

        Args:
            chrm: A chromosome id (string)
            pos: A base pair position (int)
            base: A base (string)
        Returns:
            A frequency within (0, 1) (float)
        """
        key = _key(chrm, pos, base)
        if key in self.freqs:
            if self.freqs[key] == 1.0:
                # if this site is fixed in the population,
                # return 1 minus a pseudocount.
                return 1 - (0.5 / self.total)
            return self.freqs[key]
        else:
            print("Not Found! {} {} {}".format(chrm, pos, base),
                  file=sys.stderr)
            # if base is not found in population, return a pseudocount freq.
            return 0.5 / self.total

    def isin(self, chrm, pos, base):
        """
        Returns True if there is an observed frequency for this
        chromosome/position/base combination.
        """
        return _key(chrm, pos, base) in self.freqs

    def min_af(self, chrm, pos):
        """
        Returns the minimum allele frequency at the specified position.
        """
        keys = [_key(chrm, pos, base) for base in 'ACGT']
        afs = [self.freqs[key] for key in keys if key in self.freqs]
        if not afs:
            # No entries found, report the pseudocount
            return 0.5 / self.total
        return min(afs)


def _key(chrm, pos, base):
    """
    Returns a key string based a chromosome id, position, and base char.

    Args:
        chrm: A chromosome id (string)
        pos: A base pair position (int)
        base: A base (string)
    Returns:
        A string to be used as a key in self.freqs.
    """
    return "%s:%d:%s" % (chrm, pos, base)
