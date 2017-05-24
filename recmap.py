"""
recmap.py

This file contains functions for reading a genetic map (or recombination map)
into an interval tree so genetic distance can be estimated for arbitrary
physical distances (in base pairs).

"""

import collections

import intervaltree


class RecMap(object):
    """
    Class for representing a genetic map. Genetic distances (in centimorgans)
    are stored according to their associated intervals in an interval tree
    data structure. This tree is used to estimate genetic distances for
    physical intervals (in base pairs).

    Attributes:
        scaf_ints: A dictionary storing the interval tree for each chromosome.
    """
    def __init__(self):
        """
        Initialize a new RecMap
        """
        self.scaf_ints = collections.defaultdict(intervaltree.IntervalTree)

    def _read_tab(self, rec_in):
        """
        Read genetic distances in from a tab delimited file containing
        markers (physical positions) and genetic distances (in centimorgans)
        and populate the interval tree for each chromosome.

        Args:
            rec_in: a file object for reading the recombination map.
        Returns: nothing
        """
        last_chrm = None
        last_pos = None
        last_dist = None
        for line in rec_in:
            items = line.split('\n')
            chrm = items[0]
            pos = int(items[1])
            dist = float(items[2])

            if chrm == last_chrm:
                self.scaf_ints[chrm][last_pos, pos] = dist - last_dist

            last_chrm = chrm
            last_pos = pos
            last_dist = dist
        return

    def gene_dist(self, chrm, start, end):
        """
        Return an estimate of the genetic distance between physical positions
        'start' and 'end' on chromosome 'chrm'.

        Args:
            chrm: A chromosome ID
            start: The start position of the interval
            end: The end position of the interval (end > start)
        returns: a genetic distance in centimorgans
        """
        # super simple version of this.
        intervals = self.scaf_ints[chrm][start:end]
        return sum([interval.data for interval in intervals])
