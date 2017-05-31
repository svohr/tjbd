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
    def __init__(self, rec_in=None):
        """
        Initialize a new RecMap
        Args:
            rec_in: file object for reading in the recombination map.
        Returns: nothing
        """
        self.scaf_ints = collections.defaultdict(intervaltree.IntervalTree)
        if rec_in:
            self.read_tab(rec_in)

    def read_tab(self, rec_in):
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
            items = line.split('\t')
            chrm = items[0]
            pos = int(items[1])
            dist = float(items[2])

            if chrm == last_chrm and pos != last_pos:
                self.scaf_ints[chrm][last_pos:pos] = dist - last_dist

            last_chrm = chrm
            last_pos = pos
            last_dist = dist
        return

    def distance(self, chrm, start, end):
        """
        Return an estimate of the genetic distance between physical positions
        'start' and 'end' on chromosome 'chrm'. If a map distance partially
        overlaps the range, the genetic distance is scaled assuming a
        uniform recombination rate between the available physical positions.

        Args:
            chrm: A chromosome ID
            start: The start position of the interval
            end: The end position of the interval (end > start)
        returns: a genetic distance in centimorgans
        """
        if not self.scaf_ints[chrm].overlaps(start, end):
            return None # query interval not included in the map.
        interval = intervaltree.Interval(start, end)
        total_dist = 0.0
        for map_interval in self.scaf_ints[chrm][start:end]:
            if interval.contains_interval(map_interval):
                total_dist += map_interval.data
            else:
                frac = (float(min(interval.end, map_interval.end)
                              - max(interval.begin, map_interval.begin))
                        / map_interval.length())
                total_dist += frac * map_interval.data
        return total_dist
