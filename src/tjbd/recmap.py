"""
recmap.py

This file contains functions for reading a genetic map (or recombination map)
into two sorted lists so genetic distance can be estimated for arbitrary
physical distances (in base pairs).

"""

import collections
import bisect


def _bg_rec_rate(start, end):
    """
    Returns a genetic distance based on an average recombination rate of
    1 centimorgan per megabase.

    Args:
        start: Start position of interval.
        end: End position of interval.
    Returns:
        a genetic distance in centimorgans between the start and end.
    """
    return float(end - start) / 1000000.0


class RecMap(object):
    """
    Class for representing a genetic map. Genetic distances (in centimorgans)
    are stored according to their associated intervals in an interval tree
    data structure. This tree is used to estimate genetic distances for
    physical intervals (in base pairs).

    Attributes:
        scaf_ppos: A dictionary storing a sorted list of physical positions
        scaf_gpos: A dictionary storing a sorted list of the corresponding
                   genetic positions.
    """
    def __init__(self, rec_in=None):
        """
        Initialize a new RecMap
        Args:
            rec_in: file object for reading in the recombination map.
        Returns: nothing
        """
        self.scaf_ppos = collections.defaultdict(list)
        self.scaf_gpos = collections.defaultdict(list)
        if rec_in:
            self.read_tab(rec_in)

    def read_tab(self, rec_in):
        """
        Read physical and genetic positions in from a plink formatted map file
        containing markers (chromosome IDs and physical positions) and genetic
        distances (in centimorgans) and populates the internal lists for each
        chromosome. Input is assumed to be sorted by physical position.

        Args: rec_in: a file object for reading the recombination map.
        Returns: nothing
        """
        for line in rec_in:
            items = line.split(' ')
            chrm = items[0]
            pos = int(items[3])
            gpos = float(items[2])

            self.scaf_ppos[chrm].append(pos)
            self.scaf_gpos[chrm].append(gpos)
        return

    def _index_gen_position(self, chrm, pos, idx):
        """
        Returns an estimated genetic position on the chromosome given the
        bisect insertion point in the physical map. If the marker is included
        in the recombination map, the genetic position in centimorgans from the
        start of the map is used. If the physical position is not found in the
        recombination made, a genetic position found by interpolation between
        the two closest markers is returned. Helper function for position()
        and distance()

        Args:
            chrm: A chromosome ID
            pos: The physical position in base pairs on the chromosome.
        Returns: A genetic positions in centimorgans from the initial marker in
                 the map.
        """
        if idx == len(self.scaf_ppos[chrm]):
            return (self.scaf_gpos[chrm][idx - 1]
                    + _bg_rec_rate(self.scaf_ppos[chrm][idx - 1], pos))
        elif self.scaf_ppos[chrm][idx] == pos:
            return self.scaf_gpos[chrm][idx]
        elif idx == 0:
            return (self.scaf_gpos[chrm][0]
                    - _bg_rec_rate(pos, self.scaf_ppos[chrm][idx]))
        frac = (float(pos - self.scaf_ppos[chrm][idx - 1])
                / (self.scaf_ppos[chrm][idx] - self.scaf_ppos[chrm][idx - 1]))
        gdist = frac * (self.scaf_gpos[chrm][idx]
                        - self.scaf_gpos[chrm][idx - 1])
        return self.scaf_gpos[chrm][idx - 1] + gdist


    def position(self, chrm, pos):
        """
        Returns an estimated genetic position on the chromosome. If the
        marker is included in the recombination map, the genetic position in
        centimorgans from the start of the map is used. If the physical position
        is not found in the recombination made, a genetic position found by
        interpolation between the two closest markers is returned.

        Args:
            chrm: A chromosome ID
            pos: The physical position in base pairs on the chromosome.
        Returns: A genetic positions in centimorgans from the initial marker in
                 the map.
        """
        idx = bisect.bisect_left(self.scaf_ppos[chrm], pos)
        return self._index_gen_position(chrm, pos, idx)

    def distance(self, chrm, start, end):
        """
        Return an estimate of the genetic distance between physical positions
        'start' and 'end' on chromosome 'chrm'. If the start or end markers
        do not appear in the map, the genetic distance will be interpolated
        based on the surrounding markers, assuming a uniform recombination
        rate between markers.

        Args:
            chrm: A chromosome ID
            start: The start position of the interval
            end: The end position of the interval (end > start)
        returns: a genetic distance in centimorgans
        """
        start_idx = bisect.bisect_left(self.scaf_ppos[chrm], start)
        end_idx = bisect.bisect_left(self.scaf_ppos[chrm], end, lo=start_idx)
        start_gpos = self._index_gen_position(chrm, start, start_idx)
        end_gpos = self._index_gen_position(chrm, end, end_idx)
        return end_gpos - start_gpos

