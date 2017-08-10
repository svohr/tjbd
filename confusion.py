"""
"""

import itertools

import numpy


class ConfusionTable(object):
    def __init__(self):
        """
        """
        self.relatedness = numpy.zeros((2, 2), dtype=numpy.uint32)
        self.positional = numpy.zeros((2, 2), dtype=numpy.uint32)
        return

    def update(self, ibd_segs, called_segs):
        """
        Takes a vector containing the True IBD state for each position
        and a vector of IBD state calls (0 for no IBD, 1 for IBD) and
        updates the confusion matrices for both relatedness and by position.

        Args:
            ibd_segs: the true state of each position.
            called_segs: the called state of each position.
        returns:
            Nothing.
        """
        self.relatedness[1 if ibd_segs.any() else 0,
                         1 if called_segs.any() else 0] += 1
        for ibd, call in itertools.izip(ibd_segs, called_segs):
            self.positional[1 if ibd else 0, 1 if call else 0] += 1
        return

    def rel_tn(self):
        return self.relatedness[0, 0]
    def rel_fp(self):
        return self.relatedness[0, 1]
    def rel_fn(self):
        return self.relatedness[1, 0]
    def rel_tp(self):
        return self.relatedness[1, 1]

    def pos_tn(self):
        return self.positional[0, 0]
    def pos_fp(self):
        return self.positional[0, 1]
    def pos_fn(self):
        return self.relatedness[1, 0]
    def pos_tp(self):
        return self.relatedness[1, 1]

    def rel_sensitivity(self):
        return numpy.float(self.relatedness[1, 1]) / sum(self.relatedness[1, ])
    def rel_fpr(self):
        return numpy.float(self.relatedness[0, 1]) / sum(self.relatedness[:, 1])

    def pos_sensitivity(self):
        return numpy.float(self.positional[1, 1]) / sum(self.positional[1, ])
    def pos_fpr(self):
        return numpy.float(self.positional[0, 1]) / sum(self.positional[:, 1])
