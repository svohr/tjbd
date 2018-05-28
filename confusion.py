"""
"""

import itertools
import numpy
import pandas


N_HIST_BINS = 1000
HIST_RANGE = (0.0, 1.0)


def get_segment_intervals(segments):
    '''
    Takes a numpy array of booleans and returns the start and end indexes
    containing the intervals [inclusive].
    '''
    starts = segments & ~(numpy.roll(segments, 1))
    starts[0] = segments[0]
    ends = segments & ~numpy.roll(segments, -1)
    ends[-1] = segments[-1]
    return zip((s[0] for s in numpy.argwhere(starts)),
               (e[0] for e in numpy.argwhere(ends)))


class TrialResults(object):
    def __init__(self):
        self.name = ''
        self.coverage = None
        self.n_gens = None

        empty_hist, breaks = numpy.histogram([], N_HIST_BINS, HIST_RANGE)
        self.post_prob_hist_ibd = empty_hist
        self.post_prob_hist_noibd = empty_hist.copy()
        self.post_prob_hist_breaks = breaks

        self.relatedness = ConfusionTable()
        self.positional = ConfusionTable()
        self.segments_dfs = []

    def update(self, pos, post_prob, ibd_segs, called_segs):
        """
        Updates adds the HMM results a comparison to the current set
        of results.
        """
        self.relatedness.update(numpy.array([ibd_segs.any()]),
                                numpy.array([called_segs.any()]))
        self.positional.update(ibd_segs, called_segs)

        self.post_prob_hist_ibd += numpy.histogram(post_prob[ibd_segs],
                                                   bins=N_HIST_BINS,
                                                   range=HIST_RANGE)
        self.post_prob_hist_noibd += numpy.histogram(post_prob[~ibd_segs],
                                                     bins=N_HIST_BINS,
                                                     range=HIST_RANGE)
        if called_segs.any():
            self._update_segments(pos, ibd_segs, called_segs)

    def _update_segments(self, pos, ibd_segs, called_segs):
        """
        Helper function for processing IBD calls as segments.
        """
        seg_interval_indexes = get_segment_intervals(called_segs)
        segs_df = pandas.DataFrame(columns=['start_idx',
                                            'end_idx',
                                            'num_snps',
                                            'physical_length'])
        # TODO:
        #                                   'genetic_length',
        #                                   'snp_overlap_with_ibd',
        #                                   'bp_overlap_with_ibd',
        #                                   'cm_overlap_with_ibd'])
        segs_df[['start_idx', 'end_idx']] = seg_interval_indexes
        segs_df['num_snps'] = segs_df['end_idx'] - segs_df['start_idx'] + 1
        segs_df['physical_length'] = (
              segs_df['end_idx'].map(lambda x: pos[x])
            - segs_df['start_idx'].map(lambda x: pos[x])
            + 1)

    def dump(self, output_dir):
        pass


class ConfusionTable(object):
    def __init__(self):
        """
        """
        self.contingency = numpy.zeros((2, 2), dtype=numpy.uint32)
        return

    def update(self, ground_truth, called_state):
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
        self.contingency[0, 0] += (~ground_truth & ~called_state).sum()
        self.contingency[0, 1] += (~ground_truth & called_state).sum()
        self.contingency[1, 0] += (ground_truth & ~called_state).sum()
        self.contingency[1, 1] += (ground_truth & called_state).sum()
        return

    def true_negatives(self):
        return self.contingency[0, 0]
    def false_positives(self):
        return self.contingency[0, 1]
    def false_negatives(self):
        return self.contingency[1, 0]
    def true_positives(self):
        return self.contingency[1, 1]

    def sensitivity(self):
        return numpy.float(self.contingency[1, 1]) / sum(self.contingency[1, ])
    def false_positive_rate(self):
        return numpy.float(self.contingency[0, 1]) / sum(self.contingency[0, ])
    def false_discovery_rate(self):
        return numpy.float(self.contingency[0, 1]) / sum(self.contingency[:, 1])
