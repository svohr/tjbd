"""
Contains functions and classes for keeping track of results from test
runs of tjbd.
"""

import numpy
import pandas


N_HIST_BINS = 1000
HIST_RANGE = (0.0, 1.0)


def get_segment_intervals(segments):
    """
    Takes a numpy array of booleans and returns the start and end indexes
    containing the intervals [inclusive].
    """
    starts = segments & ~(numpy.roll(segments, 1))
    starts[0] = segments[0]
    ends = segments & ~numpy.roll(segments, -1)
    ends[-1] = segments[-1]
    return zip([s[0] for s in numpy.argwhere(starts)],
               [e[0] for e in numpy.argwhere(ends)])


class TrialResults(object):
    """
    Class for storing aggregated results from several tjbd runs on artificial
    data for testing.

    Attributes:
        name: an identifier for the test run
        coverage: the simulated sequence coverage of the historical sample.
        n_gens: number of generations used in HMM transition probabilities.
        post_prob_hist_ibd: numpy array storing a histogram of HMM posterior
                            probabilities for markers in true IBD state.
        post_prob_hist_noibd: numpy array storing a histogram of HMM posterior
                              probabilities for markers not in an IBD interval.
        relatedness: A ConfusionTable comparing any detected IBD with true
                     sharing relationship.
        positional: A ConfusionTable for positional accuracy of IBD calls vs
                    true IBD state.
        segment_dfs: a list of pandas DataFrames storing the length and
                     fraction overlapping a true IBD state for all called
                     IBD segments.
    """

    def __init__(self):
        """ initialize TrialResults object with empty results. """
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

    def update(self, pos, gpos, post_prob, ibd_segs, called_segs):
        """
        Updates adds the HMM results a comparison to the current set
        of results.

        Args:
            pos: array containing physical positions for each marker
            gpos: array containing genetic positions for each marker
            post_prob: array of HMM posterior probabilities
            ibd_segs: array of booleans indicating true IBD state (true=IBD)
            called_segs: array of booleans indicating called IBD state
        """
        self.relatedness.update(numpy.array([ibd_segs.any()]),
                                numpy.array([called_segs.any()]))
        self.positional.update(ibd_segs, called_segs)

        self.post_prob_hist_ibd += numpy.histogram(post_prob[ibd_segs],
                                                   bins=N_HIST_BINS,
                                                   range=HIST_RANGE)[0]
        self.post_prob_hist_noibd += numpy.histogram(post_prob[~ibd_segs],
                                                     bins=N_HIST_BINS,
                                                     range=HIST_RANGE)[0]
        if called_segs.any():
            self._update_segments(pos, gpos, ibd_segs, called_segs)

    def _update_segments(self, pos, gpos, ibd_segs, called_segs):
        """
        Helper function for processing IBD calls as segments.

        Args:
            pos: array containing physical positions for each marker
            post_prob: array of HMM posterior probabilities
            ibd_segs: array of booleans indicating true IBD state (true=IBD)
            called_segs: array of booleans indicating called IBD state
        """
        seg_interval_indexes = get_segment_intervals(called_segs)
        segs_df = pandas.DataFrame(columns=['start_idx',
                                            'end_idx',
                                            'num_snps',
                                            'physical_length',
                                            'genetic_length'])
        # TODO:
        #                                   'snp_overlap_with_ibd',
        #                                   'bp_overlap_with_ibd',
        #                                   'cm_overlap_with_ibd'])
        segs_df[['start_idx', 'end_idx']] = seg_interval_indexes
        segs_df['num_snps'] = segs_df['end_idx'] - segs_df['start_idx'] + 1
        segs_df['physical_length'] = (
            segs_df['end_idx'].map(lambda x: pos[x])
            - segs_df['start_idx'].map(lambda x: pos[x])
            + 1)
        segs_df['genetic_length'] = (
            segs_df['end_idx'].map(lambda x: gpos[x])
            - segs_df['start_idx'].map(lambda x: gpos[x])
            + 1)
        self.segments_dfs.append(segs_df)

    def dump(self, output_dir):
        """
        Write out the aggregated results from this trial to a series of files
        in the specified directory.

        Args:
            output_dir: path to directory to write table to.
        """
        # Make a DataFrame for posterior probabilities.
        # Write table for positional accuracy and relatedness detection.
        # Combine segment DataFrames into a single data frame.
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
