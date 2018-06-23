"""
Contains functions and classes for keeping track of results from test
runs of tjbd.
"""

import numpy
import pandas

N_HIST_BINS = 500
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
        post_prob_hist_rel: numpy array storing a histogram of HMM posterior
                            probabilities for markers in any IBD state when
                            the pair shares some IBD.
        post_prob_hist_norel: numpy array storing a histogram of HMM posterior
                              probabilities for markers in any IBD state when
                              there is no IBD shared between the pair.
        relatedness: A ConfusionTable comparing any detected IBD with true
                     sharing relationship.
        positional: A ConfusionTable for positional accuracy of IBD calls vs
                    true IBD state.
        segment_dfs: a list of pandas DataFrames storing the length and
                     fraction overlapping a true IBD state for all called
                     IBD segments.
    """

    def __init__(self, name=None, ibd_seg_size=None,
                 coverage=None, n_gens=None, call_params=None):
        """ initialize TrialResults object with empty results. """
        self.name = name
        self.ibd_seg_size = ibd_seg_size
        self.coverage = coverage
        self.n_gens = n_gens
        self.call_params = call_params

        empty_hist, breaks = numpy.histogram([], N_HIST_BINS, HIST_RANGE)
        self.post_prob_hist_ibd = empty_hist
        self.post_prob_hist_noibd = empty_hist.copy()
        self.post_prob_hist_rel = empty_hist.copy()
        self.post_prob_hist_norel = empty_hist.copy()
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
        if ibd_segs.any():
            self.post_prob_hist_rel += numpy.histogram(post_prob,
                                                       bins=N_HIST_BINS,
                                                       range=HIST_RANGE)[0]
        else:
            self.post_prob_hist_norel += numpy.histogram(post_prob,
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
        segs_df = pandas.DataFrame(seg_interval_indexes,
                                   columns=['start_index', 'end_index'],
                                   dtype='int')

        segs_df['num_snps'] = segs_df['end_index'] - segs_df['start_index'] + 1
        segs_df['physical_length'] = (
            segs_df['end_index'].map(lambda x: pos[x])
            - segs_df['start_index'].map(lambda x: pos[x]))
        segs_df['genetic_length'] = (
            segs_df['end_index'].map(lambda x: gpos[x])
            - segs_df['start_index'].map(lambda x: gpos[x]))

        ibd_overlap = []
        for _, seg in segs_df.iterrows():
            ibd_overlap.append(
                ibd_segs[int(seg['start_index']): int(seg['end_index']) + 1].sum())

        segs_df['ibd_overlap'] = ibd_overlap
        self.segments_dfs.append(segs_df)

    def _add_run_meta_data_to_df(self, stats_df):
        """
        Adds columns for trial run parameters to the end of a data frame.

        Args:
            stats_df: a pandas Series or DataFrame
        """
        if self.name is not None:
            stats_df['trial_name'] = self.name
        if self.ibd_seg_size is not None:
            stats_df['ibd_seg_size'] = self.ibd_seg_size
        if self.coverage is not None:
            stats_df['coverage'] = self.coverage
        if self.n_gens is not None:
            stats_df['n_gens'] = self.n_gens
        if self.call_params is not None:
            stats_df['call_low'] = self.call_params[0]
            stats_df['call_high'] = self.call_params[1]
            stats_df['call_length'] = self.call_params[2]

    def dump(self, output_dir):
        """
        Write out the aggregated results from this trial to a series of files
        in the specified directory.

        Args:
            output_dir: path to directory to write table to.
        """
        # Make a DataFrame for posterior probabilities.
        post_prob_df = pandas.DataFrame(
            {'bin_start': self.post_prob_hist_breaks[:-1],
             'bin_end': self.post_prob_hist_breaks[1:],
             'ibd_count': self.post_prob_hist_ibd,
             'noibd_count': self.post_prob_hist_noibd,
             'rel_count': self.post_prob_hist_rel,
             'norel_count': self.post_prob_hist_norel})
        post_prob_df.to_csv(
            '{}/{}.posterior_probs_histogram.tab'.format(output_dir, self.name),
            columns=['bin_start', 'bin_end',
                     'ibd_count', 'noibd_count',
                     'rel_count', 'norel_count'],
            sep='\t', index=False)

        # Write table for positional accuracy and relatedness detection.
        relatedness_df = self.relatedness.to_df()
        self._add_run_meta_data_to_df(relatedness_df)
        relatedness_df.to_csv('{}/{}.relatedness_class.tab'.format(output_dir,
                                                                   self.name),
                              index=False, sep='\t')
        positional_df = self.positional.to_df()
        self._add_run_meta_data_to_df(positional_df)
        positional_df.to_csv('{}/{}.positional_class.tab'.format(output_dir,
                                                                 self.name),
                             index=False, sep='\t')

        # Combine segment DataFrames into a single data frame.
        segments_df = pandas.concat(self.segments_dfs)
        self._add_run_meta_data_to_df(segments_df)
        segments_df.to_csv('{}/{}.called_segments.tab'.format(output_dir,
                                                              self.name),
                           sep='\t', index=False)


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

    def to_df(self):
        df = pandas.DataFrame([[self.true_positives(),
                                self.true_negatives(),
                                self.false_positives(),
                                self.false_negatives()]],
                              columns=['true_positives',
                                        'true_negatives',
                                        'false_positives',
                                        'false_negatives'],
                              dtype='int')
        df['sensitivity'] = self.sensitivity()
        df['false_positive_rate'] = self.false_positive_rate()
        df['false_discovery_rate'] = self.false_discovery_rate()
        return df
