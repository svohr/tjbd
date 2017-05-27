"""
ibd_hmm.py

This file contains functions for applying the forward-backward algorithm
to obtain posterior probabilities for the hidden states (IBD and No IBD)
for each position in a chromosome.

"""

import sys
import numpy

#import recmap
#import freqs


def pos_to_idb_prob(rec, positions):
    """
    Takes RecMap object and a list of positions and returns a vector that
    contains the transition probabilities of remaining in the IBD state.

    Args:
        rec: A RecMap object containing a genetic map.
        positions: a numpy vector of base positions
    Returns:
        a numpy vector containing transition probabilities of remaining IBD
    """
    return


def prob_obs_ibd(freq, obs_match):
    """
    Returns the probability of making observation "obs" within an IBD segment.

    Args:
        freq: Frequency of observed base in historic population.
        obs_match: boolean whether historical allele matches one of the
                   present day alelle.
    Returns:
        The probability observation "obs_match" within an IBD segment.
    """
    if obs_match:
        return (1 + freq) / 2.0
    return freq / 2.0 # no match observed


def prob_obs_noibd(freq, obs_match):
    """
    Returns the probability of making observation "obs" outside of an
    IBD segment.

    Args:
        freq: Frequency of observed base in historic population.
        obs_match: boolean whether historical allele matches one of the
                   present day alelle.
    Returns:
        The probability observation "obs_match" outside of an IBD segment.
    """
    # Does not matter if allele matches or not, outside of IBD the
    # probability of observation is always just the allele frequency.
    return freq


def forward_backward(observations, freqs, ibd_trs, noibd_trs):
    """
    """
    fwd_ibd_scaled = numpy.empty(len(observations))
    fwd_noibd_scaled = numpy.empty(len(observations))
    fwd_scale = numpy.empty(len(observations))

    # Forward probabilities
    # fill in the first, assume an equal chance of starting in an ibd segment.
    fwd_ibd_scaled[0] = prob_obs_ibd(freqs[0], observations[0]) * 0.5
    fwd_noibd_scaled[0] = prob_obs_noibd(freqs[0], observations[0]) * 0.5
    fwd_scale[0] = fwd_ibd_scaled[0] + fwd_noibd_scaled[0]
    fwd_ibd_scaled[0] /= fwd_scale[0]
    fwd_noibd_scaled[0] /= fwd_scale[0]

    for i, obs in enumerate(observations):
        if i == 0:
            continue
        ibd_tmp = ((prob_obs_ibd(freqs[i], obs)
                    * fwd_ibd_scaled[i - 1]
                    * ibd_trs[i])
                   + (prob_obs_ibd(freqs[i], obs)
                      * fwd_noibd_scaled[i - 1]
                      * (1 - noibd_trs[i])))
        noibd_tmp = ((prob_obs_noibd(freqs[i], obs)
                      * fwd_noibd_scaled[i - 1]
                      * noibd_trs[i])
                     + (prob_obs_noibd(freqs[i], obs)
                        * fwd_ibd_scaled[i - 1]
                        * (1 - ibd_trs[i])))
        fwd_scale[i] = ibd_tmp + noibd_tmp
        fwd_ibd_scaled[i] = ibd_tmp / fwd_scale[i]
        fwd_noibd_scaled[i] = noibd_tmp / fwd_scale[i]

    # Backward probabilities
    bwd_ibd_scaled = numpy.empty(len(observations))
    bwd_noibd_scaled = numpy.empty(len(observations))

    bwd_ibd_scaled[-1] = 1.0
    bwd_noibd_scaled[-1] = 1.0

    for i in xrange(len(observations) - 2, -1, -1):
        scale = fwd_scale[i + 1]
        bwd_ibd_scaled[i] = ((ibd_trs[i + 1]
                              * prob_obs_ibd(freqs[i + 1],
                                             observations[i + 1])
                              * bwd_ibd_scaled[i + 1])
                             + ((1 - ibd_trs[i + 1])
                                * prob_obs_noibd(freqs[i + 1],
                                                 observations[i + 1])
                                * bwd_noibd_scaled[i + 1])) / scale
        bwd_noibd_scaled[i] = ((noibd_trs[i + 1]
                                * prob_obs_noibd(freqs[i + 1],
                                                 observations[i + 1])
                                * bwd_noibd_scaled[i + 1])
                               + ((1 - noibd_trs[i + 1])
                                  * prob_obs_ibd(freqs[i + 1],
                                                 observations[i + 1])
                                  * bwd_ibd_scaled[i + 1])) / scale

    # posterior decoding:
    return fwd_ibd_scaled * bwd_ibd_scaled


def main():
    o = [False, False, False, True, True, False, True,
         False, False, False, False, False, True, False, False]
    f = numpy.array([0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.01,
                     0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1])
    i = numpy.array([0.9] * 15)
    n = numpy.array([0.9] * 15)

    print forward_backward(o, f, i, n)

if __name__ == "__main__":
    sys.exit(main())
