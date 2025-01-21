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


def prob_no_recomb(gens, gdist):
    """
    Returns the probability of no recombination occuring within the span of
    distance gdist (in centimorgans) across 'gens' generations.

    Args:
        gens: Number of generations (int > 1)
        gdist: A genetic distance in centimorgans.
    Returns:
        The probability of no recombinations occuring in 'gens' generations.
    """
    return numpy.exp(-0.01 * gdist) ** (gens - 1)


def state_trans(rec, gens, chrm, positions):
    """
    Takes RecMap object, a number of generations, a chromosome ID and a list
    of positions and returns two vectors that contain the transition
    probabilities of remaining in the IBD state and the no IBD state.

    Args:
        rec: A RecMap object containing a genetic map.
        gens: Number of generations between historic and present-day individual.
        chrm: The chromosome ID for the positions.
        positions: a numpy vector of base positions
    Returns:
        numpy vectors containing transition probabilities of remaining IBD
        and remaining in a no-IBD segment.
    """
    ibd_trs = numpy.empty(len(positions))

    ibd_trs[0] = 0.99999

    for i in range(1, len(positions)):
        gen_dist = rec.distance(chrm, positions[i - 1], positions[i])
        ibd_trs[i] = prob_no_recomb(gens, gen_dist)

    # Important: any recombination will break an IBD segment, but a
    # recombination in a no IBD segment can be in between two no IBD segments.
    # Weight noibd transitions accordingly.
    exp_noibd_prop = 1.0 - (1.0 / ((2.0 ** (gens - 1) - 1)))
    noibd_trs = ibd_trs + (exp_noibd_prop * (1.0 - ibd_trs))

    return ibd_trs, noibd_trs


def prob_obs_ibd(freq, obs_match, err=0.01):
    """
    Returns the probability of making observation "obs" within an IBD segment.

    Args:
        freq: Frequency of observed base in historic population.
        obs_match: Number of times in historical allele was observed in the
                   present-day individual (0, 1, or 2)
    Returns:
        The probability observation "obs_match" within an IBD segment.
    """
    freq = ((1.0 - err) * freq) + err
    if obs_match == 2:
        return (1 + freq) / 2.0
    elif obs_match == 1:
        return (1 + 2 * freq) / 4.0
    return freq / 2.0 # no match observed


def prob_obs_noibd(freq, _, err=0.01):
    """
    Returns the probability of making observation "obs" outside of an
    IBD segment.

    Args:
        freq: Frequency of observed base in historic population.
        obs_match: Number of times in historical allele was observed in the
                   present-day individual (0, 1, or 2). Not used in no IBD,
                   but kept as an argument for symmetry.
    Returns:
        The probability observation "obs_match" outside of an IBD segment.
    """
    # Does not matter if allele matches or not, outside of IBD the
    # probability of observation is always just the allele frequency.
    return ((1.0 - err) * freq) + err


def forward_backward(gens, observations, freqs, ibd_trs, noibd_trs):
    """
    Finds the posterior probability of the historic and present-day
    individuals sharing an IBD segment at each observed position using
    the forward-backward algorithm.

    Args:
        gens: number of generations between historic and present-day
              individual (int > 0)
        observations: Sequence of observations.
        freqs: Numpy vector of allele frequencies for each observed historical
               allele
        ibd_trs: Numpy vector containing probabilities of remaining in IBD
                 segment between the current and previous positions.
        noibd_trs: Numpy vector containing probabilities of remaining in no IBD
                   segment between the current and previous positions.
    Returns:
        A numpy vector containing the posterior probability of each position
        being in an IBD segment.
    """
    fwd_ibd_scaled = numpy.empty(len(observations))
    fwd_noibd_scaled = numpy.empty(len(observations))
    fwd_scale = numpy.empty(len(observations))

    # Forward probabilities
    # fill in the first, chance of starting in IBD is the expected fraction
    # of the genome inherited from an ancestor N generations ago.
    init_ibd_prob = 0.5 ** (gens - 1)
    fwd_ibd_scaled[0] = (prob_obs_ibd(freqs[0], observations[0])
                         * init_ibd_prob)
    fwd_noibd_scaled[0] = (prob_obs_noibd(freqs[0], observations[0])
                           * (1.0 - init_ibd_prob))
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

    for i in range(len(observations) - 2, -1, -1):
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


def logprob_obs_ibd(freq, obs_match):
    """
    To avoid clutter from calls to numpy.log, wraps a call to prob_obs_ibd
    with log transform.

    Args:
        freq: Frequency of observed base in historic population.
        obs_match: boolean whether historical allele matches one of the
                   present day alelle.
    Returns:
        The log-probability observation "obs_match" within an IBD segment.
    """
    return numpy.log(prob_obs_ibd(freq, obs_match))


def logprob_obs_noibd(freq, obs_match):
    """
    To avoid clutter from calls to numpy.log, wraps a call to prob_obs_noibd
    with log transform.

    Args:
        freq: Frequency of observed base in historic population.
        obs_match: boolean whether historical allele matches one of the
                   present day alelle.
    Returns:
        The log-probability observation "obs_match" not in an IBD segment.
    """
    return numpy.log(prob_obs_noibd(freq, obs_match))


def forward_backward_log_prob(gens, observations, freqs, ibd_trs, noibd_trs):
    """
    Finds the posterior probability of the historic and present-day
    individuals sharing an IBD segment at each observed position using
    the forward-backward algorithm. This function uses log-probabilities
    internally rather than a scaling factor to avoid underflow.

    Args:
        gens: number of generations between historic and present-day
              individual (int > 0)
        observations: Sequence of observations.
        freqs: Numpy vector of allele frequencies for each observed historical
               allele
        ibd_trs: Numpy vector containing probabilities of remaining in IBD
                 segment between the current and previous positions.
        noibd_trs: Numpy vector containing probabilities of remaining in no IBD
                   segment between the current and previous positions.
    Returns:
        A numpy vector containing the posterior probability of each position
        being in an IBD segment.
    """
    fwd_ibd = numpy.empty(len(observations))
    fwd_noibd = numpy.empty(len(observations))

    ibd_stay = numpy.log(ibd_trs)
    ibd_switch = numpy.log(1.0 - ibd_trs)
    noibd_stay = numpy.log(noibd_trs)
    noibd_switch = numpy.log(1.0 - noibd_trs)

    # forward probabilities
    # fill in the first, chance of starting in IBD is 1 / N_generations
    fwd_ibd[0] = (logprob_obs_ibd(freqs[0], observations[0])
                  + numpy.log(1.0 / gens))
    fwd_noibd[0] = (logprob_obs_noibd(freqs[0], observations[0])
                    + numpy.log((gens - 1.0) / gens))

    for i, obs in enumerate(observations):
        if i == 0:
            continue
        fwd_ibd[i] = numpy.logaddexp(logprob_obs_ibd(freqs[i], obs)
                                     + fwd_ibd[i - 1]
                                     + ibd_stay[i],
                                     logprob_obs_ibd(freqs[i], obs)
                                     + fwd_noibd[i - 1]
                                     + noibd_switch[i])
        fwd_noibd[i] = numpy.logaddexp(logprob_obs_noibd(freqs[i], obs)
                                       + fwd_noibd[i - 1]
                                       + noibd_stay[i],
                                       logprob_obs_noibd(freqs[i], obs)
                                       + fwd_ibd[i - 1]
                                       + ibd_switch[i])

    # backward probabilities
    bwd_ibd = numpy.empty_like(fwd_ibd)
    bwd_noibd = numpy.empty_like(fwd_noibd)

    bwd_ibd[-1] = 0.0
    bwd_noibd[-1] = 0.0

    for i in range(len(observations) - 2, -1, -1):
        bwd_ibd[i] = numpy.logaddexp(ibd_stay[i + 1]
                                     + logprob_obs_ibd(freqs[i + 1],
                                                       observations[i + 1])
                                     + bwd_ibd[i + 1],
                                     ibd_switch[i + 1]
                                     + logprob_obs_noibd(freqs[i + 1],
                                                         observations[i + 1])
                                     + bwd_noibd[i + 1])
        bwd_noibd[i] = numpy.logaddexp(noibd_stay[i + 1]
                                       + logprob_obs_noibd(freqs[i + 1],
                                                           observations[i + 1])
                                       + bwd_noibd[i + 1],
                                       noibd_switch[i + 1]
                                       + logprob_obs_ibd(freqs[i + 1],
                                                         observations[i + 1])
                                       + bwd_ibd[i + 1])
    post_prob = (fwd_ibd + bwd_ibd) - numpy.logaddexp(fwd_ibd[-1],
                                                      fwd_noibd[-1])
    return numpy.exp(post_prob)


def find_ibd_blocks_basic(post_probs, hi_score, lo_score):
    """
    Takes a vector of posterior probablilities from the forward backward
    algorithm and identifies regions containing high posterior probabilities
    (>hi_score) that are uninterrupted by probabilities lower than lo_score.

    Args:
        post_probs: vector of posterior probabilities from the
                    forward-backward algorithm.
        hi_score: value required to begin and end a IBD block.
        lo_score: value that interrupts an IBD block.
    Returns:
        A vector indicating whether a position is in an IBD block (True) or
        not (False) for each position.
    """
    called_ibd = numpy.zeros(len(post_probs), dtype=numpy.bool)

    in_ibd = False
    block_start = None
    block_stop = None

    for i, prob in enumerate(post_probs):
        if not in_ibd:
            if prob > hi_score:
                in_ibd = True
                block_start = i
                block_stop = i
        else:
            if prob > hi_score:
                block_stop = i
            elif prob < lo_score and block_stop > block_start:
                called_ibd[block_start:block_stop + 1] = True
                in_ibd = False
    if in_ibd and block_stop > block_start:
        called_ibd[block_start:block_stop + 1] = True
    return called_ibd


def find_ibd_blocks(post_probs, hi_score, lo_score, pos=None, min_len=0, merge_len=0):
    """
    Takes a vector of posterior probabilities from the forward backward
    algorithm and finds regions where the probability exceeds lo_score,
    containing at least one point where hi_score is met and that are
    longer than min_len as defined by the vector of positions (can be
    either physical or genetic positions).

    Args:
        post_probs: vector of posterior probabilities from the
                    forward-backward algorithm.
        hi_score: value required to begin and end a IBD block.
        lo_score: value that interrupts an IBD block.
        pos: a vector of positions (physical or genetic) for each marker in
             post_probs.
        min_len: a minimum length that a contiguous segment must meet.
        merge_len: merge blocks within this distance.
    Returns:
        A vector indicating whether a position is in an IBD block (True) or
        not (False) for each position.
    """
    def set_segment_ibd(start, end):
        """
        Calls a segment that has met the lo_score threshold if the hi_score
        and length requirments are met.
        """
        ibd_state = (post_probs[start:end + 1] >= hi_score).any()
        if pos is not None:
            ibd_state &= (pos[end] - pos[start]) > min_len
        called_ibd[start:end + 1] = ibd_state

    called_ibd = numpy.zeros(len(post_probs), dtype=numpy.bool)

    in_ibd = False
    block_start = None
    block_stop = None

    for i, prob in enumerate(post_probs):
        if not in_ibd:
            if prob >= lo_score:
                in_ibd = True
                block_start = i
                block_stop = i
        else:
            if prob >= lo_score:
                block_stop = i
            else:
                set_segment_ibd(block_start, block_stop)
                in_ibd = False
    if in_ibd:
        set_segment_ibd(block_start, block_stop)

    if pos is not None:
        called_ibd = merge_blocks_by_dist(called_ibd, pos, merge_len)
    return called_ibd


def merge_blocks_by_dist(called_ibd, pos, merge_len):
    """
    Merges IBDs that are within the specified distance.

    Args:
        called_ibd: a vector of booleans indicating if the given position is
            in an IBD block.
        pos: a vector of positions (physical or genetic) for each marker in
             post_probs.
        merge_len: merge blocks within this distance.
    Returns:
        A vector indicating whether a position is in an IBD block (True) or
        not (False) for each position.
    """
    merged_ibd = called_ibd.copy()
    gap_starts = list(numpy.where(called_ibd[:-1] & ~called_ibd[1:])[0])
    gap_ends = list(numpy.where(~called_ibd[:-1] & called_ibd[1:])[0] + 1)

    if len(gap_starts) + len(gap_ends) < 2:
        return merged_ibd

    if gap_ends[0] < gap_starts[0]:
        gap_ends.pop(0)

    for start, end in zip(gap_starts[:len(gap_ends)], gap_ends):
        if pos[end] - pos[start] < merge_len:
            merged_ibd[start:end] = True

    return merged_ibd


def main():
    o = [False, False, False, True, True, False, True,
         False, False, False, False, False, True, False, False]
    f = numpy.array([0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.01,
                     0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1])
    i = numpy.array([0.9] * 15)
    n = numpy.array([0.9] * 15)

    print(forward_backward(6, o, f, i, n))
    return 0


if __name__ == "__main__":
    sys.exit(main())
