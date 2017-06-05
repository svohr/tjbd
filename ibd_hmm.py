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
        gens: Number of generations (int > 0)
        gdist: A genetic distance in centimorgans.
    Returns:
        The probability of no recombinations occuring in 'gens' generations.
    """
    return (1 - gdist / 100.) ** gens


def bg_rec_rate(start, end):
    """
    Returns a genetic distance based on an average recombination rate of
    1 centimorgan per megabase.

    Args:
        start: Start position of interval.
        end: End position of interval.
    Returns:
        a genetic distance between start and end.
    """
    return float(end - start) / 1000000.0


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

    ibd_trs[0] = 1.0

    for i in xrange(1, len(positions)):
        gen_dist = rec.distance(chrm, positions[i - 1], positions[i])
        if not gen_dist:
            gen_dist = bg_rec_rate(positions[i - 1], positions[i])
        ibd_trs[i] = prob_no_recomb(gens, gen_dist)

    # Important: any recombination will break an IBD segment, but a
    # recombination in a no IBD segment can be in between two no IBD segments.
    # Weight noibd transitions accordingly.
    noibd_trs = numpy.add(gens - 1, ibd_trs)
    noibd_trs = numpy.divide(noibd_trs, gens, noibd_trs)

    return ibd_trs, noibd_trs


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


def prob_obs_noibd(freq, _):
    """
    Returns the probability of making observation "obs" outside of an
    IBD segment.

    Args:
        freq: Frequency of observed base in historic population.
        _ (obs_match): boolean whether historical allele matches one of the
                       present day alelle. Not used in no IBD, but kept as
                       an argument for symmetry.
    Returns:
        The probability observation "obs_match" outside of an IBD segment.
    """
    # Does not matter if allele matches or not, outside of IBD the
    # probability of observation is always just the allele frequency.
    return freq


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
        observations: Sequence of observations. Numpy vector of bools
                      for whether historic matched one of the present-day
                      alleles for all observed positions.
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

    for i in xrange(len(observations) - 2, -1, -1):
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


def forward_backward(gens, observations, freqs, ibd_trs, noibd_trs):
    """
    Finds the posterior probability of the historic and present-day
    individuals sharing an IBD segment at each observed position using
    the forward-backward algorithm.

    Args:
        gens: number of generations between historic and present-day
              individual (int > 0)
        observations: Sequence of observations. Numpy vector of bools
                      for whether historic matched one of the present-day
                      alleles for all observed positions.
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
    # fill in the first, chance of starting in IBD is 1 / N_generations
    fwd_ibd_scaled[0] = prob_obs_ibd(freqs[0], observations[0]) * (1.0 / gens)
    fwd_noibd_scaled[0] = (prob_obs_noibd(freqs[0], observations[0])
                           * ((gens - 1.0) / gens))
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

    print forward_backward(6, o, f, i, n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
