
import pandas

from tjbd import confusion, ibd_hmm


def make_results_output(args, ppos, gpos, hmm_post_probs):
    """
    Returns:
        pos_results_df: a DataFrame containing the positional results from
                        the HMM scan.
        seg_results_df: a DataFrame containing the results summarized by
                        segment.
    """
    called_ibd = ibd_hmm.find_ibd_blocks(post_probs=hmm_post_probs,
                                         hi_score=args.max_run,
                                         lo_score=args.min_run,
                                         pos=gpos,
                                         min_len=args.min_len,
                                         merge_len=args.merge_len)

    pos_results_df = pandas.DataFrame({'chrom': args.chrom,
                                       'pos': ppos,
                                       'gpos': gpos,
                                       'posterior': hmm_post_probs,
                                       'ibd_call': called_ibd})

    seg_interval_indexes = confusion.get_segment_intervals(called_ibd)
    seg_results_df = pandas.DataFrame(seg_interval_indexes,
                                      columns=['start_index', 'end_index'],
                                      dtype='int')
    seg_results_df['chrom'] = args.chrom

    seg_results_df['n_snps'] = (seg_results_df['end_index']
                                - seg_results_df['start_index'] + 1)

    seg_results_df['start'] = pos_results_df.loc[seg_results_df['start_index'], 'pos'].values
    seg_results_df['end'] = pos_results_df.loc[seg_results_df['end_index'], 'pos'].values
    seg_results_df['physical_length'] = (seg_results_df['end']
                                         - seg_results_df['start'])

    seg_results_df['start_cm'] = pos_results_df.loc[seg_results_df['start_index'], 'gpos'].values
    seg_results_df['end_cm'] = pos_results_df.loc[seg_results_df['end_index'], 'gpos'].values
    seg_results_df['genetic_length'] = (seg_results_df['end_cm']
                                        - seg_results_df['start_cm'])

    seg_results_df['max_post_prob'] = 0
    if len(seg_results_df) > 0:
        seg_results_df['max_post_prob'] = seg_results_df.apply(
            lambda r: hmm_post_probs[r['start_index']:r['end_index'] + 1].max(),
            axis=1
        )

    return (pos_results_df[['chrom', 'pos', 'gpos', 'posterior', 'ibd_call']],
            seg_results_df[['chrom', 'start', 'end',
                            'n_snps', 'physical_length',
                            'start_cm', 'end_cm', 'genetic_length',
                            'max_post_prob']])
