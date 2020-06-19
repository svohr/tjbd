'''
Takes the posterior probability by position output from tjbd and calls IBD
segments with a given set of segment calling parameters.
'''
import argparse
import sys

import pandas

import results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pos_fn", metavar="tjbd.pos.tsv", type=str,
                        help="Posterior probability file from tjbd")
    parser.add_argument("-m", "--min-run", dest="min_run",
                        type=float, default=0.8,
                        help="Minimum post. prob to allow in a detected "
                             "IBD segment.")
    parser.add_argument("-M", "--max-run", dest="max_run", type=float,
                        metavar="M", default=0.98,
                        help="Call IBD runs using post. probs exceeding M.")
    parser.add_argument("-L", "--min-len", dest="min_len", type=float,
                        metavar="cM", default=0.0,
                        help="Minimum length of IBD segment to call.")
    parser.add_argument("--merge-len", dest="merge_len", type=float,
                        metavar="cM", default=0.0,
                        help="Merge IBD segments within this distance.")
    parser.add_argument("-o", "--out-prefix", type=str, default='output',
                        help="Set prefix for the output files.")
    args = parser.parse_args()

    pos_input_df = pandas.read_csv(
        args.pos_fn,
        sep='\t',
        dtype={'chrom': str, 'pos': int})

    args.chrom = pos_input_df['chrom'].iloc[0]
    assert (pos_input_df['chrom'] == args.chrom).all()

    pos_results_df, seg_results_df = results.make_results_output(
        args=args,
        ppos=pos_input_df['pos'],
        gpos=pos_input_df['gpos'],
        hmm_post_probs=pos_input_df['posterior'])
    pos_results_df.to_csv('{}.pos.tsv'.format(args.out_prefix),
                          index=False, sep='\t')
    seg_results_df.to_csv('{}.seg.tsv'.format(args.out_prefix),
                          index=False, sep='\t')
    return 0


if __name__ == '__main__':
    sys.exit(main())
