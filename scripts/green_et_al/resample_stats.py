"""
Generates resampled scans per individual, merges IBD segments by branch,
computes ERSA likelihood under the 3 paternity scenarios and writes summary
statistics.

Expected columns
rel_branches.csv: rel_id,new_id,subbranch,branch,line
ersa_params.csv: scenario,branch,threshold,pois_fit,exp_fit
scanN_merged_segments.tsv (tab-separated): id,level,threshold,chromosome,start_bp,end_bp,start_cm,end_cm,size_cm


Example usage:
python resample_stats.py rel_branches.csv ersa_params.csv scan1_merged_segments.tsv ... scanN_merged_segments.tsv
"""
import argparse
import itertools

import numpy as np
import pandas as pd
from scipy import stats


def merge_segs(segs_df):
    """Merges overlapping segments."""
    merged = []

    for chromosome, chrom_segs_df in segs_df.groupby("chromosome"):

        current_segment = None
        for segment in chrom_segs_df.sort_values(["start_bp", "end_bp"]).itertuples():

            if current_segment is None or segment.start_bp > current_segment["end_bp"]:
                # First segment or a segment that doesn't overlap the current.
                if current_segment is not None:
                    merged.append(current_segment)
                current_segment = {
                    "chromosome": chromosome,
                    "start_bp": segment.start_bp,
                    "end_bp": segment.end_bp,
                    "start_cm": segment.start_cm,
                    "end_cm": segment.end_cm,
                    "size_cm": segment.size_cm,
                }
            elif ((current_segment["start_bp"] <= segment.start_bp <= current_segment["end_bp"])
                and (segment.end_bp > current_segment["end_bp"])):
                # Next segment overlaps and goes beyond current segment.
                # Extend the current segment.
                current_segment["end_bp"] = segment.end_bp
                current_segment["end_cm"] = segment.end_cm
                current_segment["size_cm"] = current_segment["end_cm"] - current_segment["start_cm"]
            # else: current segment overlaps the next segment completely, do nothing.
        if current_segment is not None:
            merged.append(current_segment)

    merged_df = pd.DataFrame(merged)
    return merged_df


def resample_segs(scans_df):
    """
    Generates a resampled scan by sampling the results of a randomly chosen scan
    (with replacement) for each chromosome.

    Args:
        scans_df: DataFrame containing segments from all scans for an individual.
    Returns:
        a DataFrame with the resampled scan.
    """
    chroms = scans_df["chromosome"].unique()
    scans = scans_df["scan"].unique()

    all_segs = []
    for chrom in chroms:
        sample_scan = np.random.choice(scans)

        all_segs.append(
            scans_df[
                (scans_df["scan"] == sample_scan)
                & (scans_df["chromosome"] == chrom)
            ]
        )
    return pd.concat(all_segs)


def resample_branch(scans_df, branch_new_ids):
    """
    Generates resampled scans for all individuals in a branch and merges
    overlapping segments.

    Args:
        scans_df: DataFrame containing segments from all scans for all individuals.
    Returns:
        a DataFrame with the merged, resampled segments.
    """
    resampled_branch_segs = []
    for new_id in branch_new_ids:
        rel_scans_df = scans_df[
            (scans_df["level"] == "new_id")
            & (scans_df["id"] == new_id)
        ]

        resampled_segs_df = resample_segs(rel_scans_df)
        resampled_branch_segs.append(resampled_segs_df)

    resampled_branch_segs_df = pd.concat(resampled_branch_segs)

    merged_df = merge_segs(resampled_branch_segs_df)

    return merged_df


def calc_log_likelihood(param_row, segs_df):
    """Compute the log-likelihood using the adapted ERSA likelihood
    Args:
        param_row: Series with the model parameters.
        segs_df: DataFrame of segments.
    Returns:
        The log-likelihood of the observed segments under the model.
    """
    n_log_prob = np.log(stats.poisson.pmf(len(segs_df), mu=param_row["pois_fit"]))
    seg_log_prob = np.log(
        stats.expon.pdf(
            segs_df["size_cm"] - param_row["threshold"],
            scale=param_row["exp_fit"],
        )
    )
    return n_log_prob + seg_log_prob.sum()


def calc_stats(param_df, segs_df, branch, threshold):
    """
    Computes the log-likelihoods, likelihood ratios, and posterior probabilities.
    """
    param_df = param_df[
        (param_df["branch"] == branch)
        & (param_df["threshold"] == threshold)
    ].set_index("scenario")

    log_likes = pd.Series(
        {
            scenario: calc_log_likelihood(param_df.loc[scenario], segs_df)
            for scenario in ["scTJ", "scRJ", "scRJO"]
        }
    )

    likes = np.exp(log_likes)
    total_prob = likes.sum()

    return pd.Series(
        {
            "branch": branch,
            "threshold": threshold,
            "n_segments": len(segs_df),
            "total_size_cm": segs_df["size_cm"].sum(),
            "log_likelihood_TJ": log_likes["scTJ"],
            "log_likelihood_RJ": log_likes["scRJ"],
            "log_likelihood_RJO": log_likes["scRJO"],
            "likelihood_ratio_TJ_v_RJ": np.exp(log_likes["scTJ"] - log_likes["scRJ"]),
            "likelihood_ratio_TJ_v_RJO": np.exp(log_likes["scTJ"] - log_likes["scRJO"]),
            "likelihood_ratio_RJ_v_RJO": np.exp(log_likes["scRJ"] - log_likes["scRJO"]),
            "posterior_probability_TJ": likes["scTJ"] / total_prob,
            "posterior_probability_RJ": likes["scRJ"] / total_prob,
            "posterior_probability_RJO": likes["scRJO"] / total_prob,
        }
    )


def make_table_1(summaries, branches, thresholds):
    rows = []
    for branch, threshold in itertools.product(branches, thresholds):
        summary_df = summaries[branch, threshold]
        row = summary_df.loc[
            ["mean", "50%", "2.5%", "97.5%"],
            [
                "likelihood_ratio_TJ_v_RJ",
                "likelihood_ratio_TJ_v_RJO",
                "likelihood_ratio_RJ_v_RJO",
                "posterior_probability_TJ",
                "posterior_probability_RJ",
                "posterior_probability_RJO",
            ]
        ].T.stack()
        row.index = row.index.map("_".join)
        row["branch"] = branch
        row["threshold"] = threshold
        rows.append(row)

    df = pd.DataFrame(rows)
    return df[["branch", "threshold"] + [col for col in df.columns if col not in ["branch", "threshold"]]]


def make_supplemental_table_3(post_probs_independent_father_df, post_probs_independent_father_notes_df, prefix):
    for col in ["mean", "50%", "2.5%", "97.5%"]:
        df = (
            post_probs_independent_father_df
                .describe(percentiles=[0.025, 0.50, 0.975])
                .loc[col]
                .to_frame()
                .reset_index()
        )
        df[["EH", "MH"]] = df["index"].str.extract(r"([EM]H-[JORT]*)_([EM]H-[JORT]*)")
        df = df.set_index(["EH", "MH"])
        df = df[col].unstack().loc[["EH-TJ", "EH-RJ", "EH-RJO"], ["MH-TJ", "MH-RJ", "MH-RJO"]]

        df.to_csv(f"{prefix}_supplemental_table_{col}.csv")

    (
        post_probs_independent_father_notes_df
            .describe(percentiles=[0.025, 0.50, 0.975])
            .to_csv(f"{prefix}_supplemental_table_notes.csv")
    )


def calc_joint_statistics(samples, branches, threshold):
    eh_lls = samples["EH", threshold][["log_likelihood_TJ", "log_likelihood_RJ", "log_likelihood_RJO"]]
    mh_lls = samples["MH", threshold][["log_likelihood_TJ", "log_likelihood_RJ", "log_likelihood_RJO"]]

    # same father
    jt_likelihoods = np.exp(eh_lls + mh_lls)
    post_probs_same_father_df = jt_likelihoods.divide(jt_likelihoods.sum(axis=1), axis=0)
    post_probs_same_father_df.columns = [
        "posterior_probability_TJ",
        "posterior_probability_RJ",
        "posterior_probability_RJO",
    ]
    post_probs_same_father_df["likelihood_ratio_TJ_v_RJ"] = (
        jt_likelihoods["log_likelihood_TJ"] / jt_likelihoods["log_likelihood_RJ"]
    )
    post_probs_same_father_df["likelihood_ratio_TJ_v_RJO"] = (
        jt_likelihoods["log_likelihood_TJ"] / jt_likelihoods["log_likelihood_RJO"]
    )

    # independent fathers
    joint_likelihoods = np.exp(
        np.repeat(eh_lls.values, 3, axis=1)
        + np.tile(mh_lls.values, 3)
    )
    post_probs_independent_father_df = pd.DataFrame(
        joint_likelihoods / joint_likelihoods.sum(axis=1)[:, np.newaxis],
        columns=[
            f"posterior_probability_EH-{s1}_MH-{s2}"
            for s1, s2 in itertools.product(["TJ", "RJ", "RJO"], repeat=2)
        ]
    )

    notes_df = pd.DataFrame(
        {
            "posterior_probability_EH-TJ_MH-TJ":
            post_probs_independent_father_df["posterior_probability_EH-TJ_MH-TJ"],
            "posterior_probability_EH-TJ_OR_MJ_TJ":
            post_probs_independent_father_df[[c for c in post_probs_independent_father_df.columns if "TJ" in c]].sum(axis=1),
            "posterior_probability_NOT_EH-TJ_OR_MJ_TJ":
            post_probs_independent_father_df[[c for c in post_probs_independent_father_df.columns if "TJ" not in c]].sum(axis=1),
        },
    )
    return post_probs_same_father_df, post_probs_independent_father_df, notes_df


def main(rel_branches_csv, params_csv, scan_tsvs, n_samples, out_prefix):

    id_df = pd.read_csv(rel_branches_csv)
    param_df = pd.read_csv(params_csv)

    scans = []
    for i, scan_tsv in enumerate(scan_tsvs):
        scan_df = pd.read_table(scan_tsv)
        scan_df["scan"] = i + 1
        scans.append(scan_df)
    scans_df = pd.concat(scans, ignore_index=True)

    branches = param_df["branch"].unique()
    thresholds = param_df["threshold"].unique()

    print(branches, thresholds)

    samples = {}
    summaries = {}

    for branch, threshold in itertools.product(branches, thresholds):

        prefix = f"{out_prefix}_{branch}_{threshold}"

        branch_new_ids = id_df.loc[
            id_df["branch"] == branch,
        ]["new_id"]

        model_scans_df = scans_df[
            (scans_df["level"] == "new_id")
            & (scans_df["id"].isin(branch_new_ids))
            & (scans_df["threshold"] == threshold)
        ]

        stats_rows = []
        for i in range(n_samples):

            merged_df = resample_branch(model_scans_df, branch_new_ids)
            stats_rows.append(calc_stats(param_df, merged_df, branch, threshold))

        stats_df = pd.DataFrame(stats_rows)

        stats_df.to_csv(f"{prefix}_samples.csv", index=False)
        summary_df = stats_df.describe(percentiles=[0.025, 0.50, 0.975]).drop(columns="threshold")
        print(summary_df)
        summary_df.to_csv(f"{prefix}_summary.csv")

        samples[branch, threshold] = stats_df
        summaries[branch, threshold] = summary_df

    table1_df = make_table_1(summaries, branches, thresholds)
    table1_df.to_csv(f"{out_prefix}_table1.csv", index=False)

    for threshold in thresholds:
        (
            post_probs_same_father_df,
            post_probs_independent_father_df,
            post_probs_independent_father_notes_df,
        ) = calc_joint_statistics(samples, branches, threshold)
        post_probs_same_father_df.to_csv(f"{out_prefix}_joint_{threshold}_same_father_samples.csv", index=False)
        post_probs_same_father_df.describe(
            percentiles=[0.025, 0.50, 0.975],
        ).to_csv(f"{out_prefix}_joint_{threshold}_same_father_summary.csv")
        post_probs_independent_father_df.to_csv(f"{out_prefix}_joint_{threshold}_independent_father_samples.csv", index=False)
        post_probs_independent_father_df.describe(
            percentiles=[0.025, 0.50, 0.975],
        ).to_csv(f"{out_prefix}_joint_{threshold}_independent_father_summary.csv")

        make_supplemental_table_3(post_probs_independent_father_df, post_probs_independent_father_notes_df, f"{out_prefix}_joint_{threshold}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("-n", "--n-samples", type=int, default=1000)
    parser.add_argument("-o", "--out-prefix", type=str, default="resampled_ersa")
    parser.add_argument("rel_branches_csv")
    parser.add_argument("params_csv")
    parser.add_argument("scan_tsvs", nargs="+")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)
    main(args.rel_branches_csv, args.params_csv, args.scan_tsvs, args.n_samples, args.out_prefix)

