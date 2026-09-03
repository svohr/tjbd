"""
Merges and summarizes IBD segments for all individuals by individual,
subbranch, branch, and line groupings as defined by the "rel_branches.csv". IBD
segments are read from a directory "Rel_results" containing segment TSV files
from TJBD for each individual (detected segments for all chromosomes
concatenated into a single file).

Expected columns in "rel_branches.csv":
rel_id,new_id,subbranch,branch,line

Writes 2 files:
"Rel_branch_summaries.tsv" - IBD summaries by level
"Rel_branch_merged_segments.tsv" - IBD segments merged by level
"""

import pandas as pd


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


def chrom_start_key(s):
    if s.name == "chromosome":
        return s.str.extract(r"chr([0-9]*)").astype(int)[0]
    return s


def main():
    id_df = pd.read_csv("rel_branches.csv")
    print(id_df)
    rel_seg_dfs = []

    for row in id_df.itertuples():
        rel_seg_df = pd.read_csv(
            f"Rel_results/{row.rel_id}_segments.tsv",
            sep="\t",
        )
        rel_seg_df["new_id"] = row.new_id
        rel_seg_df["subbranch"] = row.subbranch
        rel_seg_df["branch"] = row.branch
        rel_seg_df["line"] = row.line
        rel_seg_dfs.append(
            rel_seg_df,
        )

    all_segs_df = (
        pd.concat(rel_seg_dfs)
        .rename(
            columns={
                "chrom": "chromosome",
                "start": "start_bp",
                "end": "end_bp",
                "genetic_length": "size_cm",
            },
        )
    )

    all_segs_df = all_segs_df[all_segs_df["chromosome"] != "chrX"]

    print(all_segs_df)

    thresholds = [5, 10]
    all_summary = []
    all_merged = []
    for t in thresholds:
        segs_gt_df = all_segs_df[all_segs_df["size_cm"] > t]
        for level in ["new_id", "subbranch", "branch", "line"]:
            merged_segs_df = segs_gt_df.groupby(level).apply(merge_segs)

            summary_df = (
                merged_segs_df
                .groupby(level)
                .aggregate({"chromosome": lambda x: len(set(x)), "start_bp":len, "size_cm": "sum"})
                .rename(
                    columns={
                        "size_cm": "total_cm",
                        "chromosome": "n_chromosome",
                        "start_bp": "n_segments",
                    }
                )
                .rename_axis("id")
            )
            summary_df["level"] = level
            summary_df["threshold"] = t
            print(summary_df)
            all_summary.append(summary_df)

            merged_segs_df = (
                merged_segs_df
                    .reset_index()
                    .rename(columns={level: "id"}).drop(columns=["level_1"])
                    .sort_values(
                        ["id", "chromosome", "start_bp"],
                        key=chrom_start_key,
                        ignore_index=True,
                    )
            )
            merged_segs_df["level"] = level
            merged_segs_df["threshold"] = t
            print(merged_segs_df)
            all_merged.append(merged_segs_df)
    all_summary_df = pd.concat(all_summary).reset_index()

    sh_adds_df = all_summary_df.set_index("id").loc[["MH", "EH"]].groupby("threshold").sum().reset_index()

    sh_adds_df["id"] = "EH+MH"
    sh_adds_df["level"] = "line_sum"

    all_summary_df = pd.concat([all_summary_df, sh_adds_df])
    all_summary_df[["id", "level", "threshold", "n_chromosome", "n_segments", "total_cm"]].to_csv("Rel_branch_summaries.tsv", sep="\t", index=False)

    all_merged_df = pd.concat(all_merged)
    all_merged_df[["id", "level", "threshold", "chromosome", "start_bp", "end_bp", "start_cm", "end_cm", "size_cm"]].to_csv("Rel_branch_merged_segments.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()

