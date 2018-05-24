import pandas as pd


def get_df(file, index_col):
    return pd.read_table(file, encoding='utf-8', quotechar='"', sep='\t', index_col=index_col)


base_df = get_df(
    index_col=1,
    file="/home/alex/work/wf/CHUBBINT/detailed_part_of_body/model/1_base_tokenizer_stemmer/output/stp/stats_03-30-2018_13-13-14/per-tag-class-stats.csv"
    # file="/home/alex/work/wf/CHUBBINT/detailed_part_of_body/model/0_base/output/stp/stats_04-09-2018_12-36-38/per-tag-class-stats.csv"
)
base_df = base_df[["GOLDS", "F1"]]
base_df["F1"] = base_df["F1"] * 1

base_df.loc["Other"] = 0

dct = {
    # "50": "/home/alex/work/wf/CHUBBINT/detailed_part_of_body/model/classes_with_at_least_50_docs_grouped/output/stp/stats_04-06-2018_17-21-08/per-tag-class-stats.csv",
    # "100": "/home/alex/work/wf/CHUBBINT/detailed_part_of_body/model/classes_with_at_least_100_docs_grouped/output/stp/stats_04-06-2018_17-10-34/per-tag-class-stats.csv",
    # "200": "/home/alex/work/wf/CHUBBINT/detailed_part_of_body/model/classes_with_at_least_200_docs_grouped/output/stp/stats_04-06-2018_17-36-38/per-tag-class-stats.csv",
    "w2v": "/home/alex/work/wf/CHUBBINT/detailed_part_of_body/model/word_to_vec/output/stp/stats/per-tag-class-stats.csv",
}

for df_name in dct.keys():
    base_df = base_df.join(get_df(index_col=1, file=dct.get(df_name))["F1"], rsuffix="_" + df_name)
    base_df[df_name + "_diff"] = base_df["F1_" + df_name] - base_df["F1"]

base_df.to_csv("/home/alex/work/wf/CHUBBINT/detailed_part_of_body/model/compare.csv", encoding='utf-8', index=True,
               quotechar='"', sep='\t')
