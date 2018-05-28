import pandas as pd

dct = {
    "db": {
        "base": "/home/alex/work/wf/DUAL_MODEL/IE/DB/stps/0_base/stats/extraction-results.csv",
        "_100_th_0": "/home/alex/work/wf/DUAL_MODEL/IE/DB/stps/2_100_rnd_th=0/stats/extraction-results.csv",
        "_100_th_0_abs": "/home/alex/work/wf/DUAL_MODEL/IE/DB/stps/3_100_rnd_th=0_absTag/stats/extraction-results.csv",
        "abs_tag_dir": "/home/alex/work/wf/DUAL_MODEL/IE/DB/0_absent_tag_classification/",
        "output": "/home/alex/work/wf/DUAL_MODEL/IE/DB/comparison.csv",
        "tags": ["invoice_amount", "invoice_invoicee", "invoice_currency", "invoice_invoicee_address", "invoice_iban",
                 "invoice_number", "invoice_swift", "invoice_bank_name", "invoice_bank_account_number", "invoice_date",
                 "vendor_name"]
    },
    "hpe": {
        "base": "/home/alex/work/wf/DUAL_MODEL/IE/HPE/stps/0_base/stats/extraction-results.csv",
        "_100_th_0": "/home/alex/work/wf/DUAL_MODEL/IE/HPE/stps/2_100_rnd_th=0/stats/extraction-results.csv",
        "_100_th_0_abs": "/home/alex/work/wf/DUAL_MODEL/IE/HPE/stps/3_100_rnd_th=0_absTag/stats/extraction-results.csv",
        "abs_tag_dir": "/home/alex/work/wf/DUAL_MODEL/IE/HPE/0_absent_tag_classification/",
        "output": "/home/alex/work/wf/DUAL_MODEL/IE/HPE/comparison.csv",
        "tags": ["bank_account_number", "bill_to_name", "invoice_amount", "invoice_currency", "invoice_date",
                 "invoice_number", "invoice_order_number", "vendor_bank_account", "vendor_city", "vendor_country",
                 "vendor_name", "vendor_state", "vendor_street", "vendor_zip"]
    }
}
data = dct["hpe"]

base = data["base"]
_100_th_0 = data["_100_th_0"]
_100_th_0_abs = data["_100_th_0_abs"]
abs_tag_dir = data["abs_tag_dir"]
output = data["output"]

base = pd.read_csv(base, index_col=0)
_100_th_0 = pd.read_csv(_100_th_0, index_col=0)
_100_th_0_abs = pd.read_csv(_100_th_0_abs, index_col=0)


def get_classification(file_name, tag_name):
    abs_tag = pd.read_csv(abs_tag_dir + tag_name + "/output/stp/stats/extraction-results.csv", index_col=0)
    return abs_tag.loc[file_name + ".json", tag_name + "_extracted"]


i = 0
df = pd.DataFrame()
for file_name in base.index:
    for tag in data["tags"]:
        df.loc[i, "file_name"] = file_name
        df.loc[i, "tag"] = tag
        df.loc[i, "gold_orig"] = base.loc[file_name, tag + "_original_gold"]
        df.loc[i, "gold"] = base.loc[file_name, tag + "_gold"]
        df.loc[i, "base_extr_orig"] = base.loc[file_name, tag + "_original_extracted"]
        df.loc[i, "base_extr"] = base.loc[file_name, tag + "_extracted"]
        df.loc[i, "base_error_type"] = base.loc[file_name, tag + "_error_type"]
        df.loc[i, "_100_th_0_extr_orig"] = _100_th_0.loc[file_name, tag + "_original_extracted"]
        df.loc[i, "_100_th_0_extr"] = _100_th_0.loc[file_name, tag + "_extracted"]
        df.loc[i, "_100_th_0_error_type"] = _100_th_0.loc[file_name, tag + "_error_type"]
        df.loc[i, "_100_th_0_abs_extr_orig"] = _100_th_0_abs.loc[file_name, tag + "_original_extracted"]
        df.loc[i, "_100_th_0_abs_error_type"] = _100_th_0_abs.loc[file_name, tag + "_error_type"]
        # df.loc[i, "_100_th_0_abs_extr"] = _100_th_0_abs.loc[file_name, tag + "_extracted"]
        df.loc[i, "_100_th_0_cl"] = get_classification(file_name, tag)
        # for tag1 in tags:
        #     df.loc[i, tag1] = get_classification(file_name, tag1)
        i += 1

df.to_csv(output, index=False)
# print(df)
