class F:  # files
    def __init__(self):
        self.f = {
            M.db: {
                N.train_dir: "/home/alex/work/wf/DUAL_MODEL/IE/DB/data/train/",
                N.classif_models_dir: "/home/alex/work/wf/DUAL_MODEL/IE/DB/0_absent_tag_classification/",
                N.base_stp: "/home/alex/work/wf/DUAL_MODEL/IE/DB/stps/0_base/stats/per-tag-stats.csv",
                N.dual_stp: "/home/alex/work/wf/DUAL_MODEL/IE/DB/stps/1_dual/stats/per-tag-stats.csv",
                N._100_stp: "/home/alex/work/wf/DUAL_MODEL/IE/DB/stps/2_100_rnd_th=0/stats/per-tag-stats.csv",
                N._100_th_stp: "/home/alex/work/wf/DUAL_MODEL/IE/DB/stps/2_100_rnd_th=AVG-STD/stats/per-tag-stats.csv",
                N._100_cl_stp: "/home/alex/work/wf/DUAL_MODEL/IE/DB/stps/3_100_rnd_th=0_absTag/stats/per-tag-stats.csv",
            },
            M.hpe: {

            }
        }
    db_train_dir = "/home/alex/work/wf/DUAL_MODEL/IE/data/db/train/"
    db_classification_models_dir = "/home/alex/work/wf/DUAL_MODEL/IE/model_17_db_class_absent_tag/"
    # stp_dir = "/output/stp/stats/"
    stp_stats_file = "/output/stp/stats/per-tag-stats.csv"
    base_db_dir = "/home/alex/work/wf/DUAL_MODEL/IE/model_19_db_base/base_db/"
    dual_db_dir = "/home/alex/work/wf/DUAL_MODEL/IE/model_11_base_db_dual/all_tags_0/"
    db_100_stp = "/home/alex/work/wf/DUAL_MODEL/IE/model_12_base_db_100/all_with_100_rnd_0/output/stp_100_rnd/stats/per-tag-stats.csv"
    db_100_th_stp = "/home/alex/work/wf/DUAL_MODEL/IE/model_12_base_db_100/all_with_100_rnd_0/output/stp_100_rnd_threshold_avg-std/stats/per-tag-stats.csv"
    db_100_cl_stp = "/home/alex/work/wf/DUAL_MODEL/IE/model/all_with_100_rnd_0/output/stp_100_rnd_abs_tag/stats/per-tag-stats.csv"

    base_hpe_dir = "/home/alex/work/wf/DUAL_MODEL/IE/model_20_hpe_base/base_hpe"


class M:  # models
    db = "DB"
    hpe = "HPE"


class A:  # approaches
    base = "base"
    _100 = "100"  # 100 rnd docs from training set for online models
    _100_th = "100_th"  # 100 rnd docs from training set for online models and threshold
    _100_cl = "100_cl"  # 100 rnd docs from training set for online models and classification model decision

class N:  # files names
    train_dir = "train_dir"
    classif_models_dir = "classification_models_dir"
    base_stp = "base_stp"
    dual_stp = "dual_stp"
    _100_stp = "_100_stp"
    _100_th_stp = "_100_th_stp"
    _100_cl_stp = "_100_cl_stp"
