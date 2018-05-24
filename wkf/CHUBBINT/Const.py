from sklearn.linear_model import LogisticRegression, SGDClassifier, LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier


class File:
    WD = "/home/alex/work/wf/CHUBBINT"
    # use_case = "detailed_cause_of_injury"
    use_case = "detailed_part_of_body"
    DATA_DIR = "/".join([WD, use_case, "data"])
    data_body = "/".join([DATA_DIR, "Detailed_Part_of_Body_Train.csv"])
    data_injury = "/".join([DATA_DIR, "Detailed_Cause_of_Injury_Train.csv"])
    train = "/".join([DATA_DIR, "train.csv"])
    test = "/".join([DATA_DIR, "test.csv"])
    stop_words = "/home/alex/work/projects/DataAnalysis/wkf/CHUBBINT/english_stop_words.txt"


class Column:
    desc_tagged = "desc_tagged"
    category = "category"


class Category:
    def __init__(self):
        self.ABDOMEN_INCLUDING_GROIN = "Abdomen Including Groin"
        self.ANKLE = "Ankle"
        self.BODY_SYSTEMS_AND_MULTIPLE_BODY_SYSTEMS = "Body Systems and Multiple Body Systems"
        self.BRAIN = "Brain"
        self.BUTTOCKS = "Buttocks"
        self.CHEST__INC__RIBS__STERNUM__AND_SOFT_TISSUE_ = "Chest (inc. Ribs, Sternum, and Soft Tissue)"
        self.DISC = "Disc"
        self.EAR_S_ = "Ear(s)"
        self.ELBOW = "Elbow"
        self.EYE_S_ = "Eye(s)"
        self.FACIAL_BONES = "Facial Bones"
        self.FINGER_S_ = "Finger(s)"
        self.FOOT = "Foot"
        self.GREAT_TOE = "Great Toe"
        self.HAND = "Hand"
        self.HEART = "Heart"
        self.HIP = "Hip"
        self.INSUFFICIENT_INFO_TO_PROPERLY_IDENTIFY___UNCLASSIFIED = "Insufficient Info to Properly Identify - Unclassified"
        self.INTERNAL_ORGANS = "Internal Organs"
        self.KNEE = "Knee"
        self.LARYNX = "Larynx"
        self.LOW_BACK_AREA__INC__LUMBAR_AND_LUMBO_SACRAL_ = "Low Back Area (inc. Lumbar and Lumbo-Sacral)"
        self.LOWER_ARM = "Lower Arm"
        self.LOWER_LEG = "Lower Leg"
        self.LUMBAR_AND_OR_SACRAL_VERTEBRAE__VERTEBRAE_NOC_TRUNK_ = "Lumbar and/or Sacral Vertebrae (Vertebrae NOC Trunk)"
        self.LUNGS = "Lungs"
        self.MOUTH = "Mouth"
        self.MULTIPLE_BODY_PARTS = "Multiple Body Parts"
        self.MULTIPLE_HEAD_INJURY = "Multiple Head Injury"
        self.MULTIPLE_INJURY = "Multiple Injury"
        self.MULTIPLE_LOWER_EXTREMITIES = "Multiple Lower Extremities"
        self.MULTIPLE_TRUNK = "Multiple Trunk"
        self.MULTIPLE_UPPER_EXTREMITIES = "Multiple Upper Extremities"
        self.NO_PHYSICAL_INJURY = "No Physical Injury"
        self.NOSE = "Nose"
        self.PELVIS = "Pelvis"
        self.SACRUM_AND_COCCYX = "Sacrum and Coccyx"
        self.SHOULDER_S_ = "Shoulder(s)"
        self.SKULL = "Skull"
        self.SOFT_TISSUE = "Soft Tissue"
        self.SPINAL_CORD = "Spinal Cord"
        self.TEETH = "Teeth"
        self.THUMB = "Thumb"
        self.TOE_S_ = "Toe(s)"
        self.TRACHEA = "Trachea"
        self.UPPER_ARM__INC__CLAVICLE___SCAPULA_ = "Upper Arm (inc. Clavicle & Scapula)"
        self.UPPER_BACK_AREA__THORACIC_AREA_ = "Upper Back Area (Thoracic Area)"
        self.UPPER_LEG = "Upper Leg"
        self.VERTEBRAE = "Vertebrae"
        self.WHOLE_BODY = "Whole Body"
        self.WRIST = "Wrist"
        self.WRIST_S__AND_HAND_S_ = "Wrist(s) and Hand(s)"


class Models:
    LogisticRegression = LogisticRegression(
        penalty='l2',
        dual=False,
        tol=1e-4,
        C=1.0,
        fit_intercept=True,
        intercept_scaling=1,
        class_weight="balanced",
        random_state=None,
        # solver='liblinear',
        solver='newton-cg',
        max_iter=100,
        multi_class='ovr',
        verbose=0,
        warm_start=False,
        n_jobs=-1
    )

    LogisticRegressionCV = LogisticRegressionCV(
        Cs=10,
        fit_intercept=True,
        cv=3,
        dual=False,
        penalty='l2',
        scoring=None,
        solver='newton-cg',
        tol=1e-4,
        max_iter=100,
        class_weight="balanced",
        n_jobs=-1,
        verbose=0,
        refit=True,
        intercept_scaling=1.,
        multi_class='ovr',
        random_state=None
    )

    SGDClassifier = SGDClassifier(
        loss="hinge",
        penalty='elasticnet',
        alpha=0.00001,
        l1_ratio=0.15,
        fit_intercept=True,
        max_iter=80,
        tol=None,
        shuffle=False,
        verbose=0,
        n_jobs=-1,
        random_state=None,
        learning_rate="optimal",
        eta0=0.0,
        power_t=0.5,
        class_weight='balanced',
        warm_start=False,
        average=True,
        n_iter=None
    )

    DecisionTreeClassifier = DecisionTreeClassifier(
        criterion="entropy",
        splitter="random",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.,
        min_impurity_split=None,
        class_weight="balanced",
        presort=False
    )
