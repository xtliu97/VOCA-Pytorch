# training set: 8 subjects, 40 sentences
training_subject = [
    "FaceTalk_170728_03272_TA",
    "FaceTalk_170904_00128_TA",
    "FaceTalk_170725_00137_TA",
    "FaceTalk_170915_00223_TA",
    "FaceTalk_170811_03274_TA",
    "FaceTalk_170913_03279_TA",
    "FaceTalk_170904_03276_TA",
    "FaceTalk_170912_03278_TA",
]
training_sentence = [f"sentence{i:02d}" for i in range(1, 41)]

# validation set: 2 subjects, 20 sentences
validation_subject = ["FaceTalk_170811_03275_TA", "FaceTalk_170908_03277_TA"]
validation_sentence = [f"sentence{i:02d}" for i in range(21, 41)]

# test set: 2 subjects
test_subject = ["FaceTalk_170809_00138_TA", "FaceTalk_170731_00024_TA"]
