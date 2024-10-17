target_rule = {
    "NECROSIS": [[{"LOWER": "necrosis"}]],
    "MITOSIS": [[{"LOWER": {"in": ["mitosis", "mitosens", "mitose"]}}]],
    "GRADE": [
        [{"LOWER": "risk"}, {"LOWER": "stratification"}],
        [{"LOWER": {"in": ["grade", "grading"]}}],
    ],
}
