"""Single source of truth: NIH label → StatPearls article mapping."""

CONDITION_MAPPING = {
    "Atelectasis":       {"nbk_id": "NBK545316", "fetch": "api"},
    "Cardiomegaly":      {"nbk_id": "NBK542296", "fetch": "api"},
    "Consolidation":     {"nbk_id": None,         "fetch": "manual"},
    "Edema":             {"nbk_id": "NBK557611", "fetch": "api"},
    "Effusion":          {"nbk_id": "NBK448189", "fetch": "api"},
    "Emphysema":         {"nbk_id": "NBK482217", "fetch": "api"},
    "Fibrosis":          {"nbk_id": "NBK448162", "fetch": "api"},
    "Hernia":            {"nbk_id": "NBK562200", "fetch": "api"},
    "Infiltration":      {"nbk_id": None,         "fetch": "manual"},
    "Mass":              {"nbk_id": "NBK562307", "fetch": "api"},
    "Nodule":            {"nbk_id": "NBK556143", "fetch": "api"},
    "Pleural_Thickening":{"nbk_id": None,         "fetch": "manual"},
    "Pneumonia":         {"nbk_id": "NBK430749", "fetch": "api"},
    "Pneumothorax":      {"nbk_id": "NBK441885", "fetch": "api"},
}