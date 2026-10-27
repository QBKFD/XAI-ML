from xai.models.base import NotApplicable, RuleModel
from xai.models.corels_model import CORELSModel
from xai.models.ids_model import IDSModel
from xai.models.mdl_model import MDLModel
from xai.models.ruleopt_model import RuleOptModel
from xai.models.trees import C45Model, CARTModel, RandomForestModel

# Order used in tables and figures: rule learners first, then baselines.
MODELS: dict[str, type[RuleModel]] = {
    cls.key: cls
    for cls in (
        RuleOptModel,
        MDLModel,
        IDSModel,
        CORELSModel,
        CARTModel,
        C45Model,
        RandomForestModel,
    )
}

RULE_LEARNERS = ("ruleopt", "mdl", "ids", "corels")
BASELINES = ("cart", "c45", "random_forest")


def get_model(key: str, **params) -> RuleModel:
    try:
        cls = MODELS[key]
    except KeyError:
        raise KeyError(f"Unknown model {key!r}. Known models: {', '.join(MODELS)}") from None
    return cls(**params)


__all__ = [
    "MODELS",
    "RULE_LEARNERS",
    "BASELINES",
    "NotApplicable",
    "RuleModel",
    "get_model",
    "CORELSModel",
    "IDSModel",
    "MDLModel",
    "RuleOptModel",
    "CARTModel",
    "C45Model",
    "RandomForestModel",
]
