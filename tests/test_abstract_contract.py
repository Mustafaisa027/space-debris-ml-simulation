from __future__ import annotations

import json
import re
from pathlib import Path

from space_debris.core import PairResult
from space_debris.experiment import load_experiment_config
from space_debris.ml import REPORT_COLUMNS, _build_models


CONTRACT_PATH = Path("config/iac_114764_abstract_contract.json")


def _contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_accepted_abstract_contract_is_bound_to_exact_pdf_digest():
    contract = _contract()

    assert contract["schema_version"] == 1
    assert contract["paper_id"] == "114764"
    assert contract["accepted_abstract_pages"] == 1
    assert re.fullmatch(
        r"[0-9a-f]{64}", contract["accepted_abstract_pdf_sha256"]
    )
    assert (
        contract["accepted_abstract_pdf_sha256"]
        == "3614ec0335acfe4fec6b8730c6775044a6bf77ef82865b036cd8182e97d5a05d"
    )


def test_abstract_methods_are_present_in_claim_eligible_v2_pipeline():
    contract = _contract()
    requirements = contract["implementation_requirements"]
    config = load_experiment_config("config/experiment_10_days_v2.json")
    pair_fields = set(PairResult.__dataclass_fields__)
    model_names = set(_build_models(scale_pos_weight=1.0))

    assert Path("LICENSE").is_file()
    assert requirements["open_source"] is True
    assert requirements["region"] == "LEO"
    assert requirements["orbital_input"] == "TLE"
    assert requirements["propagator"] == "SGP4"
    assert set(requirements["dataset_fields"]).issubset(pair_fields)
    assert tuple(requirements["required_models"]) == config.required_models
    assert set(requirements["required_models"]).issubset(model_names)
    assert set(requirements["required_metrics"]).issubset(REPORT_COLUMNS)
    assert requirements["classical_comparator"] in model_names


def test_abstract_result_language_remains_fail_closed():
    contract = _contract()
    publication_source = Path("src/train_from_history.py").read_text(encoding="utf-8")
    evidence_source = Path("src/space_debris/evidence.py").read_text(encoding="utf-8")

    for claim in contract["empirical_claims"]:
        assert claim["policy"] == "fail_closed_until_final_real_data"
        gate = claim["publication_gate"]
        assert gate in publication_source
        assert gate in evidence_source
    assert (
        contract["interpretation_limits"]["probability_of_collision_claim"]
        is False
    )
