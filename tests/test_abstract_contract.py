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


def test_abstract_methods_are_present_in_claim_eligible_v4_pipeline():
    contract = _contract()
    requirements = contract["implementation_requirements"]
    config = load_experiment_config("config/experiment_15_days_v4.json")
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


def test_operator_documentation_matches_active_v4_contract():
    collection_guide = Path("docs/GITHUB_DATA_COLLECTION.md").read_text(
        encoding="utf-8"
    )
    plotting_guide = Path("docs/plotting_blueprint.md").read_text(encoding="utf-8")
    readme = Path("README.md").read_text(encoding="utf-8")

    for required_text in (
        "minutes `07`, `17`, `37` and `47` of every UTC hour",
        "separately requires two elapsed hours",
        "manifest schema 3",
        "config/experiment_15_days_v4.json",
        "2026-08-10T00:17:00Z",
        "108 of 120 slots",
        "at least 30% unique TLE hashes",
        "more than six consecutive bins",
    ):
        assert required_text in collection_guide

    for stale_active_instruction in (
        "minute 17 of every second UTC hour",
        "every newly generated authoritative bundle uses manifest schema 2",
        "python src/import_collection_archive.py ..\\space-debris-data\n",
        "python src/resimulate_snapshots.py --config config/experiment_60_days.json",
        "the frozen interval into half-open two-hour bins anchored at `16:17Z`",
    ):
        assert stale_active_instruction not in collection_guide

    assert "`iac26-15d-v4` accumulated history" in plotting_guide
    assert "Final IAC figures should use the 60-day history" not in plotting_guide
    assert "Active claim-eligible experiment (15-day v4)" in readme
    assert "60-day frozen-cohort protocol below" not in readme
    v4_protocol = Path("docs/EXPERIMENT_15D_V4.md").read_text(encoding="utf-8")
    assert "v2 and v3 bundles" in v4_protocol
    assert "never pooled" in v4_protocol
    assert "three-hour slot" in v4_protocol
    assert "Provider request floor: two elapsed hours" in v4_protocol
    assert "108/120 slots" in v4_protocol
    v3_protocol = Path("docs/EXPERIMENT_10D_V3.md").read_text(encoding="utf-8")
    assert "Provider-request spacing incident (corrected 2026-07-31)" in v3_protocol
    assert "29 sub-two-hour intervals before" in v3_protocol
    assert "total to 30 among the first 61" in v3_protocol
    assert "Quality-gate deviations and restoration" in v3_protocol
    assert "four enforcement sites" in v3_protocol


def test_frozen_v3_quality_gates_remain_enforced():
    cadence_source = Path("src/collection_cadence_health.py").read_text(
        encoding="utf-8"
    )
    evidence_source = Path("src/space_debris/evidence.py").read_text(
        encoding="utf-8"
    )
    ml_source = Path("src/space_debris/ml.py").read_text(encoding="utf-8")

    assert "len(set(hashes)) / len(hashes)" in cadence_source
    assert "Partition TLE-update diversity gate failed" in evidence_source
    assert "snapshot_coverage_fraction=" in ml_source
    assert "max_identical_tle_hash_run_bins=" in ml_source
