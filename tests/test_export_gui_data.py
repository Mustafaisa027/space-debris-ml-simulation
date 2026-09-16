import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import export_gui_data as export_mod


def _write_encounters(path: Path, rows: str) -> None:
    path.write_text(
        "# source: validation.csv\n"
        "object_1,object_2,object_1_catalog_id,object_2_catalog_id,"
        "snapshot_utc,tca_utc,time_to_tca_min,current_distance_km,"
        "min_distance_km,relative_velocity_km_s,altitude_1_km,"
        "altitude_2_km,altitude_difference_km,relative_radial_km,"
        "relative_intrack_km,relative_crosstrack_km,approach_angle_deg,"
        "risk_score,fixed_threshold_alarm,risk_label\n"
        f"{rows}",
        encoding="utf-8",
    )


def test_export_gui_data_separates_replay_from_paper_aggregate(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(export_mod, "DEFAULT_SUPPLEMENTAL_MODEL_REPORTS", ())
    encounters = tmp_path / "encounters.csv"
    _write_encounters(
        encounters,
        "AURA,DEBRIS,1,2,2026-01-01Z,2026-01-01T00:10:00Z,10,1000,"
        "12.5,14.2,700,710,10,1,2,3,90,2.0,1,1\n",
    )
    series = tmp_path / "series.csv"
    series.write_text(
        "minute,utc_iso,distance_km\n0,2026-01-01Z,1000\n10,2026-01-01T00:10:00Z,12.5\n",
        encoding="utf-8",
    )
    output = tmp_path / "simulation.json"

    payload = export_mod.export_gui_data(encounters, series, output)

    assert output.exists()
    assert payload["paper"]["models"][0]["pr_auc"] == 0.725
    assert payload["paper"]["evidence_status"] == "exploratory"
    assert "local_reruns" not in payload
    assert "supplemental_model_reports" not in payload["paper"]
    replay = payload["replay"]["scenarios"][0]
    assert replay["label"].startswith("AURA (1)")
    assert "DEBRIS (2)" in replay["label"]
    assert replay["min_distance_km"] == 12.5
    assert replay["distance_series"][1]["distance_km"] == 12.5
    assert replay["replay_label"] == "Validation snapshot replay"
    assert payload["replay"]["distance_series_source"] == series.as_posix()
    assert payload["replay"]["distance_series_metadata"]["generation_mode"] == "archived_first_scenario_only"
    assert "not probability of collision" in payload["disclaimer"]


def test_export_gui_data_moves_local_reruns_out_of_paper(tmp_path: Path, monkeypatch) -> None:
    report_a = tmp_path / "report_a.csv"
    report_a.write_text(
        "# source: synthetic.csv\n"
        "model,pr_auc,roc_auc,precision,recall,f1,accuracy,false_alarm_rate,false_positive,false_negative,true_positive,true_negative,train_rows,test_rows,train_pairs,test_pairs,train_positive_rows,test_positive_rows,train_positive_pairs,test_positive_pairs,train_positive_snapshots,test_positive_snapshots,excluded_rows,cutoff_utc,split,note\n"
        "logistic_regression,0.61,0.84,0.55,0.72,0.62,0.91,0.08,4,2,9,45,120,60,30,15,20,10,12,6,5,3,0,,random,\n",
        encoding="utf-8",
    )
    report_b = tmp_path / "report_b.csv"
    report_b.write_text(
        "# source: history.csv\n"
        "model,pr_auc,roc_auc,precision,recall,f1,accuracy,false_alarm_rate,false_positive,false_negative,true_positive,true_negative,train_rows,test_rows,train_pairs,test_pairs,train_positive_rows,test_positive_rows,train_positive_pairs,test_positive_pairs,train_positive_snapshots,test_positive_snapshots,excluded_rows,cutoff_utc,split,note\n"
        "not_enough_data,,,,,,,,,,,,2227,73,1318,54,127,2,89,2,17,1,1044,2026-07-13T01:19:26+00:00,pair_grouped_time,quality gate failed\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        export_mod,
        "DEFAULT_SUPPLEMENTAL_MODEL_REPORTS",
        (
            {"key": "report_a", "label": "Report A", "path": report_a},
            {"key": "report_b", "label": "Report B", "path": report_b},
        ),
    )
    encounters = tmp_path / "encounters.csv"
    _write_encounters(
        encounters,
        "AURA,DEBRIS,1,2,2026-01-01Z,2026-01-01T00:10:00Z,10,1000,12.5,14.2,700,710,10,1,2,3,90,2.0,1,1\n",
    )
    output = tmp_path / "simulation.json"

    payload = export_mod.export_gui_data(encounters, None, output)

    assert "supplemental_model_reports" not in payload["paper"]
    assert payload["local_reruns"]["source_note"] == export_mod.LOCAL_RERUN_SOURCE_NOTE
    reports = {report["key"]: report for report in payload["local_reruns"]["supplemental_model_reports"]}
    assert reports["report_a"]["rows"][0]["roc_auc"] == 0.84
    assert reports["report_a"]["rows"][0]["precision"] == 0.55
    assert reports["report_a"]["rows"][0]["false_alarm_rate"] == 0.08
    assert reports["report_b"]["rows"][0]["status"] == "not_enough_data"
    assert reports["report_b"]["rows"][0]["roc_auc"] is None
    assert reports["report_b"]["rows"][0]["note"] == "quality gate failed"


def test_export_gui_data_generates_series_for_all_selected_scenarios(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(export_mod, "DEFAULT_SUPPLEMENTAL_MODEL_REPORTS", ())
    encounters = tmp_path / "encounters.csv"
    _write_encounters(
        encounters,
        "AURA,DEBRIS,1,2,2026-01-01Z,2026-01-01T00:10:00Z,10,1000,12.5,14.2,700,710,10,1,2,3,90,3.0,1,1\n"
        "BETA,GAMMA,3,4,2026-01-01Z,2026-01-01T00:20:00Z,20,2000,25.0,12.0,705,715,10,4,5,6,45,2.0,0,0\n",
    )
    tle = tmp_path / "catalog.txt"
    tle.write_text("placeholder", encoding="utf-8")
    output = tmp_path / "simulation.json"

    def fake_tle_distance_series(tle_path: Path, scenario: dict[str, object], **_: object) -> list[dict[str, object]]:
        return [
            {
                "minute": 0.0,
                "utc_iso": str(scenario["snapshot_utc"]),
                "distance_km": float(scenario["current_distance_km"]),
            },
            {
                "minute": float(scenario["time_to_tca_min"]),
                "utc_iso": str(scenario["tca_utc"]),
                "distance_km": float(scenario["min_distance_km"]),
            },
        ]

    monkeypatch.setattr(export_mod, "_tle_distance_series", fake_tle_distance_series)

    payload = export_mod.export_gui_data(encounters, None, output, limit=2, tle_path=tle)

    assert len(payload["replay"]["scenarios"]) == 2
    assert len(payload["replay"]["scenarios"][0]["distance_series"]) == 2
    assert len(payload["replay"]["scenarios"][1]["distance_series"]) == 2
    assert payload["replay"]["distance_series_source"] == "generated_from_tle"
    assert payload["replay"]["distance_series_metadata"]["generation_mode"] == "tle_all_scenarios"


def test_export_gui_data_skips_mismatched_archived_series(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(export_mod, "DEFAULT_SUPPLEMENTAL_MODEL_REPORTS", ())
    encounters = tmp_path / "encounters.csv"
    _write_encounters(
        encounters,
        "AURA,DEBRIS,1,2,2026-01-01Z,2026-01-01T00:10:00Z,10,1000,12.5,14.2,700,710,10,1,2,3,90,2.0,1,1\n",
    )
    series = tmp_path / "series.csv"
    series.write_text(
        "minute,utc_iso,distance_km\n0,2026-01-01Z,999\n10,2026-01-01T00:10:00Z,12.5\n",
        encoding="utf-8",
    )
    output = tmp_path / "simulation.json"

    payload = export_mod.export_gui_data(encounters, series, output)

    assert payload["replay"]["distance_series_source"] is None
    assert payload["replay"]["scenarios"][0]["distance_series"] == []
    assert payload["replay"]["distance_series_metadata"]["generation_mode"] == "archived_series_mismatch_skipped"
    assert payload["replay"]["distance_series_metadata"]["archived_matches_first_scenario"] == "false"


def test_build_parser_defaults_distance_series_to_none() -> None:
    parser = export_mod.build_parser()
    args = parser.parse_args([])
    assert args.distance_series is None
