from pathlib import Path

from export_gui_data import export_gui_data


def test_export_gui_data_separates_replay_from_paper_aggregate(tmp_path: Path):
    encounters = tmp_path / "encounters.csv"
    encounters.write_text(
        "# source: validation.csv\n"
        "object_1,object_2,object_1_catalog_id,object_2_catalog_id,"
        "snapshot_utc,tca_utc,time_to_tca_min,current_distance_km,"
        "min_distance_km,relative_velocity_km_s,altitude_1_km,"
        "altitude_2_km,altitude_difference_km,relative_radial_km,"
        "relative_intrack_km,relative_crosstrack_km,approach_angle_deg,"
        "risk_score,fixed_threshold_alarm,risk_label\n"
        "AURA,DEBRIS,1,2,2026-01-01Z,2026-01-01T00:10:00Z,10,1000,"
        "12.5,14.2,700,710,10,1,2,3,90,2.0,1,1\n",
        encoding="utf-8",
    )
    series = tmp_path / "series.csv"
    series.write_text(
        "minute,utc_iso,distance_km\n0,2026-01-01Z,1000\n10,2026-01-01T00:10:00Z,12.5\n",
        encoding="utf-8",
    )
    output = tmp_path / "simulation.json"

    payload = export_gui_data(encounters, series, output)

    assert output.exists()
    assert payload["paper"]["models"][0]["pr_auc"] == 0.725
    assert payload["paper"]["evidence_status"] == "exploratory"
    replay = payload["replay"]["scenarios"][0]
    assert replay["label"] == "AURA × DEBRIS"
    assert replay["min_distance_km"] == 12.5
    assert replay["distance_series"][1]["distance_km"] == 12.5
    assert replay["replay_label"] == "Validation snapshot replay"
    assert "not probability of collision" in payload["disclaimer"]
