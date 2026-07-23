from __future__ import annotations

import json

from capture_runtime_environment import runtime_environment, write_runtime_environment


def test_runtime_environment_records_python_platform_and_packages():
    report = runtime_environment()

    assert report["schema_version"] == 1
    assert report["python"]["version"]
    assert report["python"]["implementation"]
    assert report["platform"]["system"]
    assert any(package["name"].lower() == "pytest" for package in report["packages"])
    assert report["packages"] == sorted(
        report["packages"], key=lambda item: (item["name"].casefold(), item["version"])
    )


def test_write_runtime_environment_is_valid_json(tmp_path):
    output = tmp_path / "environment" / "runtime.json"

    write_runtime_environment(output)

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema_version"] == 1
    assert report["packages"]

