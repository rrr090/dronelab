import json

from dronelab.hover.testing import SCENARIOS, PDController, hover_policy
from dronelab.hover.viewer import build_viewer_data, write_viewer


def test_viewer_export_roundtrip(tmp_path):
    data = build_viewer_data({"PD": PDController(), "HOVER": hover_policy},
                             [SCENARIOS["wind"], SCENARIOS["impulse"]], episodes=2)
    assert [s["name"] for s in data["scenarios"]] == ["wind", "impulse"]
    wind = data["scenarios"][0]
    assert wind["has_wind"] and len(wind["runs"]) == 4
    run = wind["runs"][0]
    assert len(run["x"]) == len(run["err"]) == len(run["t0"]) + 1 == len(run["fx"]) + 1

    html = write_viewer(data, tmp_path / "v.html").read_text(encoding="utf-8")
    assert "/*__DATA__*/" not in html
    payload = html.split("const DATA = ", 1)[1].split(";\n", 1)[0]
    assert json.loads(payload)["meta"]["policies"] == ["PD", "HOVER"]
