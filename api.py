"""
api.py — TIMDR Mold Risk, lokalne REST API
================================================================
Serwer Flask udostepniajacy:
  GET  /api/health        -> healthcheck samego API
  GET  /api/scenarios     -> lista dostepnych scenariuszy demo
  GET  /api/demo          -> syntetyczny zestaw danych (?scenario=<nazwa>)
  POST /api/analyze       -> pelna analiza TIMDR (fuse + twist/trend/anomalies/rhythm
                              + time_to_visible_growth/health_score)

Uruchomienie: `python api.py`, potem http://127.0.0.1:5002 .

PODLACZALNOSC: TIMDRMoldFusion/TIMDRMoldPredict maja ten sam ksztalt
API co TIMDRSolarFusion/TIMDRBatteryFusion (fuse -> E,cos_dodatkowego;
twist/trend/anomalies/rhythm na E; fusion_score) - mozna skopiowac
(zwendorowac, z naglowkiem "ZWENDOROWANE" jak reszta ekosystemu)
timdr_mold_fusion.py + timdr_mold_predict.py do dowolnego innego repo.
Jedyna rzecz specyficzna dla tej domeny to argumenty fuse()
(temperature, humidity - w GODZINACH jako jednostka czasu, nie dniach
jak w Solar-PV).
"""

import numpy as np
from flask import Flask, jsonify, request

from demo_scenarios import DEFAULT_THRESHOLDS, SCENARIOS, make_demo_data
from timdr_mold_fusion import TIMDRMoldFusion
from timdr_mold_predict import TIMDRMoldPredict, risk_level_label

app = Flask(__name__)

fusion = TIMDRMoldFusion()
predict = TIMDRMoldPredict()

REQUIRED_FIELDS = ["temperature", "humidity"]


@app.route("/")
def index():
    return jsonify({
        "service": "TIMDR-Mold-Risk",
        "endpoints": ["/api/health", "/api/scenarios", "/api/demo", "/api/analyze (POST)"],
    })


@app.route("/api/health")
def api_health():
    return jsonify({"status": "ok"})


@app.route("/api/scenarios")
def api_scenarios():
    return jsonify([
        {"id": name, "description": desc, "default_threshold": DEFAULT_THRESHOLDS[name]}
        for name, desc in SCENARIOS.items()
    ])


@app.route("/api/demo")
def api_demo():
    scenario = request.args.get("scenario", "normal_ventilated_room")
    try:
        t, sensors = make_demo_data(scenario)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({
        "scenario": scenario,
        "default_threshold": DEFAULT_THRESHOLDS.get(scenario, 3.0),
        "t_hours": t.tolist(),
        "temperature": sensors["temperature"].tolist(),
        "humidity": sensors["humidity"].tolist(),
    })


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """
    Body (JSON):
      t_hours: [..]      - znaczniki czasu w GODZINACH
      temperature: [..]  - temperatura [degC]
      humidity: [..]     - wilgotnosc wzgledna [%]
      vulnerability_class: str="bardzo_wrazliwy"
      threshold: float=3.0     - prog "widocznego wzrostu" (indeks M)
      window_hours: int=168    - okno trendu/predykcji (domyslnie 1 tydzien)

    Zwraca pelny wynik analizy jako JSON.
    """
    body = request.get_json(force=True, silent=True) or {}

    missing_top = [f for f in ["t_hours"] + REQUIRED_FIELDS if f not in body]
    if missing_top:
        return jsonify({"error": f"brakujace pola: {missing_top}"}), 400

    try:
        t = np.asarray(body["t_hours"], dtype=float)
        temperature = np.asarray(body["temperature"], dtype=float)
        humidity = np.asarray(body["humidity"], dtype=float)
    except (TypeError, ValueError) as exc:
        return jsonify({"error": f"niepoprawne dane wejsciowe: {exc}"}), 400

    if len(t) == 0:
        return jsonify({"error": "t_hours nie moze byc puste"}), 400

    vulnerability_class = body.get("vulnerability_class", "bardzo_wrazliwy")
    threshold = float(body.get("threshold", 3.0))
    window_hours = int(body.get("window_hours", 168))

    try:
        local_fusion = TIMDRMoldFusion(vulnerability_class=vulnerability_class)
        E, margin = local_fusion.fuse(t, temperature, humidity)
        tw_idx, tw_z = local_fusion.twist(t, E)
        tr_sl, tr_z = local_fusion.trend(t, E, window=min(window_hours, max(2, len(t))))
        an_idx, an_z = local_fusion.anomalies(E)
        periods, r_score = local_fusion.rhythm(E)
        score = local_fusion.fusion_score(tw_z, tr_z, an_z, r_score)

        ttg = predict.time_to_visible_growth(t, E, threshold=threshold, window_hours=window_hours)
        health = predict.health_score(E, threshold=threshold)
        level, label = risk_level_label(E[-1]) if len(E) else (0, "brak danych")
    except (KeyError, Exception) as exc:  # noqa: BLE001
        return jsonify({"error": f"blad analizy: {exc}"}), 400

    def clean(x):
        if x is None:
            return None
        x = float(x)
        return None if not np.isfinite(x) else x

    def clean_list(x):
        return [clean(v) for v in np.asarray(x, float)]

    return jsonify({
        "t_hours": t.tolist(),
        "M": clean_list(E),
        "margin": clean_list(margin),
        "twist_idx": tw_idx.tolist(),
        "trend_slopes": clean_list(tr_sl),
        "anomaly_idx": an_idx.tolist(),
        "rhythm_periods": periods,
        "rhythm_score": clean(r_score),
        "fusion_score": clean(score),
        "time_to_visible_growth_hours": clean(ttg) if ttg != np.inf else None,
        "health_score": clean(health),
        "current_risk_level": level,
        "current_risk_label": label,
        "vulnerability_class": vulnerability_class,
        "threshold": threshold,
        "window_hours": window_hours,
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5002, debug=False)
