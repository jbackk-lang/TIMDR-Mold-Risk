"""
api.py — TIMDR Mold Risk, lokalne REST API
================================================================
Serwer Flask udostepniajacy:
  GET  /api/health        -> healthcheck samego API
  GET  /api/scenarios     -> lista dostepnych scenariuszy demo
  GET  /api/demo          -> syntetyczny zestaw danych (?scenario=<nazwa>)
  POST /api/analyze       -> pelna analiza TIMDR (fuse + twist/trend/anomalies/rhythm
                              + time_to_visible_growth/health_score)
  POST /api/ble/start, POST /api/ble/stop, GET /api/ble/status, GET /api/ble/live
                           -> czujnik Bluetooth na zywo (patrz ble_sensor.py)
  POST /api/ha/ingest, GET /api/ha/status, POST /api/ha/reset
                           -> most do Home Assistant (patrz README.md,
                              sekcja "Integracja z Home Assistant") - HA samo
                              czyta sensor swoja wbudowana integracja "Xiaomi
                              BLE" i PUSHuje odczyty tutaj przez automatyzacje +
                              rest_command; RESTful sensor w HA czyta wynik z
                              /api/ha/status z powrotem jako encje HA.

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

import os
import threading
import asyncio

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from demo_scenarios import DEFAULT_THRESHOLDS, SCENARIOS, make_demo_data
from timdr_mold_fusion import TIMDRMoldFusion, VULNERABILITY_CLASSES
from timdr_mold_predict import TIMDRMoldPredict, risk_level_label
from ble_sensor import LiveReadingBuffer, scan_forever

app = Flask(__name__)

fusion = TIMDRMoldFusion()
predict = TIMDRMoldPredict()

REQUIRED_FIELDS = ["temperature", "humidity"]

# --- Czujnik Bluetooth na zywo (Xiaomi Mijia LYWSD03MMC, custom firmware) ---
# Zobacz ble_sensor.py - UWAGA, parser nie byl testowany na prawdziwym
# sprzecie w tym srodowisku (brak adaptera Bluetooth w sandboxie).
ble_buffer = LiveReadingBuffer()
_ble_thread = None
_ble_stop_event = threading.Event()

# --- Most do Home Assistant: HA samo czyta sensor (integracja "Xiaomi BLE"),
# a swoja automatyzacja PUSHuje odczyty tutaj (POST /api/ha/ingest). Osobny
# bufor niz ble_buffer, bo to inne zrodlo danych (nie ten sam proces skanuje).
ha_buffer = LiveReadingBuffer()


def _run_full_analysis(t, temperature, humidity, vulnerability_class="bardzo_wrazliwy",
                        threshold=3.0, window_hours=168):
    """Wspolna logika analizy TIMDR, uzywana zarowno przez POST /api/analyze
    (surowe dane w body), jak i GET /api/ha/status (dane z ha_buffer) - zeby
    nie utrzymywac dwoch kopii tej samej logiki fuse/twist/trend/anomalies/
    rhythm/fusion_score/predict."""
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

    def clean(x):
        if x is None:
            return None
        x = float(x)
        return None if not np.isfinite(x) else x

    def clean_list(x):
        return [clean(v) for v in np.asarray(x, float)]

    return {
        "t_hours": np.asarray(t, float).tolist(),
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
    }


def _ble_thread_target():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(scan_forever(ble_buffer, stop_event=_ble_stop_event))
    except Exception as exc:  # noqa: BLE001 - watek w tle, musi przezyc dowolny blad
        ble_buffer.last_error = f"blad watku skanowania BLE: {exc}"
    finally:
        loop.close()


@app.route("/")
def index():
    return jsonify({
        "service": "TIMDR-Mold-Risk",
        "endpoints": [
            "/dashboard",
            "/api/health", "/api/scenarios", "/api/demo", "/api/analyze (POST)",
            "/api/ble/start (POST)", "/api/ble/stop (POST)", "/api/ble/status", "/api/ble/live",
            "/api/ha/ingest (POST)", "/api/ha/status", "/api/ha/reset (POST)",
        ],
    })


@app.route("/dashboard")
def dashboard():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")


@app.route("/api/ble/start", methods=["POST"])
def api_ble_start():
    global _ble_thread
    try:
        import bleak  # noqa: F401
    except ImportError:
        return jsonify({"error": "pakiet 'bleak' nie jest zainstalowany - uruchom: pip install bleak"}), 400

    body = request.get_json(force=True, silent=True) or {}
    mac = body.get("mac")
    if mac:
        ble_buffer.target_mac = mac

    if _ble_thread is None or not _ble_thread.is_alive():
        _ble_stop_event.clear()
        ble_buffer.last_error = None
        _ble_thread = threading.Thread(target=_ble_thread_target, daemon=True)
        _ble_thread.start()

    return jsonify({"status": "started", "target_mac": ble_buffer.target_mac})


@app.route("/api/ble/stop", methods=["POST"])
def api_ble_stop():
    _ble_stop_event.set()
    return jsonify({"status": "stopping"})


@app.route("/api/ble/status")
def api_ble_status():
    try:
        import bleak  # noqa: F401
        bleak_available = True
    except ImportError:
        bleak_available = False

    running = _ble_thread is not None and _ble_thread.is_alive()
    last = None
    if ble_buffer.readings:
        unix_t, temp, hum, mac = ble_buffer.readings[-1]
        last = {"unix_time": unix_t, "temperature": temp, "humidity": hum, "mac": mac}

    return jsonify({
        "running": running,
        "bleak_available": bleak_available,
        "readings_count": len(ble_buffer.readings),
        "last_reading": last,
        "last_error": ble_buffer.last_error,
        "target_mac": ble_buffer.target_mac,
    })


@app.route("/api/ble/live")
def api_ble_live():
    t_hours, temps, hums = ble_buffer.as_series()
    return jsonify({"t_hours": t_hours, "temperature": temps, "humidity": hums})


@app.route("/api/health")
def api_health():
    return jsonify({"status": "ok"})


@app.route("/api/scenarios")
def api_scenarios():
    return jsonify([
        {"id": name, "description": desc, "default_threshold": DEFAULT_THRESHOLDS[name]}
        for name, desc in SCENARIOS.items()
    ])


@app.route("/api/vulnerability-classes")
def api_vulnerability_classes():
    return jsonify(sorted(VULNERABILITY_CLASSES.keys()))


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
        result = _run_full_analysis(t, temperature, humidity, vulnerability_class, threshold, window_hours)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"blad analizy: {exc}"}), 400

    return jsonify(result)


# --- Most do Home Assistant --------------------------------------------
#
# HA CZYTA sensor samo (wbudowana integracja "Xiaomi BLE" dla custom
# firmware pvvx/ATC - nie potrzeba tu bleak/ble_sensor.py wcale, bo HA
# juz ma dostep do adaptera Bluetooth). Automatyzacja w HA PUSHuje kazdy
# nowy odczyt T/RH tutaj (POST /api/ha/ingest), a RESTful sensor w HA
# okresowo czyta wynik analizy z powrotem (GET /api/ha/status) jako
# zwykle encje HA. Pelna konfiguracja YAML: README.md, sekcja
# "Integracja z Home Assistant".

_EMPTY_HA_STATUS = {
    "readings_count": 0,
    "current_risk_level": 0,
    "current_risk_label": "brak danych - czekam na odczyty z Home Assistant",
    "M": None,
    "margin": None,
    "health_score": 1.0,
    "time_to_visible_growth_hours": None,
    "fusion_score": 0.0,
    "last_temperature": None,
    "last_humidity": None,
}


@app.route("/api/ha/ingest", methods=["POST"])
def api_ha_ingest():
    """Body (JSON): {"temperature": float, "humidity": float, "timestamp": opcjonalnie}
    `timestamp` moze byc unix-epoch (sekundy, liczba) albo ISO8601 string
    (np. to, co HA daje w `{{ now().isoformat() }}`) - jesli brak, uzywany
    jest czas serwera w momencie wywolania."""
    body = request.get_json(force=True, silent=True) or {}
    missing = [f for f in REQUIRED_FIELDS if f not in body]
    if missing:
        return jsonify({"error": f"brakujace pola: {missing}"}), 400

    try:
        temperature = float(body["temperature"])
        humidity = float(body["humidity"])
    except (TypeError, ValueError) as exc:
        return jsonify({"error": f"niepoprawne dane wejsciowe: {exc}"}), 400

    unix_time = None
    ts = body.get("timestamp")
    if ts is not None:
        try:
            unix_time = float(ts)
        except (TypeError, ValueError):
            try:
                import datetime
                s = str(ts).replace("Z", "+00:00")
                unix_time = datetime.datetime.fromisoformat(s).timestamp()
            except ValueError:
                return jsonify({"error": f"nie rozumiem formatu timestamp: {ts!r}"}), 400

    ha_buffer.add(temperature, humidity, mac="home-assistant", unix_time=unix_time)
    return jsonify({"status": "ok", "readings_count": len(ha_buffer.readings)})


@app.route("/api/ha/status")
def api_ha_status():
    """Zwraca ZAWSZE ten sam ksztalt JSON (nawet przy braku danych), zeby
    RESTful sensor w Home Assistant nie wywalal sie na brakujacych kluczach.
    Parametry query (opcjonalne): vulnerability_class, threshold, window_hours."""
    t_hours, temps, hums = ha_buffer.as_series()
    if len(t_hours) < 3:
        resp = dict(_EMPTY_HA_STATUS)
        resp["readings_count"] = len(t_hours)
        if temps:
            resp["last_temperature"] = temps[-1]
            resp["last_humidity"] = hums[-1]
        return jsonify(resp)

    vulnerability_class = request.args.get("vulnerability_class", "bardzo_wrazliwy")
    threshold = float(request.args.get("threshold", 3.0))
    window_hours = int(request.args.get("window_hours", 168))

    try:
        result = _run_full_analysis(t_hours, temps, hums, vulnerability_class, threshold, window_hours)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"blad analizy: {exc}"}), 400

    return jsonify({
        "readings_count": len(t_hours),
        "current_risk_level": result["current_risk_level"],
        "current_risk_label": result["current_risk_label"],
        "M": result["M"][-1] if result["M"] else None,
        "margin": result["margin"][-1] if result["margin"] else None,
        "health_score": result["health_score"],
        "time_to_visible_growth_hours": result["time_to_visible_growth_hours"],
        "fusion_score": result["fusion_score"],
        "last_temperature": temps[-1],
        "last_humidity": hums[-1],
    })


@app.route("/api/ha/reset", methods=["POST"])
def api_ha_reset():
    """Czysci bufor odczytow z Home Assistant (np. po usunieciu przyczyny
    wilgoci - zeby zaczac liczenie M(t) od zera zamiast dalej integrowac
    stary, juz nieaktualny sygnal)."""
    ha_buffer.clear()
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    # threaded=True: kazde zadanie HTTP w wlasnym watku, zeby ewentualne
    # zawieszenie skanowania BLE (np. systemowy monit Windows o pozwolenie
    # na Bluetooth) NIE blokowalo reszty API (wykresow scenariuszy demo,
    # ktore z BLE nie maja nic wspolnego).
    app.run(host="127.0.0.1", port=5002, debug=False, threaded=True)
