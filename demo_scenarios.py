"""
demo_scenarios.py — syntetyczne, fizycznie ugruntowane scenariusze
temperatury/wilgotności dla TIMDR-Mold-Risk.

Rozdzielczość: GODZINOWA (jednostka czasu modelu VTT), tak jak DALTON
po agregacji z surowej częstotliwości ~1 próbka/sekundę (patrz
real_dalton_test.py). Kontekst scenariuszy: typowe mieszkania indyjskie
z monsunowym klimatem (region, z którego pochodzi DALTON), bo to konkretny,
realny kontekst geograficzny datasetu, nie dowolny.

Wartości bazowe RH/T NIE są dobrane pod z góry założony wynik testu —
zostały ustalone z ogólnej wiedzy o typowych warunkach mieszkalnych
(RH 40-65% = komfortowe wg ASHRAE 55, >80% = krytyczne wg modelu VTT —
patrz timdr_mold_fusion.py) PRZED uruchomieniem mold_index_series() na
nich; sam PRZEBIEG M(t) (kiedy dokładnie przekracza próg 3) jest
wynikiem modelu, nie założeniem.
"""

import numpy as np

HOURS_PER_DAY = 24


def _time_hours(n_days):
    return np.arange(n_days * HOURS_PER_DAY, dtype=float)


def normal_ventilated_room(seed=0, n_days=60):
    """Dobrze wentylowany pokój: RH oscyluje 40-60% (dobowy cykl,
    minimum w nocy/rano - typowe), T 22-28 degC. Brak ryzyka - test
    kontroli negatywnej (M powinno zostac ~0 przez caly czas)."""
    rng = np.random.default_rng(seed)
    t = _time_hours(n_days)
    daily_phase = 2 * np.pi * (t % 24) / 24
    RH = 50.0 + 8.0 * np.sin(daily_phase - np.pi / 2) + rng.normal(0, 2.0, len(t))
    T = 25.0 + 2.0 * np.sin(daily_phase) + rng.normal(0, 0.5, len(t))
    RH = np.clip(RH, 20, 75)
    return t, {"temperature": T, "humidity": RH}


def poor_ventilation_bathroom(seed=1, n_days=45):
    """Łazienka bez wentylacji: bazowe RH 55%, codzienny prysznic o
    losowej porze podnosi RH do 90-95% na ok. 1h, potem WOLNY
    wykladniczy powrot (stala czasowa 4h - brak wentylacji = wolne
    schniecie) do bazowego poziomu. Test kumulacji ryzyka z powtarzanych,
    krotkich, ale czestych epizodow wysokiej wilgotnosci."""
    rng = np.random.default_rng(seed)
    t = _time_hours(n_days)
    n = len(t)
    RH = np.full(n, 55.0)
    T = np.full(n, 24.0) + rng.normal(0, 0.3, n)

    for day in range(n_days):
        shower_hour = day * 24 + rng.uniform(6, 9)  # prysznic rano
        peak_rh = rng.uniform(90, 95)
        decay_tau = 4.0
        mask = t >= shower_hour
        elapsed = t[mask] - shower_hour
        contribution = (peak_rh - 55.0) * np.exp(-elapsed / decay_tau)
        RH[mask] = np.maximum(RH[mask], 55.0 + contribution)

    RH += rng.normal(0, 1.5, n)
    RH = np.clip(RH, 30, 99)
    return t, {"temperature": T, "humidity": RH}


def monsoon_humidity_onset(seed=2, n_days=120, onset_day=30, monsoon_days=60):
    """30 dni normalnych warunkow (RH 55-65%), potem SEZONOWE
    (monsunowe) podwyzszenie RH do 85-92% przez `monsoon_days` dni,
    potem powrot do normy. Test dla trend() (powolne narastanie na
    poczatku sezonu) i anomalies()/twist() na przejsciach."""
    rng = np.random.default_rng(seed)
    t = _time_hours(n_days)
    n = len(t)
    daily_phase = 2 * np.pi * (t % 24) / 24
    RH = np.full(n, 60.0)
    T = 27.0 + 1.0 * np.sin(daily_phase) + rng.normal(0, 0.4, n)

    onset_h = onset_day * 24
    end_h = onset_h + monsoon_days * 24
    in_monsoon = (t >= onset_h) & (t < end_h)
    RH[in_monsoon] = 88.0 + 3.0 * np.sin(daily_phase[in_monsoon])
    RH += rng.normal(0, 2.0, n)
    RH = np.clip(RH, 40, 98)
    return t, {"temperature": T, "humidity": RH}


def water_leak_sudden(seed=3, n_days=50, leak_day=20):
    """Normalne warunki (RH~55%), potem NAGLY, TRWALY skok do RH~96%
    od dnia `leak_day` (np. peknieta rura w kacie pokoju) - nie wraca
    do konca serii. Test: twist() powinien wykryc SAM MOMENT przejscia
    (skokowa zmiana krzywizny M(t) przy starcie wzrostu), trend()
    powinien wykryc narastanie po nim."""
    rng = np.random.default_rng(seed)
    t = _time_hours(n_days)
    n = len(t)
    RH = np.full(n, 55.0) + rng.normal(0, 2.0, n)
    T = np.full(n, 23.0) + rng.normal(0, 0.3, n)
    leak_h = leak_day * 24
    RH[t >= leak_h] = 96.0 + rng.normal(0, 1.5, np.sum(t >= leak_h))
    RH = np.clip(RH, 30, 99)
    return t, {"temperature": T, "humidity": RH}


SCENARIOS = {
    "normal_ventilated_room": "Dobrze wentylowany pokoj, RH 40-60%, brak ryzyka",
    "poor_ventilation_bathroom": "Lazienka bez wentylacji, codzienne prysznice, powolne schniecie",
    "monsoon_humidity_onset": "Sezonowe (monsunowe) podwyzszenie RH na 60 dni",
    "water_leak_sudden": "Nagly, trwaly wyciek - skok RH do ~96% od danego dnia",
}

DEFAULT_THRESHOLDS = {
    "normal_ventilated_room": 3.0,
    "poor_ventilation_bathroom": 3.0,
    "monsoon_humidity_onset": 3.0,
    "water_leak_sudden": 3.0,
}

GENERATORS = {
    "normal_ventilated_room": normal_ventilated_room,
    "poor_ventilation_bathroom": poor_ventilation_bathroom,
    "monsoon_humidity_onset": monsoon_humidity_onset,
    "water_leak_sudden": water_leak_sudden,
}


def make_demo_data(scenario, **kwargs):
    if scenario not in GENERATORS:
        raise ValueError(f"Nieznany scenariusz '{scenario}'. Dostepne: {list(GENERATORS)}")
    return GENERATORS[scenario](**kwargs)
