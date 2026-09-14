"""
timdr_mold_predict.py — TIMDR Mold Risk Predict
===================================================
Predykcyjne ostrzeganie: czas do widocznego wzrostu (indeks pleśni >= 3,
"10-30% pokrycia, widoczne gołym okiem" wg oryginalnej skali VTT) i
wynik zdrowia (health_score) — na sygnale E(t)=M(t) z
timdr_mold_fusion.py::TIMDRMoldFusion.fuse().

Ten sam wzorzec co TIMDRSolarPredict/TIMDRBatteryPredict: lokalna
regresja liniowa w oknie, ekstrapolacja do progu. UWAGA (uczciwe
zastrzeżenie inne niż w PV/baterii): M(t) jest z definicji OGRANICZONE
(0-6, nasyca się przy Mmax(phi)) — liniowa ekstrapolacja lokalnego
nachylenia jest rozsądna TYLKO gdy M jest wciąż daleko od nasycenia
(rosnące, niezakrzywione jeszcze przez k2). Gdy M zbliża się do
Mmax(phi), lokalne nachylenie samo maleje (k2->0), więc predykcja
naturalnie "spowalnia" razem z modelem - ale w okresach naprzemiennej
wilgoci (patrz ograniczenie #2 w timdr_mold_fusion.py) może to dawać
zbyt ostrożne (zbyt długie) prognozy dla scenariuszy, gdzie warunki
później się pogarszają.

SKALA INDEKSU (Viitanen & Ritschkoff 1991, cytowana w Viitanen & Ojanen
2007, ASHRAE Buildings X):
    0  Brak wzrostu             — zarodniki nieaktywne
    1  Ślady pleśni (mikroskop) — początkowe stadium wzrostu
    2  <10% pokrycia (mikroskop)
    3  10-30% pokrycia (WIDOCZNE gołym okiem) — nowe zarodniki
    4  30-70% pokrycia           — umiarkowany wzrost
    5  >70% pokrycia             — obfity wzrost
    6  Bardzo silny, zwarty wzrost — pokrycie ~100%
"""

import numpy as np

MOLD_INDEX_LABELS = {
    0: "brak wzrostu (zarodniki nieaktywne)",
    1: "slady pleśni - poziom mikroskopowy, poczatek wzrostu",
    2: "<10% pokrycia - poziom mikroskopowy",
    3: "10-30% pokrycia - WIDOCZNE golym okiem, nowe zarodniki",
    4: "30-70% pokrycia - umiarkowany wzrost",
    5: ">70% pokrycia - obfity wzrost",
    6: "bardzo silny/zwarty wzrost - pokrycie ok. 100%",
}


def risk_level_label(M_value):
    """Etykieta opisowa dla wartosci indeksu M (zaokraglanej w dol do
    najblizszego zdefiniowanego poziomu 0-6)."""
    level = int(np.clip(np.floor(M_value), 0, 6))
    return level, MOLD_INDEX_LABELS[level]


class TIMDRMoldPredict:
    def __init__(self, mad_scale=1.4826):
        self.mad_scale = mad_scale

    def _local_slope(self, t_hours, M, window_hours=168):
        """Regresja liniowa w ostatnim oknie (domyslnie 168h = 1 tydzien,
        skala czasowa modelu VTT), centrowana (t0=t_win[0]) - ten sam fix
        numeryczny co reszta ekosystemu."""
        t = np.asarray(t_hours, float)
        M = np.asarray(M, float)
        valid = np.isfinite(M)
        t_valid, M_valid = t[valid], M[valid]
        if len(t_valid) < 2:
            return None
        cutoff = t_valid[-1] - window_hours
        sel = t_valid >= cutoff
        t_win, M_win = t_valid[sel], M_valid[sel]
        if len(t_win) < 2:
            return None
        t0 = t_win[0]
        t_rel = t_win - t0
        A = np.column_stack([t_rel, np.ones_like(t_rel)])
        a, b = np.linalg.lstsq(A, M_win, rcond=None)[0]
        t_ref = t_valid[-1] - t0
        return a, b, t_ref

    def time_to_visible_growth(self, t_hours, M, threshold=3.0, window_hours=168):
        """Godziny OD OSTATNIEGO pomiaru do przekroczenia progu (domyslnie
        indeks 3 = widoczne gołym okiem). Zwraca np.inf gdy M nie rosnie
        w oknie (a<=0) lub juz jest >= progu (0.0)."""
        M = np.asarray(M, float)
        if M.size and M[-1] >= threshold:
            return 0.0
        fit = self._local_slope(t_hours, M, window_hours)
        if fit is None:
            return None
        a, b, t_ref = fit
        if a <= 0:
            return np.inf
        ttd = (threshold - b) / a - t_ref
        return float(max(0.0, ttd))

    def health_score(self, M, threshold=3.0, window=48):
        """Mediana OSTATNIEGO okna (nie cala historia), ten sam wzorzec
        co TIMDRBatteryPredict/TIMDRSolarPredict health_score."""
        M = np.asarray(M, float)
        valid = M[np.isfinite(M)]
        if valid.size == 0:
            return 1.0
        recent = valid[-window:]
        level = float(np.median(recent))
        score = np.clip(level / threshold, 0.0, 1.0) if threshold > 0 else 0.0
        return float(1.0 - score)
