"""
timdr_mold_fusion.py — TIMDR Mold Risk Fusion
=================================================
Fuzja temperatury i wilgotności względnej w indeks ryzyka pleśni M(t) wg
modelu VTT (Hukka & Viitanen 1999, rozszerzony Viitanen 2004) — DOKŁADNIE
tej samej rodziny "jeden fizyczny model -> E(t) -> standardowy zestaw
detektorów TIMDR" co `TIMDR-Solar-PV` (tam: PVWatts -> PR -> E=1-PR).

Różnica względem PV: tu E(t)=M(t) NIE jest przekształcane (0=brak
ryzyka, 6=pełne porośnięcie) — model VTT już zwraca wielkość w
orientacji "wyżej = gorzej", więc E=M wprost.

## Model VTT — dokładne równania (źródło: Berger, Le Meur, Dutykh, Nguyen,
## Grillet, "Analysis and improvement of the VTT mold growth model:
## application to bamboo fiberboard", arXiv:1804.02992, sekcja 2.1,
## cytujący pierwotnie Hukka & Viitanen 1999 i Viitanen 2004)

    dM/dt = k1(M) * k2(M) / f(T,phi),   t w GODZINACH, M(0)=0

    f(T,phi) = b0 * exp(b1*ln(T) + b2*ln(phi) + b3)
    b0=168, b1=-0.68, b2=-13.9, b3=66.02

    k1(M) = k11 gdy M<1, k12 gdy M>=1

    k2(M) = max(1 - exp(2.3*(M - Mmax(phi))), 0)

    Mmax(phi) = A + B*((phi_c-phi)/(phi_c-100)) + C*((phi_c-phi)/(phi_c-100))^2

Klasy wrażliwości materiału (k11,k12,A,B,C,phi_c) z tej samej pracy,
Tabela 1 (pierwotnie Hukka&Viitanen 1999 / Viitanen 2004):

    bardzo_wrazliwy:    k11=1,     k12=2,     A=1,   B=7, C=-2,   phi_c=80
    wrazliwy:           k11=0.578, k12=0.386, A=0.3, B=6, C=-1,   phi_c=80
    srednio_odporny:    k11=0.072, k12=0.097, A=0,   B=5, C=-1.5, phi_c=85
    odporny:            k11=0.033, k12=0.014, A=0,   B=3, C=-1,   phi_c=85

Domyślna klasa w tym module: `bardzo_wrazliwy` (najbardziej czuła/
konserwatywna) — uzasadnienie: to system WCZESNEGO OSTRZEGANIA, nie
model konkretnego materiału budowlanego; kurz/organiczne zanieczyszczenia
na dowolnej powierzchni (typowe w mieszkaniach) zachowują się bliżej tej
klasy niż "resistant" (patrz też ASHRAE Viitanen&Ojanen 2007: "pine
sapwood... use to represent the risk of mold growth on any material
surface containing organic compounds - dust, spores, soil").

## Uczciwe ograniczenia modelu (nie ukryte)

1. Ten sam artykuł źródłowy (arXiv:1804.02992) jest w istocie KRYTYKĄ
   modelu VTT — abstrakt wprost stwierdza, że "the VTT mathematical
   formulation... is not reliable" przy estymacji parametrów z danych
   eksperymentalnych na bambusowej płycie pilśniowej, i proponuje
   ulepszony model logistyczny. Używam tu oryginalnego VTT (nie
   ulepszonego), bo jest szerzej cytowany/zwalidowany historycznie
   (Vereecken & Roels 2012, przegląd modeli pleśni) i bo parametry klas
   wrażliwości (Tabela 1) pochodzą z tej samej, ugruntowanej linii prac
   (Hukka&Viitanen, Viitanen), NIE z nowego modelu logistycznego tego
   artykułu — ale to jest jawne ograniczenie, nie ukryte założenie.
2. NIE zaimplementowano rozszerzenia o zanik/regresję M w okresach
   suchych (Viitanen & Ojanen 2007, ASHRAE, rownanie dla "dry period" -
   tamten dokument miał zdegradowaną ekstrakcję tekstu z PDF, ryzyko
   błędnej transkrypcji współczynników; NIE zgadywano). W tym module
   M(t) jest wiec MONOTONICZNIE NIEMALEJĄCE (poza naturalnym
   zatrzymaniem wzrostu przez k2=0 gdy M>=Mmax(phi)) - konserwatywne
   uproszczenie: model NIE modeluje samoistnego "wysychania" ryzyka po
   okresie wilgoci, tylko jego zatrzymanie. To oznacza, że ten model
   będzie PRZESZACOWYWAĆ ryzyko przy naprzemiennych mokro/sucho
   warunkach względem pełnego modelu VTT z regresją - jawnie
   udokumentowane, nie domyślnie "bezpieczne".
3. Model zakłada STAŁĄ klasę materiału na cały przebieg — w realnym
   pomieszczeniu różne powierzchnie (drewno, gips, beton) mają różną
   wrażliwość; ten moduł daje JEDEN wspólny sygnał dla całego
   pomieszczenia (na podstawie czujnika T/RH powietrza, nie
   powierzchni), zgodnie z tym, co faktycznie mierzy tani czujnik
   ambientowy (jak w DALTON) — nie mierzy RH POWIERZCHNI, tylko RH
   powietrza w pomieszczeniu, co jest przybliżeniem z góry (zwykle RH
   przy chłodnej powierzchni jest WYŻSZE niż RH powietrza w rdzeniu
   pomieszczenia - efekt punktu rosy przy przegrodach zewnętrznych).
"""

import numpy as np

VULNERABILITY_CLASSES = {
    "bardzo_wrazliwy": dict(k11=1.0, k12=2.0, A=1.0, B=7.0, C=-2.0, phi_c=80.0),
    "wrazliwy": dict(k11=0.578, k12=0.386, A=0.3, B=6.0, C=-1.0, phi_c=80.0),
    "srednio_odporny": dict(k11=0.072, k12=0.097, A=0.0, B=5.0, C=-1.5, phi_c=85.0),
    "odporny": dict(k11=0.033, k12=0.014, A=0.0, B=3.0, C=-1.0, phi_c=85.0),
}

_B0, _B1, _B2, _B3 = 168.0, -0.68, -13.9, 66.02


def _f(T, phi):
    """f(T,phi) z Eq. 2.2 — T w stopniach C, phi w %. Zdefiniowane tylko
    dla T>0 i phi>0 (ln); poza tym zakresem model VTT nie jest
    zdefiniowany (patrz oryginalna praca: zakres 0-50 degC)."""
    T = np.maximum(T, 0.1)
    phi = np.maximum(phi, 0.1)
    return _B0 * np.exp(_B1 * np.log(T) + _B2 * np.log(phi) + _B3)


def _m_max(phi, A, B, C, phi_c):
    x = (phi_c - phi) / (phi_c - 100.0)
    return A + B * x + C * x**2


def mold_index_series(t_hours, temperature, humidity, vulnerability_class="bardzo_wrazliwy"):
    """Całkuje dM/dt = k1(M)*k2(M)/f(T,phi) metodą Eulera (krok wg
    roznic t_hours[i+1]-t_hours[i]) od M(0)=0. Zwraca tablice M(t) tej
    samej dlugosci co t_hours.

    UWAGA NUMERYCZNA: model zaklada wolno zmieniajace sie w czasie
    warunki (skala tygodni) - dla bardzo duzych krokow czasowych (>24h)
    lub gwaltownych skokow RH/T miedzy probkami, calkowanie Eulera moze
    byc niedokladne. Dla danych o rozdzielczosci godzinowej (typowe
    zastosowanie, w tym DALTON po agregacji) blad jest pomijalny -
    zweryfikowane w test_demo_scenarios.py przez porownanie z krokiem
    2x gestszym (test_euler_step_convergence)."""
    t = np.asarray(t_hours, float)
    T = np.asarray(temperature, float)
    phi = np.asarray(humidity, float)
    n = len(t)
    if n == 0:
        return np.array([])

    params = VULNERABILITY_CLASSES[vulnerability_class]
    k11, k12, A, B, C, phi_c = (params["k11"], params["k12"], params["A"],
                                  params["B"], params["C"], params["phi_c"])

    M = np.zeros(n)
    for i in range(1, n):
        dt = t[i] - t[i - 1]
        if dt <= 0:
            M[i] = M[i - 1]
            continue
        m_prev = M[i - 1]
        mmax = _m_max(phi[i - 1], A, B, C, phi_c)
        k1 = k11 if m_prev < 1.0 else k12
        k2 = max(1.0 - np.exp(2.3 * (m_prev - mmax)), 0.0)
        f_val = _f(T[i - 1], phi[i - 1])
        dM_dt = (k1 * k2) / f_val if f_val > 0 else 0.0
        M[i] = np.clip(m_prev + dM_dt * dt, 0.0, 6.0)
    return M


class TIMDRMoldFusion:
    """Ten sam operator co TIMDRSolarFusion/TIMDRBatteryFusion:
    fuse -> twist/trend/anomalies/rhythm -> fusion_score, na E(t)=M(t)."""

    def __init__(self, mad_scale=1.4826, vulnerability_class="bardzo_wrazliwy"):
        self.mad_scale = mad_scale
        self.vulnerability_class = vulnerability_class

    def _mad_z(self, x):
        x = np.asarray(x, float)
        if x.size == 0:
            return np.zeros_like(x)
        med = np.median(x)
        mad = np.median(np.abs(x - med)) * self.mad_scale
        if mad == 0:
            span = np.max(x) - np.min(x)
            if span == 0:
                return np.zeros_like(x)
            return (x - med) / (span / 4.0)
        return (x - med) / mad

    def fuse(self, t_hours, temperature, humidity):
        """Zwraca (E, margin) gdzie E=M(t) (indeks pleśni 0-6) i
        margin=phi(t)-phi_crit(T(t)) (dodatni = warunki SPRZYJAJĄCE
        wzrostowi w danej chwili, ujemny = niesprzyjające) - diagnostyka
        analogiczna do PR w TIMDR-Solar-PV."""
        t = np.asarray(t_hours, float)
        T = np.asarray(temperature, float)
        phi = np.asarray(humidity, float)
        M = mold_index_series(t, T, phi, self.vulnerability_class)

        # phi_crit(T) - klasyczna krzywa graniczna VTT (Viitanen & Ritschkoff
        # 1991 / Hukka & Viitanen 1997), cytowana niezaleznie od parametrow
        # klasy wrazliwosci (ta sama krzywa dla wszystkich klas materialu -
        # definiuje TYLKO gdzie wzrost jest w ogole mozliwy, nie jego tempo):
        # RHcrit(T) = -0.00267*T^3 + 0.160*T^2 - 3.13*T + 100.0 dla T<=20, 80% dla T>20.
        # UWAGA: pierwsze zrodlo (ASHRAE PDF, Viitanen&Ojanen 2007) mialo
        # zdegradowana ekstrakcje tekstu ze zgubionym znakiem przy pierwszych
        # dwoch wyrazach - zweryfikowano numerycznie przez warunek ciaglosci
        # (wzor MUSI dawac ~80% przy T=20, na styku z galezia T>20->80%);
        # ta wersja (ze znakiem -/+) spelnia ten warunek (80.04% przy T=20),
        # pierwotna transkrypcja (+/-) dawala -5.24% - wewnetrznie sprzeczne,
        # odrzucone.
        phi_crit = np.where(T <= 20.0, -0.00267 * T**3 + 0.160 * T**2 - 3.13 * T + 100.0, 80.0)
        margin = phi - phi_crit
        return M, margin

    def twist(self, t, E):
        t = np.asarray(t, float)
        E = np.asarray(E, float)
        if len(t) < 3:
            return np.array([], int), np.zeros_like(E)
        dE = np.gradient(E, t)
        ddE = np.gradient(dE, t)
        z = np.abs(self._mad_z(ddE))
        idx = np.where(z > 3.5)[0]
        return idx, z

    def trend(self, t, E, window=30):
        t = np.asarray(t, float)
        E = np.asarray(E, float)
        n = len(t)
        slopes = np.zeros_like(E)
        if n < 2:
            return slopes, np.zeros_like(slopes)
        for i in range(n):
            j0 = max(0, i - window + 1)
            tt = t[j0:i + 1]
            ee = E[j0:i + 1]
            A_ = np.column_stack([tt, np.ones_like(tt)])
            a, b = np.linalg.lstsq(A_, ee, rcond=None)[0]
            slopes[i] = a
        z = self._mad_z(slopes)
        return slopes, z

    def anomalies(self, E):
        E = np.asarray(E, float)
        if E.size == 0:
            return np.array([], int), np.zeros_like(E)
        z = np.abs(self._mad_z(E))
        idx = np.where(z > 3.0)[0]
        return idx, z

    def rhythm(self, E, max_lag=120, power_thresh=0.4):
        E = np.asarray(E, float)
        n = len(E)
        if n < 3:
            return [], 0.0
        t_idx = np.arange(n, dtype=float)
        slope, intercept = np.polyfit(t_idx, E, 1)
        E = E - (slope * t_idx + intercept)
        max_lag = min(max_lag, n - 1)
        ac = np.zeros(max_lag + 1)
        for lag in range(max_lag + 1):
            if lag == 0:
                ac[lag] = np.dot(E, E) / n
            else:
                overlap = n - lag
                if overlap <= 0:
                    break
                ac[lag] = np.dot(E[:-lag], E[lag:]) / overlap
        if ac[0] == 0:
            return [], 0.0
        ac /= ac[0]
        peaks = [
            (i, float(ac[i])) for i in range(1, len(ac) - 1)
            if ac[i] > ac[i - 1] and ac[i] > ac[i + 1] and ac[i] >= power_thresh
        ]
        if not peaks:
            return [], 0.0
        score = max(p for _, p in peaks)
        return [p for p, _ in peaks], score

    def fusion_score(self, twist_z, trend_z, anomaly_z, rhythm_score):
        def safe_max(x):
            x = np.asarray(x, float)
            return float(np.max(x)) if x.size else 0.0

        return float(
            0.4 * safe_max(twist_z) +
            0.3 * safe_max(trend_z) +
            0.2 * safe_max(anomaly_z) +
            0.1 * rhythm_score
        )
