"""
test_timdr_mold_fusion.py — testy JEDNOSTKOWE rownania VTT (nie
scenariuszy - te sa w test_demo_scenarios.py).

PRE-REJESTRACJA (ustalona PRZED uruchomieniem, na podstawie opisu w
zrodlowej literaturze - Viitanen & Ojanen 2007 ASHRAE: "fungal growth
began in particle board after one, three, and seven weeks... depending
on... humidity and temperature"):

- Przy stalych warunkach 97% RH / 23 degC, klasa 'bardzo_wrazliwy',
  indeks powinien przekroczyc próg WIDOCZNEGO wzrostu (M>=3) gdzies
  miedzy 2. a 4. tygodniem (14-28 dni) - zgodne z rzedem wielkosci z
  literatury dla materialow podatnych w wysokiej wilgotnosci, NIE
  dokladna replikacja (inny material/warunki eksperymentu).
- Przy stalych warunkach 50% RH (ponizej jakiegokolwiek phi_crit dla
  typowych temperatur pokojowych), M MUSI pozostac dokladnie 0 przez
  caly czas - to bezposrednia konsekwencja k2=0 gdy M(0)=0 > Mmax(phi)
  ujemne, nie empiryczny wynik do "sprawdzenia" - to test poprawnosci
  IMPLEMENTACJI rownania, nie kalibracji.
- RHcrit(T=20)~RHcrit(T=20+) musi byc ciagle (obie galezie wzoru licza
  ~80% przy T=20) - to byl blad transkrypcji z pierwszego zdegradowanego
  zrodla PDF (ASHRAE Viitanen&Ojanen 2007), poprawiony i zweryfikowany
  numerycznie przed napisaniem tego testu (patrz timdr_mold_fusion.py,
  komentarz przy phi_crit).
"""

import numpy as np
import pytest

from timdr_mold_fusion import (
    TIMDRMoldFusion,
    mold_index_series,
    VULNERABILITY_CLASSES,
)


def test_rhcrit_continuous_at_20_degrees():
    """Warunek ciaglosci krzywej granicznej VTT - MUSI dawac ~80% po
    obu stronach T=20 (patrz komentarz w fuse())."""
    fusion = TIMDRMoldFusion()
    t = np.array([0.0, 1.0])
    T = np.array([20.0, 20.0])
    RH = np.array([80.0, 80.0])
    _, margin = fusion.fuse(t, T, RH)
    # margin = RH - RHcrit(T); przy T=20 RHcrit powinno byc bardzo blisko 80
    assert abs(margin[0]) < 1.0, f"RHcrit(20) powinno byc ~80%, margin={margin[0]}"


def test_rhcrit_decreases_from_100_at_0_degrees_to_80_at_20():
    fusion = TIMDRMoldFusion()
    t = np.array([0.0, 1.0, 2.0])
    T = np.array([0.0, 10.0, 20.0])
    RH = np.array([0.0, 0.0, 0.0])
    _, margin = fusion.fuse(t, T, RH)
    rhcrit = -margin  # margin = RH-RHcrit, RH=0 wiec -margin=RHcrit
    assert rhcrit[0] == pytest.approx(100.0, abs=0.5)
    assert rhcrit[2] == pytest.approx(80.0, abs=1.0)
    assert rhcrit[0] > rhcrit[1] > rhcrit[2], "RHcrit powinno monotonicznie malec z temperatura"


def test_negative_control_low_humidity_gives_exactly_zero():
    """50% RH, typowa temperatura pokojowa - PONIZEJ jakiegokolwiek
    phi_crit - M MUSI zostac dokladnie 0 (test poprawnosci k2, nie
    kalibracji progu)."""
    n = 24 * 30
    t = np.arange(n, dtype=float)
    T = np.full(n, 22.0)
    RH = np.full(n, 50.0)
    M = mold_index_series(t, T, RH, "bardzo_wrazliwy")
    assert np.all(M == 0.0)


def test_positive_control_high_humidity_crosses_visible_threshold_in_expected_window():
    """Stale 97% RH / 23 degC, 'bardzo_wrazliwy' - M powinno przekroczyc
    3.0 (widoczny wzrost) miedzy dniem 14 a 28 (2.-4. tydzien) - patrz
    pre-rejestracja w naglowku pliku."""
    n_days = 40
    n = 24 * n_days
    t = np.arange(n, dtype=float)
    T = np.full(n, 23.0)
    RH = np.full(n, 97.0)
    M = mold_index_series(t, T, RH, "bardzo_wrazliwy")
    over3 = np.where(M >= 3.0)[0]
    assert len(over3) > 0, "M nigdy nie przekroczylo 3.0 w 40 dniach - niezgodne z literatura"
    day_crossed = t[over3[0]] / 24.0
    assert 14.0 <= day_crossed <= 28.0, (
        f"M przekroczylo 3.0 dnia {day_crossed:.1f} - oczekiwano w oknie 14-28 dni "
        f"(2.-4. tydzien, zgodnie z rzedem wielkosci z Viitanen&Ojanen 2007)"
    )


def test_mold_index_is_monotonically_nondecreasing_under_constant_conditions():
    """Udokumentowane ograniczenie (patrz naglowek timdr_mold_fusion.py):
    model NIE ma regresji w suchych okresach - pod stalymi warunkami M
    MUSI byc niemalejace (albo rosnie, albo plateau przy Mmax, nigdy nie
    spada)."""
    n = 24 * 60
    t = np.arange(n, dtype=float)
    T = np.full(n, 23.0)
    RH = np.full(n, 90.0)
    M = mold_index_series(t, T, RH, "bardzo_wrazliwy")
    assert np.all(np.diff(M) >= -1e-9)


def test_vulnerable_classes_ordered_by_growth_speed():
    """Przy tych samych warunkach, 'bardzo_wrazliwy' powinien rosnac
    szybciej niz 'wrazliwy', szybciej niz 'srednio_odporny', szybciej
    niz 'odporny' - kolejnosc z definicji klas (Tabela 1 zrodla), nie
    zalozenie do "sprawdzenia" per se, ale test ze parametry z tabeli
    zostaly przepisane w poprawnej kolejnosci/bez zamiany wierszy."""
    n = 24 * 30
    t = np.arange(n, dtype=float)
    T = np.full(n, 23.0)
    RH = np.full(n, 95.0)
    finals = {}
    for cls in ["bardzo_wrazliwy", "wrazliwy", "srednio_odporny", "odporny"]:
        M = mold_index_series(t, T, RH, cls)
        finals[cls] = M[-1]
    assert finals["bardzo_wrazliwy"] > finals["wrazliwy"]
    assert finals["wrazliwy"] > finals["srednio_odporny"]
    assert finals["srednio_odporny"] > finals["odporny"]


def test_unknown_vulnerability_class_raises():
    with pytest.raises(KeyError):
        mold_index_series(np.array([0.0, 1.0]), np.array([20.0, 20.0]),
                           np.array([90.0, 90.0]), "nieistniejaca_klasa")
