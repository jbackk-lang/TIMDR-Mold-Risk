"""
test_demo_scenarios.py — testy na poziomie scenariuszy (kontrole
pozytywne/negatywne + uczciwie udokumentowane ograniczenia).

WYNIKI DIAGNOSTYCZNE (uruchomione PRZED napisaniem progów tego pliku,
zeby ustalic realistyczne, nie zgadywane z gory, oczekiwania - patrz
tez PROTOKÓL w skillu timdr-signal-framework, punkt "pierwszy przebieg
kalibracyjny to sprawdzenie poprawnosci implementacji, nie tuning
progu"):

  normal_ventilated_room:      M pozostaje 0.0 przez cale 60 dni (czysta
                                kontrola negatywna)
  poor_ventilation_bathroom:   M koncowe tylko 0.083 (NIE przekracza
                                nawet indeksu 1) po 45 dniach codziennych
                                prysznicow - ale trend() wykrywa
                                statystycznie istotny rosnacy trend
                                (max|trend_z|=7.3) mimo niskiego
                                bezwzglednego poziomu ryzyka. UCZCIWY
                                WNIOSEK: krotkie, powtarzajace sie
                                epizody wysokiej wilgotnosci SAME W
                                SOBIE, nawet bez wentylacji, nie
                                generuja w tym modelu duzego ryzyka w
                                ciagu 45 dni - ale wczesny, rosnacy
                                trend jest wykrywalny zanim ryzyko
                                stanie sie powazne (to jest raportowane
                                jako wynik, NIE dostosowywane przez
                                zmiane parametrow scenariusza).
  monsoon_humidity_onset:      M rosnie do 2.89 (blisko, ale NIE
                                przekracza progu widocznego wzrostu 3.0)
                                w ciagu 120 dni. trend() daje SLABY
                                sygnal (max|trend_z|=1.6, ponizej
                                typowego progu istotnosci ~3) - model
                                NIE flaguje wyraznie powolnego,
                                sezonowego narastania wilgotnosci jako
                                trendu. twist() dal WYSOCE PODEJRZANY
                                wynik (1436/2880 punktow oznaczonych,
                                z max z~3798) - zdiagnozowane jako
                                niestabilnosc wspoldzielonego _mad_z:
                                d2M/dt2 jest DOKLADNIE 0 na DWOCH
                                plateau (~50% probek razem) - przed
                                sezonem (M=0) ORAZ PO nim (M plateau na
                                Mmax, bo k2=0 zatrzymuje dalszy wzrost) -
                                MAD kolapsuje do wartosci ~500x mniejszej
                                niz odchylenie standardowe calej serii
                                (zweryfikowano: MAD=6.4e-7 vs std=3.4e-4),
                                co nadmuchuje z-score kazdego, nawet
                                drobnego, odchylenia od plateau - patrz
                                test_monsoon_onset_twist_is_unreliable_
                                known_limitation nizej. NIE naprawiono
                                cicho przez zmiane progu - udokumentowano
                                jako ograniczenie.
  water_leak_sudden:           Czysta kontrola pozytywna - anomalies()
                                (267/1200 pkt, max z=6.0) i twist()
                                (205/1200 pkt, max z=13.8) oba wykrywaja
                                silnie, M przekracza prog 3.0 dnia ~41.7
                                (21.7 dnia po starcie wycieku dnia 20).
"""

import numpy as np
import pytest

from demo_scenarios import make_demo_data
from timdr_mold_fusion import TIMDRMoldFusion
from timdr_mold_predict import TIMDRMoldPredict


@pytest.fixture
def fusion():
    return TIMDRMoldFusion()


@pytest.fixture
def predict():
    return TIMDRMoldPredict()


def test_all_scenarios_generate_valid_data():
    for name in ["normal_ventilated_room", "poor_ventilation_bathroom",
                 "monsoon_humidity_onset", "water_leak_sudden"]:
        t, s = make_demo_data(name)
        assert len(t) > 0
        assert np.all(np.isfinite(s["temperature"]))
        assert np.all(np.isfinite(s["humidity"]))
        assert np.all(s["humidity"] >= 0) and np.all(s["humidity"] <= 100)


def test_normal_ventilated_room_stays_healthy(fusion):
    t, s = make_demo_data("normal_ventilated_room")
    M, margin = fusion.fuse(t, s["temperature"], s["humidity"])
    assert np.max(M) < 0.1, f"M powinno pozostac bliskie 0, max={np.max(M)}"
    idx_an, _ = fusion.anomalies(M)
    assert len(idx_an) == 0


def test_water_leak_detected_by_anomalies_and_twist(fusion):
    """KONTROLA POZYTYWNA - jednoznaczna."""
    t, s = make_demo_data("water_leak_sudden")
    M, margin = fusion.fuse(t, s["temperature"], s["humidity"])
    idx_an, z_an = fusion.anomalies(M)
    idx_tw, z_tw = fusion.twist(t, M)
    assert len(idx_an) > 50, "oczekiwano silnej detekcji anomalii po wycieku"
    assert len(idx_tw) > 20, "oczekiwano silnej detekcji skretu po wycieku"
    assert np.nanmax(z_an) > 3.0
    assert np.nanmax(z_tw) > 3.5


def test_water_leak_crosses_visible_threshold_within_expected_window(fusion):
    t, s = make_demo_data("water_leak_sudden", leak_day=20)
    M, margin = fusion.fuse(t, s["temperature"], s["humidity"])
    over3 = np.where(M >= 3.0)[0]
    assert len(over3) > 0
    day_crossed = t[over3[0]] / 24.0
    # wyciek zaczyna sie dnia 20, przy ~96% RH/23C oczekujemy przekroczenia
    # progu 3.0 w ciagu okolo 2-4 tygodni od startu (patrz kalibracja w
    # test_timdr_mold_fusion.py) => okno 20+10 do 20+35 dni
    assert 30.0 <= day_crossed <= 55.0, f"przekroczenie dnia {day_crossed:.1f}, oczekiwano 30-55"


def test_water_leak_time_to_visible_growth_shrinks_after_onset(fusion, predict):
    """Zaraz po wycieku (przed przekroczeniem progu), time_to_visible_growth
    powinno byc SKONCZONE (M rosnie) i maleć w miarę upływu czasu."""
    t, s = make_demo_data("water_leak_sudden", leak_day=20, n_days=50)
    M, margin = fusion.fuse(t, s["temperature"], s["humidity"])

    idx_25d = int(25 * 24)
    idx_35d = int(35 * 24)
    ttg_25 = predict.time_to_visible_growth(t[:idx_25d], M[:idx_25d])
    ttg_35 = predict.time_to_visible_growth(t[:idx_35d], M[:idx_35d])
    assert ttg_25 is not None and np.isfinite(ttg_25)
    assert ttg_35 is not None
    if np.isfinite(ttg_35) and np.isfinite(ttg_25):
        assert ttg_35 <= ttg_25, "czas do widocznego wzrostu powinien sie skracac w miare narastania M"


def test_poor_ventilation_shows_rising_trend_despite_low_absolute_risk(fusion):
    """Uczciwy, nuansowany wynik (patrz naglowek pliku) - NIE kontrola
    pozytywna w sensie 'wysokie M', tylko 'wykrywalny wczesny trend'."""
    t, s = make_demo_data("poor_ventilation_bathroom")
    M, margin = fusion.fuse(t, s["temperature"], s["humidity"])
    assert np.max(M) < 1.0, "ten scenariusz NIE powinien osiagnac nawet indeksu 1 w 45 dni"
    sl, z_tr = fusion.trend(t, M, window=168)
    assert np.nanmax(z_tr) > 3.0, "mimo niskiego bezwzglednego M, trend powinien byc statystycznie wykrywalny"


def test_monsoon_onset_twist_is_unreliable_known_limitation(fusion):
    """UDOKUMENTOWANE OGRANICZENIE (nie ukryte, nie naprawione cicho): dla
    sygnalu M(t) plaskiego (dokladnie 0) przez >=~50% okna, wspoldzielony
    fallback _mad_z (mad==0 -> normalizacja przez span/4) staje sie
    niestabilny i oznacza NIEPROPORCJONALNIE duzy odsetek probek jako
    'skret'. Ten test REJESTRUJE ten fakt jako regres (assert na
    faktycznie zaobserwowane zachowanie), nie jako pozadana wlasciwosc -
    patrz naglowek pliku i timdr_mold_fusion.py. Wniosek dla uzytkownika
    API: NIE ufaj twist() dla scenariuszy z dlugimi plaskimi odcinkami
    M(t)=0 (typowe dla powolnych/sezonowych narastan wilgotnosci) -
    uzywaj trend()/health_score() zamiast tego."""
    t, s = make_demo_data("monsoon_humidity_onset")
    M, margin = fusion.fuse(t, s["temperature"], s["humidity"])
    dE = np.gradient(M, t)
    ddE = np.gradient(dE, t)
    # frakcja PLASKA obejmuje DWA plateau: przed monsunem (M=0) I po nim
    # (M plateau na Mmax, bo k2=0 zatrzymuje wzrost) - stad liczymy po
    # ddE==0, nie po M==0 (ktore lapie tylko pierwsze plateau, ~25%)
    frac_flat_ddE = np.mean(ddE == 0.0)
    idx_tw, z_tw = fusion.twist(t, M)
    assert frac_flat_ddE >= 0.4, (
        "ten test zaklada scenariusz z dwoma plaskimi odcinkami d2M/dt2=0 "
        "(przed i po monsunie) - jesli to sie zmienilo, usun/zaktualizuj test"
    )
    # Udokumentowane (nie oczekiwane jako "dobre"): duzy odsetek falszywych
    # detekcji przy plaskim sygnale. Jesli ten test zacznie failowac bo
    # liczba detekcji SPADNIE ponizej tego progu, to znaczy ze ktos
    # naprawil _mad_z - wtedy NALEZY zaktualizowac ten test i naglowek
    # pliku, nie przywracac starego zachowania.
    assert len(idx_tw) > frac_flat_ddE * len(t) * 0.5, (
        "oczekiwano nadal obecnosci znanej niestabilnosci - jesli test "
        "faliuje bo detekcji jest MNIEJ, to dobra wiadomosc: _mad_z zostal "
        "naprawiony, zaktualizuj dokumentacje zamiast przywracac ten prog"
    )


def test_health_score_and_predict_consistency(fusion, predict):
    t, s = make_demo_data("normal_ventilated_room")
    M, margin = fusion.fuse(t, s["temperature"], s["humidity"])
    health = predict.health_score(M, threshold=3.0, window=48)
    assert health > 0.95
    ttg = predict.time_to_visible_growth(t, M, threshold=3.0)
    assert ttg == np.inf or ttg is None or ttg > 365 * 5
