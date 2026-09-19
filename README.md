# TIMDR-Mold-Risk

Wczesne ostrzeganie przed ryzykiem rozwoju pleśni w pomieszczeniach — na
tym samym operatorze TIMDR (fuse → twist/trend/anomalies/rhythm →
fusion_score), którego używają `TIMDR-Battery-Predict`,
`TIMDR-Industrial-Predict` i `TIMDR-Solar-PV`, więc **dokłada się do
reszty ekosystemu bez zmiany interfejsu wywołania** (patrz sekcja
"Podłączalność" niżej).

Geneza: druga z dwóch nowych domen wybranych po przeglądzie internetu
(research pokazał, że monitoring wilgoci/pleśni w budynkach jest
prawdziwą, niezaspokojoną potrzebą — tanie czujniki T/RH + model
predykcyjny mogłyby ostrzec ZANIM pleśń stanie się widoczna, zamiast
wykrywać ją po fakcie).

## Instalacja

```bash
pip install -r requirements.txt
```

## Szybki start

```bash
python -c "
from demo_scenarios import make_demo_data
from timdr_mold_fusion import TIMDRMoldFusion
from timdr_mold_predict import TIMDRMoldPredict

t, s = make_demo_data('water_leak_sudden')
fusion = TIMDRMoldFusion()
M, margin = fusion.fuse(t, s['temperature'], s['humidity'])

predict = TIMDRMoldPredict()
print('godziny do widocznego wzrostu:', predict.time_to_visible_growth(t, M))
"

python api.py       # REST API na http://127.0.0.1:5002
pytest -q           # 15 testow (7 jednostkowych rownania VTT + 8 scenariuszy)
```

## Model: VTT Mold Growth Model (Hukka & Viitanen 1999 / Viitanen 2004)

Zamiast fuzji ad-hoc MAD (jak w `TIMDR-Battery-Predict`, gdzie nie ma
prostego modelu fizycznego łączącego napięcie/prąd/temperaturę), tu
użyto **ugruntowanego w literaturze budowlanej** modelu różniczkowego
wzrostu pleśni:

```
dM/dt = k1(M) * k2(M) / f(T,phi),  t w godzinach, M(0)=0, M w [0,6]

f(T,phi) = 168 * exp(-0.68*ln(T) - 13.9*ln(phi) + 66.02)
k1(M) = k11 (M<1) albo k12 (M>=1)
k2(M) = max(1 - exp(2.3*(M - Mmax(phi))), 0)
Mmax(phi) = A + B*((phi_c-phi)/(phi_c-100)) + C*((phi_c-phi)/(phi_c-100))^2
```

Źródło dokładnych równań i tabeli klas wrażliwości materiału:
[Berger, Le Meur, Dutykh, Nguyen, Grillet — "Analysis and improvement of
the VTT mold growth model: application to bamboo fiberboard"
(arXiv:1804.02992)](https://arxiv.org/pdf/1804.02992), sekcja 2.1,
cytujący pierwotnie Hukka & Viitanen (1999, *Wood Science and
Technology* 33(6):475-485) i Viitanen (2004). Krzywa graniczna
`phi_crit(T)` (poniżej której wzrost jest fizycznie niemożliwy) z
[Viitanen & Ojanen 2007, ASHRAE Buildings X](https://web.ornl.gov/sci/buildings/conf-archive/2007%20B10%20papers/162_Viitanen.pdf) —
**UWAGA: to źródło miało zdegradowaną ekstrakcję tekstu z PDF (zgubiony
znak), transkrypcja została poprawiona i zweryfikowana numerycznie
(warunek ciągłości przy 20°C) przed użyciem — patrz komentarz w
`timdr_mold_fusion.py` i `test_rhcrit_continuous_at_20_degrees`.**

E(t)=M(t) jest podawane dalej do DOKŁADNIE tych samych operatorów
`twist/trend/anomalies/rhythm` co w Battery-Predict/Solar-PV — te same
wagi w `fusion_score()`, celowo nie przetunowane pod tę domenę.

## Metodologia i uczciwe ograniczenia

**Kontrola negatywna (zweryfikowana empirycznie):** przy stałej
wilgotności 50% (poniżej dowolnego `phi_crit` dla typowych temperatur
pokojowych), M pozostaje dokładnie 0 — to bezpośrednia konsekwencja
poprawności równania (`k2=0` gdy `M > Mmax(phi)` ujemne), nie kalibracja.
Scenariusz `normal_ventilated_room` (RH 40-60%) potwierdza to samo w
warunkach zmiennych w czasie — `max(M)=0.0` przez 60 dni.

**Kontrola pozytywna (zweryfikowana empirycznie):** przy stałych 97%
RH/23°C, klasa `bardzo_wrazliwy`, M przekracza próg widocznego wzrostu
(indeks 3) między 2. a 4. tygodniem — rząd wielkości zgodny z opisem w
Viitanen & Ojanen (2007): "fungal growth began in particle board after
one, three, and seven weeks... depending on humidity and temperature".
Scenariusz `water_leak_sudden` (nagły, trwały skok RH do ~96%) daje
jednoznaczną detekcję: `anomalies()` (267/1200 pkt, max z=6.0) i
`twist()` (205/1200 pkt, max z=13.8) oba wykrywają silnie, M przekracza
próg 3.0 ok. 21.7 dnia po starcie wycieku.

**Uczciwy wynik negatywny/nuansowany (nie ukryty):** scenariusz
`poor_ventilation_bathroom` (codzienne prysznice bez wentylacji, wolny
rozpad RH) daje M końcowe zaledwie **0.083** po 45 dniach — model mówi,
że krótkie, powtarzające się epizody wysokiej wilgotności, SAME W
SOBIE, nie generują dużego ryzyka w tej skali czasu. Mimo to `trend()`
wykrywa statystycznie istotny rosnący trend (`max|trend_z|=7.3`) —
wczesne ostrzeżenie widoczne w trendzie, zanim bezwzględny poziom
ryzyka stanie się poważny.

**Znane ograniczenie/bug w `twist()` dla wolno narastających scenariuszy
(udokumentowane, NIE naprawione cichaczem):** scenariusz
`monsoon_humidity_onset` (powolne, sezonowe narastanie RH) ujawnił
niestabilność współdzielonego fallbacku `_mad_z` — gdy druga pochodna
`M(t)` jest dokładnie 0 na ~50% próbek (dwa plateau: przed sezonem przy
M=0 ORAZ po nim, gdy `k2=0` zatrzymuje dalszy wzrost), MAD kolapsuje do
wartości ~500× mniejszej niż odchylenie standardowe całej serii, co
nadmuchuje z-score każdego, nawet drobnego, odchylenia od plateau —
`twist()` oznaczył wtedy 1436/2880 punktów (powinno być garstka).
`trend()` też dał tu słaby sygnał (`max|trend_z|=1.6`, poniżej progu
istotności). **Wniosek dla użytkownika API: nie ufaj `twist()` dla
scenariuszy z długimi płaskimi odcinkami `M(t)` (typowe dla
powolnych/sezonowych narastań wilgotności) — używaj `trend()` i
`health_score()`/`time_to_visible_growth()`.** Test
`test_monsoon_onset_twist_is_unreliable_known_limitation` dokumentuje to
wprost jako regres na zaobserwowane zachowanie, nie jako pożądaną
własność.

**Model NIE ma regresji/zaniku ryzyka w okresach suchych** — świadomie
NIE zaimplementowano tego rozszerzenia (źródłowy PDF miał zdegradowaną
ekstrakcję równania dla okresu suchego, ryzyko błędnej transkrypcji —
nie zgadywano). M(t) jest więc monotonicznie NIEmalejące (poza
naturalnym zatrzymaniem wzrostu przez `k2=0`) — model PRZESZACOWUJE
ryzyko przy naprzemiennych mokro/sucho warunkach względem pełnego modelu
VTT z regresją. Jawnie udokumentowane w `timdr_mold_fusion.py`.

**Krytyka samego modelu VTT (nie ukryta):** artykuł źródłowy dokładnych
równań (arXiv:1804.02992) jest w istocie krytyką modelu VTT — jego
abstrakt wprost stwierdza, że estymacja parametrów z danych
eksperymentalnych na płycie bambusowej nie była satysfakcjonująca i
"mathematical formulation of the physical model of mold growth is not
reliable", proponując ulepszony model logistyczny. Ten moduł używa
oryginalnego VTT (nie ulepszonego) — szerzej cytowanego historycznie
(Vereecken & Roels 2012, przegląd modeli pleśni), ale to jest jawne
ograniczenie wyboru, nie ukryte założenie o nieomylności modelu.

**Walidacja na PRAWDZIWYCH danych DALTON: NIE WYKONANA w tym
środowisku.** Dataset (30 miejsc, Indie, 89.1M próbek, kolumny T/RH
zweryfikowane z README repozytorium) jest publicznie dostępny na
GitHubie (`github.com/prasenjit52282/dalton-dataset`) i sandbox miał
dostęp do samego repo (sparse-checkout się udał) — ale rzeczywiste
pliki CSV są przechowywane przez Git LFS, a hosty binarne LFS
(`media.githubusercontent.com` i podobne) są poza allowlistą sieciową
tego środowiska, a `git-lfs` nie dało się zainstalować bez sudo. Schemat
kolumn w `real_dalton_test.py` jest więc DOKŁADNIE zweryfikowany (nie
zgadywany wzorcami jak w PVDAQ dla `TIMDR-Solar-PV`), ale sam kod
ingestii nie został przetestowany na rzeczywistych wartościach liczbowych.
Instrukcja pobrania (z `git lfs pull`) i uruchomienia samodzielnie: patrz
nagłówek `real_dalton_test.py`.

## Podłączalność do reszty ekosystemu

```python
fuse(t, temperature, humidity) -> (E, margin)   # E=M (indeks 0-6), margin=RH-RHcrit(T)
twist(t, E) -> (idx, z)
trend(t, E, window) -> (slopes, z)
anomalies(E) -> (idx, z)
rhythm(E) -> (periods, score)
fusion_score(twist_z, trend_z, anomaly_z, rhythm_score) -> float
```

Jedyna rzecz specyficzna dla domeny to argumenty `fuse()`
(`temperature, humidity`, czas w GODZINACH — nie dniach jak w
Solar-PV) i orientacja `E` (tu: M, wyżej=gorzej wprost z modelu, bez
przekształcenia `1-x` jak w PV). Żeby użyć tego kodu w innym repo:
skopiuj `timdr_mold_fusion.py` + `timdr_mold_predict.py` z nagłówkiem
"ZWENDOROWANE" (ten sam wzorzec co reszta ekosystemu).

## Ograniczenia (zwięźle)

- Brak regresji ryzyka w okresach suchych (patrz wyżej) — model
  konserwatywnie zakłada trwałość ryzyka.
- `twist()` niewiarygodny dla scenariuszy z długimi płaskimi odcinkami
  `M(t)` (patrz wyżej) — używaj `trend()`/`health_score()`.
- Model zakłada JEDNĄ stałą klasę wrażliwości materiału na cały
  przebieg, mierzoną na podstawie RH POWIETRZA (czujnik ambientowy), nie
  RH POWIERZCHNI (zwykle wyższe przy przegrodach zewnętrznych — punkt
  rosy) — przybliżenie z góry (raczej niedoszacowanie realnego ryzyka
  na chłodnych powierzchniach niż przeszacowanie).
- `CO2`/`VOC` (dostępne w DALTON) są tylko diagnostycznym proxy słabej
  wentylacji (`ventilation_deficit_proxy()`, próg ASHRAE 62.1 ~1000ppm)
  — świadomie NIE wchodzą ilościowo do `mold_index_series()` (brak
  ugruntowanego w literaturze mostu CO2→tempo wzrostu pleśni, nie
  zgadywano współczynnika).
- Brak walidacji na realnych danych (patrz wyżej) — priorytet numer 1
  do zrobienia przed jakimkolwiek użyciem produkcyjnym.
- Dashboard: patrz sekcja "Dashboard" niżej — parser BLE nieprzetestowany
  na prawdziwym sprzęcie w tym środowisku (patrz sekcja "Czujnik
  Bluetooth").

## Dashboard

`dashboard.html` — panel w przeglądarce (serwowany przez `api.py` pod
`/dashboard`, ten sam Flask, więc bez CORS): zakładki dla 4 scenariuszy
demo (`demo_scenarios.py`) + zakładka "Czujnik na żywo (BLE)". Każda
zakładka pokazuje: karty statusu (poziom ryzyka, health score, czas do
widocznego wzrostu, margines RH-RHcrit), wykres `M(t)` z linią progu i
zaznaczonymi anomaliami, wykres surowych czujników (temperatura/
wilgotność), oraz panel diagnostyki operatora TIMDR
(`twist`/`trend`/`anomalies`/`rhythm`/`fusion_score`) — wszystkie liczby
pochodzą wprost z `/api/analyze`, nic nie jest liczone po stronie
przeglądarki.

Uruchomienie: `run.bat` (Windows) — instaluje zależności z
`requirements.txt`, startuje `api.py` w osobnym oknie konsoli i otwiera
`http://127.0.0.1:5002/dashboard` w domyślnej przeglądarce. Ręcznie:
`python api.py`, potem otwórz ten URL sam.

## Czujnik Bluetooth (Xiaomi Mijia LYWSD03MMC)

Zakładka "Czujnik na żywo" w dashboardzie czyta dane z prawdziwego
czujnika T/RH przez BLE (`ble_sensor.py`, biblioteka `bleak`) —
`POST /api/ble/start` (opcjonalnie z `{"mac": "AA:BB:..."}` żeby
filtrować do jednego urządzenia) uruchamia skanowanie w tle,
`GET /api/ble/live` zwraca bufor odczytów w tym samym kształcie co
`/api/demo`, więc dashboard analizuje je dokładnie tym samym
`/api/analyze` co scenariusze syntetyczne.

**Wymaganie sprzętowe:** sensor musi mieć wgrany **custom firmware**
([pvvx/ATC_MiThermometer](https://github.com/pvvx/ATC_MiThermometer)),
NIE fabryczny firmware Xiaomi — ten drugi szyfruje dane protokołem
MiBeacon i wymaga "bindkey" z aplikacji Xiaomi Home, czego ten moduł
świadomie nie obsługuje (zbyt zawodne do zaimplementowania bez dostępu
do prawdziwego urządzenia i klucza). Custom firmware nadaje dane jawnie
w Service Data pod UUID `0x181A`.

**UWAGA UCZCIWOŚCI: parser (`ble_sensor.py::parse_atc1441`/`parse_pvvx`)
NIE został zweryfikowany na prawdziwym sprzęcie w tym środowisku** —
sandbox, w którym powstał, nie ma fizycznego adaptera Bluetooth. Układ
bajtów odtworzono z pamięci na podstawie publicznie znanego formatu
firmware pvvx/atc1441; sam parser ma testy jednostkowe na ręcznie
skonstruowanych bajtach (sprawdzają tylko, że kod poprawnie odczytuje
bajty W UKŁADZIE, KTÓRY SAM ZAŁOŻYŁEM — nie że ten układ zgadza się z
prawdziwym sensorem). Przed zaufaniem odczytom na Twoim sprzęcie:

```bash
python ble_sensor.py --scan --debug
```

wypisze surowe bajty każdej odebranej reklamy oraz to, co z nich zostało
odparsowane — porównaj z wyświetlaczem samego czujnika lub apką typu
nRF Connect. Jeśli się nie zgadza, popraw `parse_atc1441`/`parse_pvvx` w
`ble_sensor.py` (najbardziej prawdopodobny błąd: kolejność bajtów MAC
albo big/little-endian liczb) i zostaw komentarz, która wersja została
faktycznie zweryfikowana na sprzęcie i kiedy — ten sam wzorzec uczciwości
co reszta tego repo (patrz sekcja "Walidacja na PRAWDZIWYCH danych
DALTON" wyżej).

## Struktura

```
TIMDR-Mold-Risk/
├── timdr_mold_fusion.py       — operator: fuse/twist/trend/anomalies/rhythm/fusion_score (model VTT)
├── timdr_mold_predict.py      — time_to_visible_growth/health_score/risk_level_label
├── demo_scenarios.py          — 4 syntetyczne scenariusze (kontekst: mieszkania indyjskie, monsun)
├── real_dalton_test.py        — walidacja na realnych danych DALTON (do uruchomienia przez usera)
├── test_timdr_mold_fusion.py  — 7 testow jednostkowych rownania VTT
├── test_demo_scenarios.py     — 8 testow scenariuszy (kontrole pozytywne/negatywne + ograniczenia)
├── api.py                     — REST API (Flask)
├── requirements.txt, LICENSE, .gitignore
```
