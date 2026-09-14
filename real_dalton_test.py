"""
real_dalton_test.py — walidacja na PRAWDZIWYCH danych DALTON
================================================================================
UCZCIWE ZASTRZEŻENIE (przeczytaj przed użyciem): ten skrypt NIE ZOSTAŁ
URUCHOMIONY na prawdziwych danych DALTON w środowisku, w którym powstał.
Sandbox mial dostep do github.com (git protokol) - sam sparse-clone
repozytorium (github.com/prasenjit52282/dalton-dataset) sie udal, WIEC
DOKŁADNY SCHEMAT KOLUMN PONIŻEJ JEST ZWERYFIKOWANY (nie zgadywany jak przy
PVDAQ w TIMDR-Solar-PV) - ale same pliki CSV w folderze `Processed/` sa
przechowywane przez Git LFS (potwierdzone: kazdy plik to wskaznik LFS
"version https://git-lfs.github.com/spec/v1, oid sha256:..., size ..."),
a hosty binarne Git LFS (github-cloud.githubusercontent.com,
media.githubusercontent.com) sa poza allowlista tego sandboxa (403
blocked-by-allowlist) i `git-lfs` nie jest zainstalowany (brak uprawnien
sudo do apt-get install). REALNE WARTOSCI LICZBOWE W CSV NIGDY NIE ZOSTALY
ZOBACZONE w tej sesji - caly kod ponizej jest napisany na podstawie
oficjalnego README repozytorium (dokladny opis kolumn), nie na podstawie
przetestowania na realnym pliku.

## Jak pobrac dane (potrzebne: git + git-lfs, na Twoim komputerze)

```bash
git clone https://github.com/prasenjit52282/dalton-dataset.git
cd dalton-dataset
git lfs install
git lfs pull  # sciaga rzeczywista zawartosc CSV (moze byc duze - 89.1M
              # probek na caly dataset; rozwaz sparse-checkout jednego
              # site'u najpierw, patrz nizej)
```

Zeby pobrac tylko JEDEN site (znacznie mniej danych) przed pelnym `lfs
pull`:
```bash
git clone --filter=blob:none --sparse https://github.com/prasenjit52282/dalton-dataset.git
cd dalton-dataset
git sparse-checkout set Processed/H1 Metadata
git lfs pull
```
(zamien `H1` na dowolny SiteID z tabeli w README repo - `H6`/`H7`/`H9`
sa najmniejsze rezydencjalne site'y, ~200-800h danych).

## Schemat kolumn (zweryfikowany z README repo, nie zgadywany)

    ts        - znacznik czasu 'yyyy/mm/dd HH:MM:SS'
    T         - temperatura [stopnie C]
    H         - wilgotnosc wzgledna [%]
    PMS1/PMS2_5/PMS10 - pyl zawieszony [ug/m3]
    CO2       - dwutlenek wegla [ppm]
    NO2       - dwutlenek azotu [ppb]
    CO        - tlenek wegla [ppm]
    VoC       - lotne zwiazki organiczne [ppb]
    C2H5OH    - etanol [ppb]
    ID        - identyfikator czujnika
    Loc       - lokalizacja w mieszkaniu (np. 'Kitchen', 'Bedroom')
    Customer  - anonimowy SiteID
    Valid     - (w wersji Processed) 1/0, czy wszystkie odczyty w zakresie
    Valid_CO2 - (w wersji Processed) 1/0, czy czujnik CO2 dziala poprawnie

Model VTT w tym repo potrzebuje TYLKO `T` i `H` - `CO2`/`VoC` sa
uzywane wylacznie jako DIAGNOSTYCZNY proxy wentylacji (patrz
ventilation_deficit_proxy() nizej), NIE wchodza do rownania mold_index_series
(brak w literaturze ugruntowanego ilosciowego mostu CO2/VOC -> tempo
wzrostu pleśni - nie zgadywano wspolczynnika, zeby uniknac numerologii).

## Rozdzielczosc czasowa

DALTON to ok. 1 probka/sekunde (89.1M probek / 13646h ~ 6500 probek/h).
Model VTT zaklada wolno zmieniajace sie warunki (skala tygodni) - ten
skrypt AGREGUJE do srednich GODZINOWYCH przed policzeniem mold_index_series
(patrz resample_hourly()) - to zarowno przyspiesza obliczenia (dziesiatki
tysiecy godzin zamiast dziesiatek milionow probek), jak i jest zgodne z
jednostka czasu (godziny) w oryginalnym rownaniu VTT.
"""

import glob

import numpy as np
import pandas as pd

from timdr_mold_fusion import TIMDRMoldFusion
from timdr_mold_predict import TIMDRMoldPredict


def describe_columns(path):
    """Wypisuje kolumny znalezionego pliku - uzyj PIERWSZE, zanim
    zawolasz ingest_dalton_csv(), zeby sprawdzic czy Twoj pobrany plik
    faktycznie ma oczekiwany uklad (schemat moze sie zmienic w kolejnych
    wersjach datasetu)."""
    files = sorted(glob.glob(path)) if "*" in path else [path]
    if not files:
        raise FileNotFoundError(f"brak plikow pasujacych do: {path}")
    df = pd.read_csv(files[0], nrows=5)
    print(f"Plik: {files[0]}")
    print(f"Kolumny ({len(df.columns)}): {list(df.columns)}")
    return list(df.columns)


def resample_hourly(df, time_col="ts", temp_col="T", hum_col="H"):
    """Agreguje do srednich godzinowych. Odrzuca wiersze z Valid==0 jesli
    kolumna Valid jest obecna (dane z folderu Processed/ maja ja z
    definicji - patrz pipeline w README repo zrodlowego)."""
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    if "Valid" in df.columns:
        df = df[df["Valid"] == 1]
    df = df.set_index(time_col)
    hourly = df[[temp_col, hum_col]].resample("1h").mean().dropna()
    return hourly


def ingest_dalton_csv(path, time_col="ts", temp_col="T", hum_col="H"):
    """Wczytuje jeden lub wiele plikow CSV (path moze byc wzorcem glob) z
    folderu Processed/, agreguje do godzin, zwraca dict {t_hours,
    temperature, humidity} gotowy do TIMDRMoldFusion.fuse(t, temp, hum)."""
    files = sorted(glob.glob(path)) if "*" in path else [path]
    if not files:
        raise FileNotFoundError(f"brak plikow pasujacych do: {path}")

    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)

    missing = [c for c in [time_col, temp_col, hum_col] if c not in df.columns]
    if missing:
        raise ValueError(
            f"brakujace kolumny: {missing}. Uruchom describe_columns('{path}') "
            f"i podaj poprawne nazwy (time_col=, temp_col=, hum_col=)."
        )

    hourly = resample_hourly(df, time_col, temp_col, hum_col)
    t0 = hourly.index[0]
    t_hours = (hourly.index - t0).total_seconds().to_numpy() / 3600.0
    return {
        "t_hours": t_hours,
        "temperature": hourly[temp_col].to_numpy(),
        "humidity": hourly[hum_col].to_numpy(),
    }


def ventilation_deficit_proxy(co2_ppm, threshold_ppm=1000.0):
    """DIAGNOSTYCZNY (nie wchodzi do mold_index_series) proxy slabej
    wentylacji: udzial czasu z CO2 powyzej progu ASHRAE 62.1/EN 16798
    (~1000ppm jako granica 'adekwatnej wentylacji' dla pomieszczen
    mieszkalnych - standard branzowy, nie wymyslony na potrzeby tego
    projektu). Wysoki wynik = pomieszczenie czesto slabo wentylowane =
    MOZLIWY (nie udowodniony ilosciowo w tym module) dodatkowy czynnik
    ryzyka - do raportowania OBOK M(t), nie mieszania z nim."""
    co2 = np.asarray(co2_ppm, float)
    valid = np.isfinite(co2)
    if valid.sum() == 0:
        return None
    return float(np.mean(co2[valid] > threshold_ppm))


def run_real_validation(path, vulnerability_class="bardzo_wrazliwy", **ingest_kwargs):
    """Pelny przebieg: wczytaj realne dane, policz M(t), zwroc podsumowanie.
    WYPISUJE wynik, nie twierdzi z gory jaki bedzie."""
    data = ingest_dalton_csv(path, **ingest_kwargs)
    fusion = TIMDRMoldFusion(vulnerability_class=vulnerability_class)
    predict = TIMDRMoldPredict()

    M, margin = fusion.fuse(data["t_hours"], data["temperature"], data["humidity"])

    print(f"Probek godzinowych: {len(data['t_hours'])} ({data['t_hours'][-1]/24:.1f} dni)")
    print(f"T: min={data['temperature'].min():.1f} mean={data['temperature'].mean():.1f} "
          f"max={data['temperature'].max():.1f} degC")
    print(f"H: min={data['humidity'].min():.1f} mean={data['humidity'].mean():.1f} "
          f"max={data['humidity'].max():.1f} %")
    print(f"M(t): max={M.max():.3f}, koncowe={M[-1]:.3f}")

    over1 = np.where(M >= 1.0)[0]
    over3 = np.where(M >= 3.0)[0]
    if len(over1):
        print(f"Pierwsze przekroczenie M>=1 (slady): dzien {data['t_hours'][over1[0]]/24:.1f}")
    if len(over3):
        print(f"Pierwsze przekroczenie M>=3 (widoczne): dzien {data['t_hours'][over3[0]]/24:.1f}")

    health = predict.health_score(M)
    ttg = predict.time_to_visible_growth(data["t_hours"], M)
    print(f"health_score: {health:.3f}")
    print(f"time_to_visible_growth: {ttg}")

    idx_an, z_an = fusion.anomalies(M)
    sl, z_tr = fusion.trend(data["t_hours"], M, window=24 * 7)
    print(f"anomalies: {len(idx_an)}/{len(M)}, max|trend_z|: {np.nanmax(np.abs(z_tr)):.2f}")

    return {"M": M, "margin": margin, "t_hours": data["t_hours"], "health_score": health}


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nUzycie: python real_dalton_test.py <sciezka_do_csv_lub_wzorca_glob> [--describe]")
        sys.exit(1)
    path = sys.argv[1]
    if "--describe" in sys.argv:
        describe_columns(path)
    else:
        run_real_validation(path)
