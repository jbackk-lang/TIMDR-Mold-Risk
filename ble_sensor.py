"""
ble_sensor.py — odczyt Xiaomi Mijia LYWSD03MMC przez BLE (Bluetooth Low Energy)
================================================================================
WYMAGANIE: sensor musi miec wgrany CUSTOM firmware (pvvx lub atc1441 —
https://github.com/pvvx/ATC_MiThermometer), NIE fabryczny firmware Xiaomi.
Fabryczny firmware szyfruje dane (protokol MiBeacon) i wymaga "bindkey"
uzyskanego przez aplikacje Xiaomi Home/Home Assistant — tego NIE obsluguje
ten modul (zbyt zawodne/niepewne do zaimplementowania bez dostepu do
prawdziwego urzadzenia i klucza). Custom firmware nadaje dane JAWNIE
(bez szyfrowania) w Service Data pod UUID 0x181A — ten format parsuje
ponizszy kod.

UWAGA UCZCIWOSCI (waznie, przeczytaj przed zaufaniem odczytom): ten
parser NIE zostal przetestowany na prawdziwym urzadzeniu w tym
srodowisku — sandbox, w ktorym to powstalo, nie ma fizycznego adaptera
Bluetooth. Uklad bajtow ponizej odtworzony z pamieci na podstawie
publicznie znanej dokumentacji formatow "custom"/"atc1441" firmware
pvvx/ATC_MiThermometer. PRZED zaufaniem odczytom na Twoim sprzecie:

    python ble_sensor.py --scan --debug

wypisze surowe bajty KAZDEJ odebranej reklamy z UUID 0x181A oraz to, co
z nich zostalo odparsowane — porownaj wynik (temperatura/wilgotnosc) z
tym, co pokazuje wyswietlacz samego czujnika lub apka nRF Connect. Jesli
sie nie zgadza, popraw funkcje parse_atc1441/parse_pvvx nizej (najbardziej
prawdopodobny blad: kolejnosc bajtow MAC albo big/little-endian liczb) —
zostaw komentarz, ktora wersja zostala faktycznie zweryfikowana na
sprzecie i kiedy.

Dwa warianty formatu Service Data (auto-wykrywane po dlugosci payloadu):
  - atc1441 (13 bajtow): MAC(6, w kolejnosci "jak wydrukowana") +
    temperatura int16 big-endian (0.1 stopnia C) + wilgotnosc uint8 (%) +
    napiecie baterii uint16 big-endian (mV) + bateria uint8 (%) +
    licznik ramek uint8.
  - pvvx (15 bajtow): MAC(6, kolejnosc odwrocona wzgledem atc1441) +
    temperatura int16 little-endian (0.01 stopnia C) + wilgotnosc
    uint16 little-endian (0.01%) + napiecie baterii uint16
    little-endian (mV) + bateria uint8 (%) + licznik uint8 + flagi uint8.
"""

import argparse
import asyncio
import struct
import time
from collections import deque

SERVICE_DATA_UUID = "0000181a-0000-1000-8000-00805f9b34fb"


def parse_atc1441(data: bytes) -> dict:
    if len(data) != 13:
        raise ValueError(f"atc1441: oczekiwano 13 bajtow, dostalem {len(data)}")
    mac = data[0:6].hex(":")
    (temp_raw,) = struct.unpack_from(">h", data, 6)
    humidity = data[8]
    (batt_mv,) = struct.unpack_from(">H", data, 9)
    batt_pct = data[11]
    counter = data[12]
    return {
        "mac": mac,
        "temperature": temp_raw / 10.0,
        "humidity": float(humidity),
        "battery_mv": batt_mv,
        "battery_pct": batt_pct,
        "counter": counter,
        "format": "atc1441",
    }


def parse_pvvx(data: bytes) -> dict:
    if len(data) != 15:
        raise ValueError(f"pvvx: oczekiwano 15 bajtow, dostalem {len(data)}")
    mac = data[0:6][::-1].hex(":")
    temp_raw, hum_raw, batt_mv = struct.unpack_from("<hHH", data, 6)
    batt_pct = data[12]
    counter = data[13]
    flags = data[14]
    return {
        "mac": mac,
        "temperature": temp_raw / 100.0,
        "humidity": hum_raw / 100.0,
        "battery_mv": batt_mv,
        "battery_pct": batt_pct,
        "counter": counter,
        "flags": flags,
        "format": "pvvx",
    }


def parse_service_data(data: bytes) -> dict:
    """Rozpoznaje format po dlugosci payloadu Service Data (UUID 0x181A)."""
    if len(data) == 13:
        return parse_atc1441(data)
    if len(data) == 15:
        return parse_pvvx(data)
    raise ValueError(
        f"Nieznany format Service Data (dlugosc {len(data)}B) — nie pasuje "
        f"ani do atc1441 (13B), ani do pvvx (15B). Upewnij sie, ze sensor ma "
        f"wgrany custom firmware pvvx/ATC_MiThermometer, nie fabryczny."
    )


class LiveReadingBuffer:
    """Bufor ostatnich odczytow w pamieci, wspoldzielony miedzy watkiem
    skanujacym BLE a endpointami Flask API."""

    def __init__(self, maxlen=100000):
        self.readings = deque(maxlen=maxlen)  # (unix_time, temperature, humidity, mac)
        self.start_t = None
        self.last_error = None
        self.target_mac = None  # jesli ustawione: ignoruj odczyty z innych MAC-ow

    def add(self, temperature, humidity, mac):
        if self.target_mac and mac.lower() != self.target_mac.lower():
            return
        now = time.time()
        if self.start_t is None:
            self.start_t = now
        self.readings.append((now, temperature, humidity, mac))

    def as_series(self):
        """Zwraca (t_hours, temperature, humidity) wzgledem pierwszego
        odczytu w buforze — dokladnie ten sam ksztalt co make_demo_data(),
        zeby dalo sie podac wprost do TIMDRMoldFusion.fuse()."""
        if not self.readings or self.start_t is None:
            return [], [], []
        t_hours = [(r[0] - self.start_t) / 3600.0 for r in self.readings]
        temps = [r[1] for r in self.readings]
        hums = [r[2] for r in self.readings]
        return t_hours, temps, hums


async def scan_forever(buffer: LiveReadingBuffer, stop_event=None, on_reading=None, debug=False):
    """Petla skanowania BLE — wymaga zainstalowanego pakietu `bleak` i
    realnego adaptera Bluetooth w systemie. Uruchamiaj w osobnym watku
    z wlasna petla asyncio (patrz api.py, endpoint /api/ble/start) —
    ta funkcja sama nie zwraca sterowania dopoki `stop_event` nie
    zostanie ustawiony."""
    from bleak import BleakScanner

    def _callback(device, advertisement_data):
        raw = advertisement_data.service_data.get(SERVICE_DATA_UUID)
        if not raw:
            return
        raw = bytes(raw)
        if debug:
            print(f"[ble_sensor] surowe bajty od {device.address}: {raw.hex()}")
        try:
            reading = parse_service_data(raw)
        except ValueError as exc:
            buffer.last_error = str(exc)
            if debug:
                print(f"[ble_sensor] blad parsowania: {exc}")
            return
        if debug:
            print(f"[ble_sensor] odczyt: {reading}")
        buffer.add(reading["temperature"], reading["humidity"], reading["mac"])
        if on_reading:
            on_reading(reading)

    scanner = BleakScanner(_callback)
    try:
        # Timeout na starcie: na Windows pierwsze uzycie BLE z aplikacji
        # spoza Microsoft Store potrafi pokazac systemowy monit o
        # pozwolenie (Ustawienia > Prywatnosc > Bluetooth) lub po prostu
        # zawiesic sie, gdy adapter Bluetooth jest wylaczony/niedostepny -
        # bez timeoutu ten watek (i cala funkcja skanowania) wisialby
        # w nieskonczonosc, bez zadnej informacji zwrotnej w dashboardzie.
        await asyncio.wait_for(scanner.start(), timeout=15.0)
    except Exception as exc:  # noqa: BLE001 - musi przezyc kazdy blad startu adaptera
        buffer.last_error = (
            f"nie udalo sie uruchomic skanowania BLE w 15s: {exc}. "
            f"Sprawdz: czy Bluetooth jest wlaczony w Windows, czy nie "
            f"pojawilo sie okienko z prosba o pozwolenie (czasem chowa sie "
            f"za innymi oknami - Alt+Tab), i czy masz zainstalowany 'bleak'."
        )
        return
    try:
        while not (stop_event and stop_event.is_set()):
            await asyncio.sleep(1.0)
    finally:
        try:
            await asyncio.wait_for(scanner.stop(), timeout=5.0)
        except Exception:
            pass


def _cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", action="store_true", help="skanuj i wypisuj odczyty na biezaco")
    parser.add_argument("--debug", action="store_true", help="wypisz tez surowe bajty kazdej reklamy")
    parser.add_argument("--mac", default=None, help="filtruj do jednego adresu MAC")
    args = parser.parse_args()

    if not args.scan:
        parser.print_help()
        return

    buffer = LiveReadingBuffer()
    buffer.target_mac = args.mac

    def _on_reading(reading):
        print(f"[{time.strftime('%H:%M:%S')}] {reading['mac']}  "
              f"T={reading['temperature']:.1f}C  RH={reading['humidity']:.1f}%  "
              f"bateria={reading.get('battery_pct', '?')}%  format={reading['format']}")

    try:
        asyncio.run(scan_forever(buffer, on_reading=_on_reading, debug=args.debug))
    except KeyboardInterrupt:
        print("\nzatrzymano.")


if __name__ == "__main__":
    _cli()
