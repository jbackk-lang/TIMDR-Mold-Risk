"""
test_ble_sensor.py — testy jednostkowe parsera BLE.

UWAGA: te testy sprawdzaja WYLACZNIE, ze parse_atc1441/parse_pvvx
poprawnie odczytuja bajty w ukladzie, ktory SAM sobie zalozylem w
docstringu ble_sensor.py (na podstawie pamieciowej rekonstrukcji formatu
pvvx/atc1441) - to jest self-consistency check kodu, NIE dowod, ze ten
uklad bajtow zgadza sie z prawdziwym sensorem LYWSD03MMC. Prawdziwa
weryfikacja wymaga `python ble_sensor.py --scan --debug` na fizycznym
urzadzeniu - patrz README.md sekcja "Czujnik Bluetooth".
"""

import struct

import pytest

from ble_sensor import (
    LiveReadingBuffer,
    parse_atc1441,
    parse_pvvx,
    parse_service_data,
)


def test_parse_atc1441_roundtrip():
    mac = bytes.fromhex("a4c138aabbcc")
    payload = (
        mac
        + struct.pack(">h", 235)   # 23.5 degC
        + bytes([55])              # 55 %RH
        + struct.pack(">H", 2980)  # 2980 mV
        + bytes([80])              # 80 % baterii
        + bytes([7])               # licznik ramek
    )
    r = parse_atc1441(payload)
    assert r["mac"] == "a4:c1:38:aa:bb:cc"
    assert r["temperature"] == pytest.approx(23.5)
    assert r["humidity"] == pytest.approx(55.0)
    assert r["battery_mv"] == 2980
    assert r["battery_pct"] == 80
    assert r["counter"] == 7
    assert r["format"] == "atc1441"


def test_parse_atc1441_wrong_length_raises():
    with pytest.raises(ValueError):
        parse_atc1441(b"\x00" * 10)


def test_parse_pvvx_roundtrip():
    mac = bytes.fromhex("a4c138aabbcc")
    mac_rev = bytes(reversed(mac))
    payload = (
        mac_rev
        + struct.pack("<h", 2350)   # 23.50 degC
        + struct.pack("<H", 5500)   # 55.00 %RH
        + struct.pack("<H", 2980)   # 2980 mV
        + bytes([80, 7, 1])         # battery%, counter, flags
    )
    r = parse_pvvx(payload)
    assert r["mac"] == "a4:c1:38:aa:bb:cc"
    assert r["temperature"] == pytest.approx(23.5)
    assert r["humidity"] == pytest.approx(55.0)
    assert r["battery_mv"] == 2980
    assert r["battery_pct"] == 80
    assert r["counter"] == 7
    assert r["flags"] == 1
    assert r["format"] == "pvvx"


def test_parse_pvvx_wrong_length_raises():
    with pytest.raises(ValueError):
        parse_pvvx(b"\x00" * 10)


def test_parse_service_data_dispatches_by_length():
    atc = parse_service_data(
        bytes.fromhex("a4c138aabbcc") + struct.pack(">h", 200) + bytes([50])
        + struct.pack(">H", 3000) + bytes([90, 1])
    )
    assert atc["format"] == "atc1441"

    pvvx = parse_service_data(
        bytes.fromhex("ccbbaa38c1a4") + struct.pack("<h", 2000) + struct.pack("<H", 5000)
        + struct.pack("<H", 3000) + bytes([90, 1, 0])
    )
    assert pvvx["format"] == "pvvx"


def test_parse_service_data_unknown_length_raises():
    with pytest.raises(ValueError):
        parse_service_data(b"\x00" * 7)


def test_live_reading_buffer_as_series_shape():
    buf = LiveReadingBuffer()
    t0 = 1_700_000_000.0
    for i in range(5):
        buf.readings.append((t0 + i * 300.0, 23.0 + i * 0.1, 55.0 + i, "a4:c1:38:aa:bb:cc"))
    buf.start_t = t0

    t_hours, temps, hums = buf.as_series()
    assert len(t_hours) == len(temps) == len(hums) == 5
    assert t_hours[0] == pytest.approx(0.0)
    assert t_hours[-1] == pytest.approx(4 * 300.0 / 3600.0)
    assert temps[0] == pytest.approx(23.0)
    assert hums[-1] == pytest.approx(59.0)


def test_live_reading_buffer_filters_by_target_mac():
    buf = LiveReadingBuffer()
    buf.target_mac = "a4:c1:38:aa:bb:cc"
    buf.add(23.0, 50.0, "a4:c1:38:aa:bb:cc")
    buf.add(99.0, 99.0, "ff:ff:ff:ff:ff:ff")  # inny sensor, powinien byc zignorowany
    assert len(buf.readings) == 1
    assert buf.readings[0][1] == 23.0


def test_live_reading_buffer_empty_series():
    buf = LiveReadingBuffer()
    t_hours, temps, hums = buf.as_series()
    assert t_hours == [] and temps == [] and hums == []
