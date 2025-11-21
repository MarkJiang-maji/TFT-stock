"""
Download TX futures settlement calendar data from TAIFEX open API and save to CSV.

The script queries the '/SettledPositionsOfContractsOnExpirationDate' endpoint and
extracts settlement dates for contracts that contain 'TX' (e.g. TX, MTX, TMF). Only
monthly contracts (delivery month formatted as YYYYMM) are kept, so weekly expiries
such as 202511W1 are ignored.
"""

from __future__ import annotations

import csv
import os
import datetime as dt
import sys
from pathlib import Path
from typing import List, Dict, Any

import json
import subprocess

API_URL = "https://openapi.taifex.com.tw/v1/SettledPositionsOfContractsOnExpirationDate"
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(__file__).resolve().parents[1] / "Data"
OUTPUT_DIR = ROOT_DIR / "outputs"
OUTPUT_PATH = OUTPUT_DIR / "TX_settlement_calendar_webcrawl.csv"
FINAL_DATA_PATH = DATA_DIR / "TX_settlement_calendar_webcrawl.csv"
RAW_JSON_PATH = OUTPUT_DIR / "TX_settlement_calendar_webcrawl_raw.json"
TARGET_CONTRACT = "TX"


def fetch_settlement_data() -> List[Dict[str, Any]]:
    if RAW_JSON_PATH.exists():
        with RAW_JSON_PATH.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    """
    Try PowerShell's WebClient first (works across WSL/Windows proxy setups),
    then fall back to curl if PowerShell is unavailable.
    """
    commands = [
        [
            "powershell.exe",
            "-Command",
            (
                "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8;"
                "(New-Object Net.WebClient).DownloadString('{url}')"
            ).format(url=API_URL),
        ],
        ["curl", "-sS", API_URL],
    ]
    last_error: Exception | None = None
    for cmd in commands:
        try:
            completed = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
            try:
                return json.loads(completed.stdout)
            except json.JSONDecodeError as exc:
                last_error = exc
        except FileNotFoundError as exc:
            last_error = exc
        except subprocess.CalledProcessError as exc:
            last_error = RuntimeError(exc.stderr.strip() or str(exc))
    raise RuntimeError(f"Unable to download settlement data: {last_error}")


def _delivery_month(row: Dict[str, Any]) -> str:
    return str(row.get("ContractDeliveryMonth") or row.get("DeliveryMonth") or "").strip()


def is_tx_monthly_contract(row: Dict[str, Any]) -> bool:
    contract = row.get("Contract", "").strip()
    delivery_month = _delivery_month(row)
    return (
        contract == TARGET_CONTRACT
        and len(delivery_month) == 6
        and delivery_month.isdigit()
    )


def normalize_rows(raw: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    keep: Dict[str, Dict[str, str]] = {}
    for row in raw:
        if not is_tx_monthly_contract(row):
            continue
        date_str = row.get("TheFinalSettlementDay")
        delivery_month = _delivery_month(row)
        if not date_str or len(date_str) != 8:
            continue
        try:
            settle_date = dt.datetime.strptime(date_str, "%Y%m%d").date()
        except ValueError:
            continue
        key = settle_date.isoformat()
        keep[key] = {
            "settlement_date": key,
            "delivery_month": delivery_month,
            "contract": row.get("Contract", ""),
            "contract_name": row.get("ContractName", ""),
            "source": API_URL,
            "fetched_at": dt.datetime.utcnow().isoformat(timespec="seconds"),
        }
    return sorted(keep.values(), key=lambda item: item["settlement_date"])


def save_csv(rows: List[Dict[str, str]]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["settlement_date", "delivery_month", "contract", "contract_name", "source", "fetched_at"]
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    sync_to_data_dir()


def sync_to_data_dir() -> None:
    if not OUTPUT_PATH.exists():
        return
    if os.access(FINAL_DATA_PATH.parent, os.W_OK):
        try:
            with OUTPUT_PATH.open("rb") as src, FINAL_DATA_PATH.open("wb") as dst:
                dst.write(src.read())
            print(f"[crawler] Copied CSV to {FINAL_DATA_PATH}")
            return
        except PermissionError:
            pass
    try:
        win_src = subprocess.run(
            ["wslpath", "-w", str(OUTPUT_PATH)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        win_dst = subprocess.run(
            ["wslpath", "-w", str(FINAL_DATA_PATH)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        subprocess.run(
            [
                "powershell.exe",
                "-Command",
                f"Copy-Item -Force '{win_src}' '{win_dst}'",
            ],
            check=True,
        )
        print(f"[crawler] Copied CSV to {FINAL_DATA_PATH} via PowerShell")
    except Exception as exc:
        print(
            f"[crawler] Unable to copy CSV into Data directory automatically ({exc}). "
            f"Please copy {OUTPUT_PATH} -> {FINAL_DATA_PATH} manually."
        )


def main() -> int:
    try:
        raw = fetch_settlement_data()
    except Exception as exc:  # pragma: no cover - network errors
        print(f"[crawler] Failed to download settlement data: {exc}", file=sys.stderr)
        return 1
    rows = normalize_rows(raw)
    if not rows:
        print("[crawler] No TX settlement rows were found in API payload.", file=sys.stderr)
        return 1
    save_csv(rows)
    print(f"[crawler] Saved {len(rows)} settlement entries to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
