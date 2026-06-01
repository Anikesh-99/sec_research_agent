"""
Pull 10-K / 10-Q filings from SEC EDGAR.

EDGAR is free but picky: every request needs a descriptive User-Agent with
contact info, or they rate-limit / block you. We read it from SEC_USER_AGENT.

Flow:
  ticker -> CIK  (via the public company_tickers.json map)
  CIK    -> filing history (submissions API)
  filter to the form we want, grab the primary document URL, download the HTML.
"""

import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

_USER_AGENT = os.getenv("SEC_USER_AGENT", "")
if not _USER_AGENT:
    # Don't hard-crash on import; fail loudly when we actually hit the network.
    _USER_AGENT = "sec_research_agent unset-contact@example.com"

HEADERS = {"User-Agent": _USER_AGENT}
TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"


def _get(url: str) -> requests.Response:
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    time.sleep(0.2)  # be polite; EDGAR fair-access is ~10 req/s
    return resp


def get_cik(ticker: str) -> int:
    """Map a ticker like 'NVDA' to its zero-padded CIK integer."""
    data = _get(TICKER_MAP_URL).json()
    ticker = ticker.upper()
    for row in data.values():
        if row["ticker"].upper() == ticker:
            return int(row["cik_str"])
    raise ValueError(f"Ticker {ticker!r} not found in EDGAR ticker map")


def get_filings(ticker: str, form: str = "10-K", count: int = 2) -> list[dict]:
    """
    Return metadata for the `count` most recent filings of `form` for `ticker`.

    Each item: {accession, form, filing_date, primary_doc, url}
    """
    cik = get_cik(ticker)
    subs = _get(SUBMISSIONS_URL.format(cik=cik)).json()
    recent = subs["filings"]["recent"]

    out = []
    for i, f in enumerate(recent["form"]):
        if f != form:
            continue
        accession = recent["accessionNumber"][i]
        primary_doc = recent["primaryDocument"][i]
        acc_nodash = accession.replace("-", "")
        url = (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{cik}/{acc_nodash}/{primary_doc}"
        )
        out.append({
            "accession": accession,
            "form": form,
            "filing_date": recent["filingDate"][i],
            "primary_doc": primary_doc,
            "url": url,
            "ticker": ticker.upper(),
        })
        if len(out) >= count:
            break
    return out


def download_filing(url: str) -> str:
    """Download a filing's primary HTML document."""
    return _get(url).text


if __name__ == "__main__":
    # Quick smoke test: list NVDA's two most recent 10-Ks.
    for f in get_filings("NVDA", "10-K", 2):
        print(f["filing_date"], f["accession"], f["url"])
