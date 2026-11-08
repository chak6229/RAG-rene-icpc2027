"""Bounded public lookup of the dataset license declaration and full terms."""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import re
import urllib.parse
import urllib.request
from datetime import datetime, timezone

from discover_sources import ROOT, OUT, SOURCES


def fetch(item):
    name, url = item
    record = {"url": url, "requested_utc": datetime.now(timezone.utc).isoformat()}
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 research-source-audit"})
        with urllib.request.urlopen(request, timeout=18) as response:
            data = response.read(3_000_000)
            path = SOURCES / name
            path.write_bytes(data)
            record.update(status=response.status, final_url=response.url, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), path=path.relative_to(ROOT).as_posix())
            text = data.decode("utf-8", errors="replace")
            record["candidate_links"] = sorted(set(re.findall(r'href=["\']([^"\']+)["\']', text)))[:200]
    except Exception as error:
        record["error"] = type(error).__name__ + ": " + str(error)
    return record


def main():
    query = urllib.parse.quote('"Computational Use of Data Agreement"')
    urls = [
        ("cuda_bing_search.html", "https://www.bing.com/search?q=" + query),
        ("cuda_msr_search.html", "https://www.microsoft.com/en-us/research/search/?q=" + query),
        ("cosqa_huggingface_metadata.json", "https://huggingface.co/api/datasets?search=cosqa&limit=10&full=true"),
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        records = list(executor.map(fetch, urls))
    (OUT / "LICENSE_LOOKUP.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    for record in records:
        print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
