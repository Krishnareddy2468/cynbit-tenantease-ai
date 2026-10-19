"""Capture the four dashboard tabs to screenshots/ with Playwright.

Assumes a Streamlit server is already running (default http://localhost:8599).
Run from the project root after `streamlit run app.py`:

    python scripts/capture_screenshots.py [--url http://localhost:8599]
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHOTS = ROOT / "screenshots"

TABS = [
    ("🎯 Tenant Risk", "01_tenant_risk.png"),
    ("📈 Revenue Forecast", "02_revenue_forecast.png"),
    ("💡 AI Recommendations", "03_recommendations.png"),
    ("🔬 Model Performance", "04_model_performance.png"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8599")
    args = parser.parse_args()

    from playwright.sync_api import sync_playwright

    SHOTS.mkdir(exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1024}, device_scale_factor=2)
        page.goto(args.url, wait_until="networkidle", timeout=60000)

        # Wait for the app to finish its first render.
        page.get_by_role("tab").first.wait_for(timeout=60000)
        page.wait_for_timeout(4000)

        for label, filename in TABS:
            tab = page.get_by_role("tab", name=label)
            tab.click()
            # Let charts render and the "running" spinner clear.
            page.wait_for_timeout(5000)
            try:
                page.wait_for_selector("[data-testid='stStatusWidget']", state="detached", timeout=8000)
            except Exception:
                pass
            page.screenshot(path=str(SHOTS / filename), full_page=True)
            print(f"  captured {filename}")

        browser.close()
    print(f"Done. Screenshots in {SHOTS}")


if __name__ == "__main__":
    main()
