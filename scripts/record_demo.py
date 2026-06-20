"""One-off script: drive the Gradio app with Playwright and capture screenshots for a GIF."""
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT_DIR = Path(__file__).parent / "gif_frames"
OUT_DIR.mkdir(exist_ok=True)
URL = "http://0.0.0.0:7861/"


def shot(page, frames, label, repeat=1):
    for _ in range(repeat):
        path = OUT_DIR / f"{len(frames):03d}_{label}.png"
        page.screenshot(path=str(path), full_page=True)
        frames.append(path)


def main():
    frames = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        page.goto(URL, wait_until="networkidle")
        page.wait_for_selector("text=LocalRAG Q&A System")
        shot(page, frames, "loaded", repeat=2)
        time.sleep(0.3)

        query_box = page.locator("textarea").first
        query_box.click()
        question = "How does RAG work?"
        for ch in question:
            query_box.type(ch, delay=35)
        shot(page, frames, "typed", repeat=2)
        time.sleep(0.3)

        search_radio = page.get_by_role("radio", name="hybrid")
        search_radio.click()
        shot(page, frames, "config", repeat=2)

        submit_btn = page.get_by_role("button", name="Generate Answer")
        submit_btn.click()

        # Capture frames while the answer streams in, polling for completion.
        status_box = page.locator("textarea").nth(2)
        for i in range(60):
            time.sleep(0.5)
            shot(page, frames, f"stream{i:02d}")
            try:
                status_val = status_box.input_value()
            except Exception:
                status_val = ""
            if status_val == "Done":
                break

        # Hold on the final answer for a couple of frames.
        shot(page, frames, "final", repeat=4)

        browser.close()
    print(f"Captured {len(frames)} frames in {OUT_DIR}")


if __name__ == "__main__":
    main()
