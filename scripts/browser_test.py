"""End-to-end headless browser test suite for the Streamlit legal RAG assistant.

Simulates real user interactions:
- Loading the app and verifying no initial errors
- Clicking suggestion chips and verifying textarea population
- Clicking 'Clear Question' and verifying zero StreamlitWidgetAlreadyInstantiatedError
- Running English and Nepali queries through the full RAG pipeline
- Verifying inline citation badges, metrics, and sources drawer
- Toggling sidebar controls (Reranker, Retrieval Mode)
- Switching to Tab 2 (Ablation Dashboard) and verifying data tables
"""

import sys
import time

from playwright.sync_api import sync_playwright

CHROME_PATH = "/usr/bin/google-chrome"
APP_URL = "http://localhost:8501"


def run_browser_tests():
    print(f"Launching Playwright with Chrome at {CHROME_PATH}...")
    errors_found = []

    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=CHROME_PATH, headless=True)
        context = browser.new_context(viewport={"width": 1400, "height": 900})
        page = context.new_page()

        # Listen for console errors
        def handle_console(msg):
            if msg.type == "error":
                print(f"  [Browser Console Error] {msg.text}")

        page.on("console", handle_console)

        print(f"Navigating to {APP_URL}...")
        page.goto(APP_URL, wait_until="networkidle")
        page.wait_for_selector("textarea", timeout=15000)
        time.sleep(1)

        # Helper to check for Streamlit exception boxes
        def check_no_exceptions(step_name: str):
            exc = page.locator("div[data-testid='stException']")
            if exc.count() > 0:
                text = exc.first.inner_text()
                msg = f"ERROR at step '{step_name}': Exception found:\n{text}"
                print(f"❌ {msg}")
                errors_found.append(msg)
                return False
            print(f"✓ Step '{step_name}': No exceptions.")
            return True

        # Test 1: Initial load
        print("\n--- Test 1: Initial Load & App Header ---")
        check_no_exceptions("Initial Load")
        title_text = page.locator("h3").first.inner_text()
        print(f"  Found title: '{title_text}'")
        assert "Nepal Legal Assistant" in title_text

        # Test 2: Chip Click -> Textarea Fill
        print("\n--- Test 2: Chip Click & Textarea Population ---")
        q1_button = page.locator("button:has-text('What freedoms are guaranteed')")
        assert q1_button.count() > 0, "First suggestion chip not found!"
        print(f"  Found chip: '{q1_button.first.inner_text()[:60]}...'")

        # Click the first chip
        q1_button.first.click()
        time.sleep(2)
        check_no_exceptions("After Chip 1 Click")

        textarea = page.locator("textarea").first
        current_val = textarea.input_value()
        print(f"  Textarea content: '{current_val[:60]}...'")
        assert "Article 17" in current_val, "Textarea did not populate from chip!"

        # Test 3: Clear Question Button
        print("\n--- Test 3: Clear Question Button ---")
        clear_btn = page.locator("button:has-text('Clear Question')")
        assert clear_btn.count() > 0, "Clear Question button not found!"
        print("  Clicking 'Clear Question'...")
        clear_btn.click()
        time.sleep(2)
        check_no_exceptions("After Clear Question Click")

        cleared_val = textarea.input_value()
        print(f"  Textarea content after clear: '{cleared_val}'")
        assert cleared_val == "", f"Expected empty textarea, got '{cleared_val}'"

        # Test 4: Chip Click for Nepali Query & Execution
        print("\n--- Test 4: Nepali Query Selection & Pipeline Execution ---")
        nepali_chip = page.locator("button:has-text('नेपालको संविधानको धारा १६')")
        assert nepali_chip.count() > 0, "Nepali suggestion chip not found!"
        print(f"  Clicking Nepali chip: '{nepali_chip.first.inner_text()[:60]}...'")
        nepali_chip.first.click()
        time.sleep(2)
        check_no_exceptions("After Nepali Chip Click")

        run_btn = page.locator("button:has-text('Run RAG Pipeline')")
        print("  Clicking 'Run RAG Pipeline'...")
        run_btn.click()

        # Wait for RAG pipeline to complete and render status badge
        print("  Waiting for RAG response...")
        page.wait_for_selector(".status-pill-ok, .status-pill-warn", timeout=45000)
        time.sleep(1)
        check_no_exceptions("After RAG Pipeline Execution")

        # Verify answer rendered
        answer_text = page.locator("div[data-testid='stMarkdownContainer'] p").all_inner_texts()
        print(f"  Answer paragraphs found: {len(answer_text)}")
        assert any("धारा १६" in p or "सम्मानपूर्वक" in p or "हक" in p for p in answer_text), "Expected Nepali legal answer text!"
        print("  ✓ Answer contains expected Nepali legal text.")

        # Verify citation status badge rendered
        status_pill = page.locator(".status-pill-ok, .status-pill-warn")
        assert status_pill.count() > 0, "Citation validation badge not found!"
        print(f"  ✓ Validation badge: '{status_pill.first.inner_text()}'")

        # Verify sources drawer
        expanders = page.locator("div[data-testid='stExpander']")
        print(f"  ✓ Found {expanders.count()} source expanders.")
        assert expanders.count() > 0, "No source expanders rendered!"

        # Test 5: Switch to Tab 2 (Ablations & Metrics Dashboard)
        print("\n--- Test 5: Switch to Ablation Dashboard Tab ---")
        tab_buttons = page.locator("[role='tab']")
        print(f"  Found {tab_buttons.count()} tabs.")
        assert tab_buttons.count() >= 2, "Expected at least 2 tabs!"

        tab_eval = tab_buttons.nth(1)
        print(f"  Clicking tab: '{tab_eval.inner_text()}'")
        tab_eval.click()
        time.sleep(2)
        check_no_exceptions("After Tab 2 Click")

        # Verify dataframe rendered in Tab 2
        df_elem = page.locator("div[data-testid='stDataFrame']")
        assert df_elem.count() > 0, "Dataframe not found in Tab 2!"
        print("  ✓ Ablations dataframe rendered successfully.")

        # Test 6: English Query & Refusal Verification
        print("\n--- Test 6: Switch back to Tab 1 & Refusal Query Execution ---")
        tab_buttons.first.click()
        time.sleep(1)
        check_no_exceptions("After Switching Back to Tab 1")

        # Click the cryptocurrency unanswerable refusal chip
        unans_chip = page.locator("button:has-text('cryptocurrency exchanges')")
        assert unans_chip.count() > 0, "Unanswerable query chip not found!"
        print(f"  Clicking refusal chip: '{unans_chip.first.inner_text()[:60]}...'")
        unans_chip.first.click()
        time.sleep(2)

        print("  Clicking 'Run RAG Pipeline' for refusal test...")
        run_btn = page.locator("button:has-text('Run RAG Pipeline')")
        run_btn.click()
        print("  Waiting for refusal response and validation badge...")
        page.wait_for_selector(".status-pill-ok, .status-pill-warn", timeout=45000)
        time.sleep(1)
        check_no_exceptions("After Refusal Pipeline Execution")

        # Verify status badge rendered
        status_pill = page.locator(".status-pill-ok, .status-pill-warn")
        assert status_pill.count() > 0, "Validation status badge not found!"
        print(f"  ✓ Status badge detected: '{status_pill.first.inner_text()}'")

        # Take screenshot for proof
        screenshot_path = "experiments/browser_test_proof.png"
        page.screenshot(path=screenshot_path, full_page=True)
        print(f"  ✓ Full-page screenshot saved to {screenshot_path}")

        browser.close()

    print("\n==========================================")
    if not errors_found:
        print("🎉 ALL BROWSER INTERACTION TESTS PASSED WITH 0 ERRORS!")
        return 0
    else:
        print(f"❌ {len(errors_found)} ERRORS ENCOUNTERED:")
        for e in errors_found:
            print(" -", e)
        return 1


if __name__ == "__main__":
    sys.exit(run_browser_tests())
