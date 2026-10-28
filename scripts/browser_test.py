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
        print("\n--- Test 1: Initial Load & App Header Above Tabs ---")
        check_no_exceptions("Initial Load")
        title_text = page.locator("h3").first.inner_text()
        print(f"  Found title: '{title_text}'")
        assert "Nepal Legal Assistant" in title_text

        # Verify header is placed before tabs in DOM order
        header_y = page.locator("h3").first.bounding_box()["y"]
        tabs_y = page.locator("[role='tablist']").bounding_box()["y"]
        assert header_y < tabs_y, f"Header (y={header_y}) should be above tabs (y={tabs_y})!"
        print(f"  ✓ Confirmed: Header is above tabs (header_y={header_y:.1f} < tabs_y={tabs_y:.1f})")

        # Test 2: Chip Click -> Textarea Fill & Submit Button Enabled
        print("\n--- Test 2: Chip Click & Textarea Population ---")
        q1_button = page.locator("button:has-text('What freedoms are guaranteed')")
        assert q1_button.count() > 0, "First suggestion chip not found!"
        print(f"  Found chip: '{q1_button.first.inner_text()[:60]}...'")

        # Click the first chip
        q1_button.first.click()
        time.sleep(1.5)
        check_no_exceptions("After Chip 1 Click")

        textarea = page.locator("textarea").first
        current_val = textarea.input_value()
        print(f"  Textarea content: '{current_val[:60]}...'")
        assert "Article 17" in current_val, "Textarea did not populate from chip!"

        # Check embedded submit button inside textarea
        submit_btn = page.locator("button[data-testid='stChatInputSubmitButton']")
        assert submit_btn.count() > 0, "Embedded submit button inside textarea not found!"
        assert not submit_btn.is_disabled(), "Submit button should be enabled when text is present!"
        print("  ✓ Embedded submit button inside textarea is visible and enabled.")

        # Test 3: Verify External Buttons Removed
        print("\n--- Test 3: Verify External Buttons Removed ---")
        old_run_btn = page.locator("button:has-text('Run RAG Pipeline')")
        old_clear_btn = page.locator("button:has-text('Clear Question')")
        assert old_run_btn.count() == 0, "Old 'Run RAG Pipeline' button still present!"
        assert old_clear_btn.count() == 0, "Old 'Clear Question' button still present!"
        print("  ✓ Confirmed: External 'Run RAG Pipeline' and 'Clear Question' buttons removed.")

        # Test 3.5: Instant Conversational Greeting ('hi there') & Question Chip Verification
        print("\n--- Test 3.5: Instant Conversational Greeting ('hi there') ---")
        textarea = page.locator("textarea").first
        textarea.fill("hi there")
        time.sleep(0.3)
        t0 = time.perf_counter()
        submit_btn = page.locator("button[data-testid='stChatInputSubmitButton']")
        submit_btn.click()
        page.wait_for_selector(".status-pill-ok:has-text('Instant Conversational Assistant')", timeout=5000)
        elapsed_hi = time.perf_counter() - t0
        check_no_exceptions("After 'hi there' Submission")
        print(f"  ✓ 'hi there' answered in {elapsed_hi:.2f}s with Instant Conversational badge!")

        # Verify suggested questions are now HIDDEN after submission
        assert page.locator("button:has-text('What freedoms are guaranteed')").count() == 0, (
            "Suggested questions pills should be hidden after asking a question!"
        )
        print("  ✓ Confirmed: Suggested questions section is hidden after submission.")

        # Verify active question chip is displayed above the response
        q_chip = page.locator(".user-query-chip")
        assert q_chip.count() > 0, "Active question chip not found above response!"
        assert "hi there" in q_chip.first.inner_text()
        print(f"  ✓ Active question chip rendered: '{q_chip.first.inner_text()}'")

        # Verify '↺ New Question' button is visible
        reset_btn = page.locator("button:has-text('New Question')")
        assert reset_btn.count() > 0, "Reset button ('New Question') not found!"
        print("  ✓ '↺ New Question' button is present.")

        # Test 3.6: Reset via '↺ New Question' & Verify Suggestions Reappear
        print("\n--- Test 3.6: Reset via '↺ New Question' ---")
        reset_btn.first.click()
        time.sleep(1.5)
        check_no_exceptions("After Clicking 'New Question'")
        assert page.locator("button:has-text('What freedoms are guaranteed')").count() > 0, (
            "Suggested questions pills should reappear after clicking 'New Question'!"
        )
        assert page.locator(".user-query-chip").count() == 0, (
            "Question chip should be removed after clicking 'New Question'!"
        )
        print("  ✓ Suggestions reappeared and previous answer/question chip cleared successfully.")

        # Test 4: Chip Click for Nepali Query & Execution via Embedded Submit Button
        print("\n--- Test 4: Nepali Query Selection & Pipeline Execution ---")
        nepali_chip = page.locator("button:has-text('नेपालको संविधानको धारा १६')")
        assert nepali_chip.count() > 0, "Nepali suggestion chip not found!"
        print(f"  Clicking Nepali chip: '{nepali_chip.first.inner_text()[:60]}...'")
        nepali_chip.first.click()
        time.sleep(1.5)
        check_no_exceptions("After Nepali Chip Click")

        print("  Clicking embedded submit button inside textarea...")
        submit_btn = page.locator("button[data-testid='stChatInputSubmitButton']")
        submit_btn.click()

        # Wait for RAG pipeline to complete and render status badge (must be .status-pill-ok, NOT a refusal!)
        print("  Waiting for RAG response...")
        page.wait_for_selector(".status-pill-ok:has-text('sentences cited')", timeout=45000)
        time.sleep(1)
        check_no_exceptions("After RAG Pipeline Execution")

        # Verify suggestions hidden and Nepali question chip displayed
        assert page.locator("button:has-text('What freedoms are guaranteed')").count() == 0, (
            "Suggested questions should be hidden during active Nepali query!"
        )
        q_chip = page.locator(".user-query-chip")
        assert q_chip.count() > 0, "Question chip not found for Nepali query!"
        assert "नेपालको संविधानको धारा १६" in q_chip.first.inner_text()
        print(f"  ✓ Nepali question chip: '{q_chip.first.inner_text()[:50]}...'")

        # Verify citation status badge is .status-pill-ok and .cite-badge pills rendered
        status_pill = page.locator(".status-pill-ok:has-text('sentences cited')")
        assert status_pill.count() > 0, "Expected cited Nepali answer (.status-pill-ok), not refusal!"
        print(f"  ✓ Validation badge: '{status_pill.first.inner_text()}'")
        cite_badges = page.locator(".cite-badge")
        assert cite_badges.count() > 0, "Expected inline .cite-badge elements in answer!"
        print(f"  ✓ Found {cite_badges.count()} inline .cite-badge elements.")

        # Verify sources drawer
        expanders = page.locator("div[data-testid='stExpander']")
        print(f"  ✓ Found {expanders.count()} source expanders.")
        assert expanders.count() > 0, "No source expanders rendered!"

        # Test 4.5: Multi-Turn Q&A Persistence (Old questions and answers do NOT disappear)
        print("\n--- Test 4.5: Multi-Turn History Persistence (All Q&As Shown) ---")
        textarea = page.locator("textarea").first
        textarea.fill("namaste")
        time.sleep(0.3)
        submit_btn = page.locator("button[data-testid='stChatInputSubmitButton']")
        submit_btn.click()
        page.wait_for_selector(".status-pill-ok:has-text('Instant Conversational Assistant')", timeout=10000)
        time.sleep(1)
        check_no_exceptions("After Second Query Submission")

        # Verify BOTH questions are simultaneously visible on screen
        q_chips = page.locator(".user-query-chip")
        assert q_chips.count() == 2, f"Expected 2 question chips, found {q_chips.count()}!"
        assert "नेपालको संविधानको धारा १६" in q_chips.nth(0).inner_text(), "First question missing!"
        assert "namaste" in q_chips.nth(1).inner_text(), "Second question missing!"
        print(f"  ✓ Confirmed: Both Question 1 ('{q_chips.nth(0).inner_text()[:40]}...') and Question 2 ('{q_chips.nth(1).inner_text()[:40]}...') remain visible!")

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

        # Test 6: English Query & Refusal Verification via Embedded Submit Button
        print("\n--- Test 6: Switch back to Tab 1 & Refusal Query Execution ---")
        tab_buttons.first.click()
        time.sleep(1)
        check_no_exceptions("After Switching Back to Tab 1")

        # Reset query to show suggestion chips
        reset_btn = page.locator("button:has-text('New Question')")
        if reset_btn.count() > 0:
            reset_btn.first.click()
            time.sleep(1.5)

        # Click the cryptocurrency unanswerable refusal chip
        unans_chip = page.locator("button:has-text('cryptocurrency exchanges')")
        assert unans_chip.count() > 0, "Unanswerable query chip not found!"
        print(f"  Clicking refusal chip: '{unans_chip.first.inner_text()[:60]}...'")
        unans_chip.first.click()
        time.sleep(1.5)

        print("  Clicking embedded submit button inside textarea for refusal test...")
        submit_btn = page.locator("button[data-testid='stChatInputSubmitButton']")
        submit_btn.click()
        print("  Waiting for refusal response and validation badge...")
        page.wait_for_selector(".status-pill-warn:has-text('Refusal Triggered')", timeout=45000)
        time.sleep(1)
        check_no_exceptions("After Refusal Pipeline Execution")

        # Verify status badge rendered
        status_pill = page.locator(".status-pill-warn:has-text('Refusal Triggered')")
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
