"""
Tracks tabs/navigation in an already-running Chrome instance via CDP.

Chrome must be launched with remote debugging enabled, e.g.:
    chrome.exe --remote-debugging-port=9222

Playwright then attaches to that instance instead of launching a new one.
"""

import logging
import os
import queue
import subprocess
import sys
import threading
import time
from typing import Optional
from weakref import WeakSet
from dotenv import load_dotenv

load_dotenv()

import requests
from playwright.sync_api import sync_playwright, Page, BrowserContext

import llm_client

logger = logging.getLogger(__name__)

CHROME_DEBUG_PORT = 9222
CDP_URL = f"http://localhost:{CHROME_DEBUG_PORT}"

_DEFAULT_CHROME_PATHS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]
_chrome_paths_env = os.environ.get("CHROME_PATHS")
CHROME_PATHS = (
    [p.strip() for p in _chrome_paths_env.split(",") if p.strip()]
    if _chrome_paths_env
    else _DEFAULT_CHROME_PATHS
)

CHROME_USER_DATA_DIR = os.environ.get(
    "BROWSEGUARD_CHROME_USER_DATA_DIR", r"C:\temp\chrome-debug"
)

ASSISTANT_BINDING = "__browseguardAsk"

ASSISTANT_UI_SCRIPT = f"""
(() => {{
  const install = () => {{
    // Presence of the button is the idempotency guard: a window flag would
    // stay set even if a run bailed out early, blocking every later retry.
    if (document.getElementById('browseguard-ask-button')) return;
    const root = document.body || document.documentElement;
    if (!root) return;

    const button = document.createElement('button');
    button.id = 'browseguard-ask-button';
    button.textContent = 'Ask Assistant';
    Object.assign(button.style, {{
      position: 'fixed', bottom: '20px', right: '20px', zIndex: 2147483647,
      padding: '10px 16px', borderRadius: '999px', border: 'none',
      background: '#2563eb', color: '#fff', fontFamily: 'system-ui, sans-serif',
      fontSize: '14px', fontWeight: '600', cursor: 'pointer',
      boxShadow: '0 2px 8px rgba(0,0,0,0.25)',
    }});

    const panel = document.createElement('div');
    panel.id = 'browseguard-ask-panel';
    Object.assign(panel.style, {{
      position: 'fixed', bottom: '70px', right: '20px', zIndex: 2147483647,
      width: '300px', background: '#fff', border: '1px solid #ddd',
      borderRadius: '12px', boxShadow: '0 4px 16px rgba(0,0,0,0.3)',
      padding: '12px', display: 'none', flexDirection: 'column', gap: '8px',
      fontFamily: 'system-ui, sans-serif',
    }});

    const title = document.createElement('div');
    title.textContent = 'Ask Assistant';
    Object.assign(title.style, {{ fontWeight: '600', fontSize: '14px', color: '#111' }});

    const results = document.createElement('div');
    results.id = 'browseguard-ask-results';
    Object.assign(results.style, {{
      display: 'none', maxHeight: '200px', overflowY: 'auto',
      fontSize: '12px', lineHeight: '1.4', color: '#222',
      background: '#f8f9fa', border: '1px solid #eee', borderRadius: '6px',
      padding: '8px', whiteSpace: 'pre-wrap', wordBreak: 'break-word',
    }});

    const input = document.createElement('input');
    input.type = 'text';
    input.placeholder = 'Type your question...';
    Object.assign(input.style, {{
      padding: '8px', borderRadius: '6px', border: '1px solid #ccc', fontSize: '13px',
    }});

    const sendBtn = document.createElement('button');
    sendBtn.textContent = 'Send';
    Object.assign(sendBtn.style, {{
      padding: '8px', borderRadius: '6px', border: 'none',
      background: '#2563eb', color: '#fff', fontSize: '13px', cursor: 'pointer',
    }});

    // innerText reflects what is rendered, so the widget's own text would
    // otherwise show up as part of the page content. Hide it while reading;
    // JS is synchronous here, so no repaint happens in between.
    const readPageText = () => {{
      if (!document.body) return '';
      const prevButton = button.style.display;
      const prevPanel = panel.style.display;
      button.style.display = 'none';
      panel.style.display = 'none';
      const text = document.body.innerText;
      button.style.display = prevButton;
      panel.style.display = prevPanel;
      return text;
    }};

    const send = async () => {{
      const message = input.value.trim();
      if (!message) return;

      input.value = '';
      results.textContent = 'Thinking...';
      results.style.display = 'block';

      if (!window.{ASSISTANT_BINDING}) {{
        results.textContent = 'Assistant is not available on this page.';
        return;
      }}

      try {{
        // The page text is gathered here and passed along, because the Python
        // side cannot call back into this page to read it (see the binding).
        const answer = await window.{ASSISTANT_BINDING}({{
          message: message,
          pageText: readPageText(),
        }});
        results.textContent = answer || '(no response)';
      }} catch (err) {{
        results.textContent = 'Error: ' + (err && err.message ? err.message : err);
      }}
    }};

    sendBtn.addEventListener('click', send);
    input.addEventListener('keydown', (e) => {{ if (e.key === 'Enter') send(); }});

    button.addEventListener('click', () => {{
      const isHidden = panel.style.display === 'none';
      panel.style.display = isHidden ? 'flex' : 'none';
      if (isHidden) input.focus();
    }});

    panel.append(title, results, input, sendBtn);
    root.append(button, panel);
  }};

  // On every navigation this script runs at document-start, when there is no
  // <body> yet (and often no documentElement) — so wait for the DOM.
  if (document.readyState === 'loading') {{
    document.addEventListener('DOMContentLoaded', install, {{ once: true }});
  }} else {{
    install();
  }}
}})();
"""

# Pages that already have the binding + init script installed, so
# re-tracking (e.g. re-running track_context) doesn't double-install.
_ui_installed_pages: "WeakSet[Page]" = WeakSet()


def describe(page: Page) -> str:
    try:
        return f"{page.title()!r} - {page.url}"
    except Exception:
        return f"<page closed> - {page.url}"


def extract_page_content(page: Page) -> str:
    """Return the page's visible text, excluding BrowseGuard's own widget."""
    try:
        return page.evaluate("""() => {
            if (!document.body) return '';
            const els = ['browseguard-ask-button', 'browseguard-ask-panel']
                .map(id => document.getElementById(id))
                .filter(Boolean);
            const prev = els.map(el => el.style.display);
            els.forEach(el => { el.style.display = 'none'; });
            const text = document.body.innerText;
            els.forEach((el, i) => { el.style.display = prev[i]; });
            return text;
        }""")
    except Exception:
        return ""


_WIDGET_IDS = ["browseguard-ask-button", "browseguard-ask-panel"]

# Index-aligned with the ids passed in; null marks an element that wasn't there,
# so restoring can skip it rather than inventing a display value.
_HIDE_WIDGET_JS = """(ids) => ids.map(id => {
    const el = document.getElementById(id);
    if (!el) return null;
    const prev = el.style.display;
    el.style.display = 'none';
    return prev;
})"""

_RESTORE_WIDGET_JS = """([ids, prev]) => {
    ids.forEach((id, i) => {
        const el = document.getElementById(id);
        if (el && prev[i] !== null) el.style.display = prev[i];
    });
}"""


def capture_page_screenshot(
    page: Page,
    path: Optional[str] = None,
    full_page: bool = False,
    wait_for_load: bool = True,
    timeout: float = 10_000,
) -> bytes:
    """Return a PNG screenshot of the page, excluding BrowseGuard's own widget.

    Also writes the PNG to `path` when given. Returns b"" if the page is
    closed or the capture fails, mirroring extract_page_content.

    Must run on the thread that owns the Playwright connection (the one
    running run_tracking). Calling it from an expose_binding callback
    deadlocks the connection, for the reason documented in
    inject_assistant_ui.
    """
    if wait_for_load:
        # Best effort: a page that never fires 'load' (long-polling, streaming)
        # is still worth screenshotting as-is.
        try:
            page.wait_for_load_state("load", timeout=timeout)
        except Exception:
            logger.debug("Load state wait timed out for %s", page.url)

    try:
        prev = page.evaluate(_HIDE_WIDGET_JS, _WIDGET_IDS)
    except Exception:
        logger.exception("Could not hide widget before screenshot")
        return b""

    try:
        return page.screenshot(
            path=path, full_page=full_page, type="png", timeout=timeout
        )
    except Exception:
        logger.exception("Screenshot failed for %s", page.url)
        return b""
    finally:
        # Unlike extract_page_content, hide and restore are separate round
        # trips with a capture in between, so the restore has to be in a
        # finally or a failed capture leaves the widget invisible.
        try:
            page.evaluate(_RESTORE_WIDGET_JS, [_WIDGET_IDS, prev])
        except Exception:
            logger.debug("Could not restore widget after screenshot")


def handle_assistant_message(page_url: str, message: str, page_content: str) -> str:
    logger.info("Assistant question on %s: %r", page_url, message)
    logger.info("Received %d chars of page content", len(page_content))

    try:
        logger.info("Calling LLM")
        answer = llm_client.ask(message, page_content)
        logger.info("LLM call succeeded")
        return answer
    except Exception as exc:
        logger.exception("LLM call failed")
        return f"(assistant error: {exc})"


def inject_assistant_ui(page: Page) -> None:
    """Install the 'Ask Assistant' button/panel on this page.

    Uses expose_binding + add_init_script, both of which survive future
    navigations on this page, so this only needs to run once per page.
    """
    if page in _ui_installed_pages:
        return
    _ui_installed_pages.add(page)

    # This callback must never call back into Playwright (page.evaluate etc.):
    # the sync API dispatches binding callbacks on the same loop that services
    # API calls, so a reentrant call deadlocks the whole connection and stops
    # all further page/navigation events. The page text is therefore collected
    # in JS and passed in as an argument instead.
    page.expose_binding(
        ASSISTANT_BINDING,
        lambda source, payload: handle_assistant_message(
            source["page"].url, payload.get("message", ""), payload.get("pageText", "")
        ),
    )
    page.add_init_script(script=ASSISTANT_UI_SCRIPT)

    # add_init_script only affects future navigations; inject into the
    # page's current document too, in case it's already loaded.
    try:
        page.evaluate(ASSISTANT_UI_SCRIPT)
    except Exception:
        pass


def track_page(page: Page) -> None:
    inject_assistant_ui(page)
    page.on("framenavigated", lambda frame: (
        print(f"[navigated] {describe(page)}") if frame == page.main_frame else None
    ))
    page.on("close", lambda: print(f"[closed] {page.url}"))


def track_context(context: BrowserContext) -> None:
    for page in context.pages:
        print(f"[tracking] {describe(page)}")
        track_page(page)

    context.on("page", lambda page: (
        print(f"[opened] {describe(page)}"),
        track_page(page),
    ))


def is_cdp_up() -> bool:
    try:
        requests.get(f"{CDP_URL}/json/version", timeout=1)
        return True
    except requests.RequestException:
        return False


def find_chrome_exe() -> str:
    custom = os.environ.get("BROWSEGUARD_CHROME_PATH")
    if custom:
        return custom
    for path in CHROME_PATHS:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "Could not locate chrome.exe. Set BROWSEGUARD_CHROME_PATH to its full path."
    )


def launch_chrome() -> subprocess.Popen:
    chrome_exe = find_chrome_exe()
    os.makedirs(CHROME_USER_DATA_DIR, exist_ok=True)
    print(f"Launching Chrome in debug mode: {chrome_exe}")
    return subprocess.Popen(
        [
            chrome_exe,
            f"--remote-debugging-port={CHROME_DEBUG_PORT}",
            f"--user-data-dir={CHROME_USER_DATA_DIR}",
        ],
        close_fds=True,
    )


def stop_chrome(chrome_process: subprocess.Popen) -> None:
    if chrome_process.poll() is not None:
        return
    print("Stopping Chrome (launched by BrowseGuard)")
    try:
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(chrome_process.pid)],
            capture_output=True,
        )
    except Exception:
        chrome_process.terminate()
    try:
        chrome_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def wait_for_cdp(timeout: float = 20.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if is_cdp_up():
            return
        time.sleep(0.5)
    raise TimeoutError(f"Chrome did not become available at {CDP_URL} within {timeout}s")


# ---------------------------------------------------------------------------
# Cross-thread command channel
#
# Playwright's sync API is owned by the one thread that created it (here, the
# thread running run_tracking), and it only dispatches events while one of its
# own calls is in flight. So no other thread — a LangGraph tool call, the
# PySide panel, an HTTP handler — may touch a Page directly.
#
# Instead they hand a callable to submit_browser_command(), and the tracking
# loop runs it between its own idle ticks and hands the result back.
# ---------------------------------------------------------------------------

_command_queue: "queue.Queue[tuple]" = queue.Queue()
_tracking_active = threading.Event()


class TrackingNotRunning(RuntimeError):
    """Raised when a browser command is submitted with no tracking loop up."""


def is_tracking_active() -> bool:
    return _tracking_active.is_set()


def submit_browser_command(fn, timeout: float = 30.0):
    """Run fn(browser) on the tracking thread and return whatever it returns.

    Safe to call from any thread *except* the tracking thread itself, which
    would wait on a queue only it can drain. Exceptions raised by fn are
    re-raised here, on the caller's thread.
    """
    if not _tracking_active.is_set():
        raise TrackingNotRunning(
            "BrowseGuard is not attached to Chrome; start tracking first."
        )

    done = threading.Event()
    box: dict = {}
    _command_queue.put((fn, box, done))

    if not done.wait(timeout):
        raise TimeoutError(f"Browser command did not complete within {timeout}s")
    if "error" in box:
        raise box["error"]
    return box["result"]


def _drain_command_queue(browser) -> None:
    """Run every queued command once, on the tracking thread."""
    while True:
        try:
            fn, box, done = _command_queue.get_nowait()
        except queue.Empty:
            return
        try:
            box["result"] = fn(browser)
        except Exception as exc:
            logger.exception("Browser command failed")
            box["error"] = exc
        finally:
            done.set()


def _fail_pending_commands() -> None:
    """Unblock anyone still waiting once the loop is on its way out."""
    while True:
        try:
            _fn, box, done = _command_queue.get_nowait()
        except queue.Empty:
            return
        box["error"] = TrackingNotRunning("Tracking stopped before command ran")
        done.set()


def run_tracking(stop_event: Optional[threading.Event] = None) -> bool:
    """Launch/attach to Chrome over CDP and track it until stopped.

    Returns True if tracking ran (even if it later stopped/disconnected),
    False if it couldn't start at all. Safe to call from a background
    thread. If stop_event is provided, setting it ends the tracking loop.

    If this call launches Chrome itself (it wasn't already running in
    debug mode), that Chrome process is killed when tracking ends,
    whether because the browser was closed or stop_event was set —
    so BrowseGuard never leaves an orphaned Chrome behind.
    """
    chrome_process: Optional[subprocess.Popen] = None
    if not is_cdp_up():
        try:
            chrome_process = launch_chrome()
            wait_for_cdp()
        except (FileNotFoundError, TimeoutError) as exc:
            print(exc)
            if chrome_process is not None:
                stop_chrome(chrome_process)
            return False

    try:
        with sync_playwright() as playwright:
            try:
                browser = playwright.chromium.connect_over_cdp(CDP_URL)
            except Exception as exc:
                print(f"Could not connect to Chrome at {CDP_URL}: {exc}")
                print("Start Chrome with: chrome.exe --remote-debugging-port=9222")
                return False

            print(f"Connected to Chrome at {CDP_URL}")

            for context in browser.contexts:
                track_context(context)

            browser.on("disconnected", lambda: print("Chrome connection closed"))

            _tracking_active.set()

            # Note: browser.is_connected()/browser.contexts reflect Playwright's
            # last-known state, which can go stale forever if Chrome's process
            # dies abruptly (no clean CDP teardown ever arrives to update it).
            # is_cdp_up() makes a fresh network probe each time, so it's the
            # reliable signal for "is Chrome still actually there".
            try:
                while True:
                    if stop_event is not None and stop_event.is_set():
                        break
                    if not is_cdp_up():
                        print("Chrome is no longer reachable")
                        break
                    try:
                        pages = [pg for ctx in browser.contexts for pg in ctx.pages]
                    except Exception:
                        break
                    if not pages:
                        print("All Chrome windows closed")
                        break

                    # Other threads' work runs here, on the thread that owns
                    # the Playwright connection. Worst-case latency for a
                    # queued command is one tick of the wait below.
                    _drain_command_queue(browser)

                    # Idle *inside* Playwright, never with time.sleep(). The
                    # sync API only dispatches incoming events while one of its
                    # own calls is in flight, so a plain sleep here freezes the
                    # dispatcher: navigations, new tabs and expose_binding
                    # callbacks all queue up undelivered until the next call.
                    try:
                        pages[0].wait_for_timeout(1000)
                    except Exception:
                        time.sleep(1)
            except KeyboardInterrupt:
                pass
            finally:
                # Clear before the connection closes, so no command is
                # accepted that would run against a dead browser.
                _tracking_active.clear()
                _fail_pending_commands()
    finally:
        _tracking_active.clear()
        _fail_pending_commands()
        if chrome_process is not None:
            stop_chrome(chrome_process)

    return True


def main() -> None:
    ok = run_tracking()
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
