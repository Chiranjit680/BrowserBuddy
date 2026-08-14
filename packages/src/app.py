import threading

import pystray
from PIL import Image, ImageDraw

import browser_track

_tracking_thread: threading.Thread | None = None
_stop_event: threading.Event | None = None
_tracking_lock = threading.Lock()


def create_icon_image():
    width = 64
    height = 64
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width // 2, height // 2), fill="black")
    draw.rectangle((width // 2, height // 2, width, height), fill="black")
    return image


def start_tracking() -> None:
    global _tracking_thread, _stop_event
    with _tracking_lock:
        if _tracking_thread is not None and _tracking_thread.is_alive():
            print("BrowseGuard tracking already running")
            return
        _stop_event = threading.Event()
        _tracking_thread = threading.Thread(
            target=browser_track.run_tracking, args=(_stop_event,), daemon=True
        )
        _tracking_thread.start()


def stop_tracking() -> None:
    global _tracking_thread, _stop_event
    with _tracking_lock:
        if _stop_event is not None:
            _stop_event.set()
        if _tracking_thread is not None:
            _tracking_thread.join(timeout=5)
        _tracking_thread = None
        _stop_event = None


def on_show(icon, item):
    print("BrowseGuard is running")
    start_tracking()


def on_quit(icon, item):
    stop_tracking()
    icon.stop()


def main():
    menu = pystray.Menu(
        pystray.MenuItem("Show", on_show, default=True),
        pystray.MenuItem("Quit", on_quit),
    )
    icon = pystray.Icon("BrowseGuard", create_icon_image(), "BrowseGuard", menu)
    icon.run()


if __name__ == "__main__":
    main()
