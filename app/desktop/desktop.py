# ruff: noqa: E402
from app.desktop.studio_server.setup_certs import setup_certs  # isort:skip

# setup_certs must run before imports to register root certs
setup_certs()

import contextlib
import logging
import os
import sys
import tkinter as tk
import webbrowser

import sentry_sdk
from kiln_ai.utils.config import Config
from PIL import Image

# Unused, but needed for pyinstaller to not miss this import
from pydantic.deprecated.decorator import deprecated  # noqa # type: ignore
from uvicorn import Config as UvicornConfig

from app.desktop.custom_tray import KilnMenuItem, KilnTray
from app.desktop.desktop_server import ThreadedServer, server_config
from app.desktop.studio_server._sentry_config import (
    SENTRY_DSN,
    SENTRY_ENV,
    SENTRY_RELEASE,
)
from app.desktop.studio_server._version import __version__
from app.desktop.util.resource_limits import setup_resource_limits

logger = logging.getLogger(__name__)

# Set writeable cache directories as soon as we start
os.environ["LLAMA_INDEX_CACHE_DIR"] = os.path.join(
    Config.settings_dir(), "cache", "llama_index_cache"
)
os.environ["NLTK_DATA"] = os.path.join(Config.settings_dir(), "cache", "nltk_data")


class DesktopApp:
    def __init__(self, port: int):
        self.port = port
        # TK without a window, to get dock events on MacOS
        self.root = tk.Tk()
        self.root.title("Kiln")
        self.root.withdraw()
        self.tray: KilnTray | None = None

    def start(self):
        """
        Start the desktop app, showing the web app, tray, and closing the splash screen.
        """

        # Register a callback for the dock icon (MacOS) to reopen the web app
        def _show_studio():
            self.show_studio()

        self.root.createcommand("tk::mac::ReopenApplication", _show_studio)

        # Run the tray
        self.run_tray()

        # Show the web app after a short delay, to avoid race with the server starting
        self.root.after(200, self.show_studio)
        self.root.after(200, self.close_splash)

        # Run the main loop until the app is quit (quit menu item usually)
        self.root.mainloop()

    def quit_app(self):
        """
        Shutdown the app (taskbar, and desktop tk app)
        """

        if self.tray is not None:
            self.tray.stop()
        if self.root is not None:
            self.root.destroy()

    def on_quit(self):
        """
        Quit event handler. Will dispatch the shutdown to the most appropriate place (main loop ideally)
        """

        # use tk mainloop if possible
        if self.root:
            self.root.after(100, self.quit_app)
        else:
            self.quit_app()

    def show_studio(self):
        """
        Open the web app in the default browser.
        """

        webbrowser.open(f"http://localhost:{self.port}")

    def resource_path(self, relative_path):
        """
        Get the path to the resource files (webapp, taskbar icon, etc)
        """

        meipass = getattr(sys, "_MEIPASS", None)
        if meipass and isinstance(meipass, str):
            base_path = meipass
        else:
            base_path = os.path.dirname(__file__)

        return os.path.join(base_path, relative_path)

    def run_tray(self):
        """
        Run the tray with menu items for quit and open studio.
        """

        if self.tray is not None:
            return

        tray_image = Image.open(self.resource_path("taskbar.png"))

        # taskbar.png is sized for macOS/Windows; Linux trays typically render icons
        # at ~22-24px, so the source image looks oversized/blurry there unless scaled down.
        if sys.platform.startswith("linux"):
            tray_image = tray_image.resize((24, 24), Image.Resampling.LANCZOS)

        # Use default on Windows to get "left click to open" behaviour.
        # It looks ugly on MacOS (just a bold effect Apple never uses), so don't use it there
        make_open_studio_default = sys.platform in ("win32", "Windows")

        menu = (
            KilnMenuItem(
                "Open Kiln Studio", self.show_studio, default=make_open_studio_default
            ),
            KilnMenuItem("Quit", self.on_quit),
        )

        self.tray = KilnTray("kiln", tray_image, "Kiln", menu)

        try:
            # running detached since we use tk mainloop to get events from dock icon
            self.tray.run_detached()
        except Exception:
            logger.error("Error running tray", exc_info=True)
            # Tray not starting on MacOS or Windows is critical.
            # Let Linux continue to start the app as tray is more fragmented there and requires system deps.
            if sys.platform in ["darwin", "win32"]:
                raise
            else:
                self.tray = None

    def close_splash(self):
        try:
            import pyi_splash  # type: ignore

            pyi_splash.close()
        except ModuleNotFoundError:
            pass


class DesktopServer(ThreadedServer):
    """
    A wrapper around the threaded app server which runs in a thread, but also shuts down the app when the server stops.
    """

    def __init__(self, app: DesktopApp, config: UvicornConfig):
        self.app = app
        super().__init__(config=config)

    @contextlib.contextmanager
    def run_in_thread(self):
        try:
            with super().run_in_thread():
                yield
        finally:
            self.app.on_quit()


def desktop_release_name() -> str:
    if SENTRY_RELEASE:
        return SENTRY_RELEASE
    if not __version__:
        logger.warning("__version__ is not set, using unknown version")
        return "kiln-studio-desktop@unknown"
    return f"kiln-studio-desktop@{__version__}"


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    multiprocessing.set_start_method("spawn", force=True)

    # Sentry is gated on DSN presence: CI bakes it into _sentry_config.py for
    # release builds, dev/test builds leave it None and init is skipped.
    if SENTRY_DSN:
        sentry_sdk.init(
            dsn=SENTRY_DSN,
            release=desktop_release_name(),
            environment=SENTRY_ENV,
            send_default_pii=False,
            traces_sample_rate=1.0,
        )

    setup_resource_limits()

    host = Config.shared().kiln_local_api_host
    port = Config.shared().kiln_local_api_port
    app = DesktopApp(port=port)

    # Create and run the server
    # run the server in a thread, and shut down server when main thread (tk mainloop) exits
    config = server_config(port=port, host=host, tk_root=app.root)
    uni_server = DesktopServer(app=app, config=config)
    with uni_server.run_in_thread():
        if not uni_server.running():
            # Can't start. Likely the port is already in use (app already running). Show the existing web app and exit.
            app.show_studio()
            app.on_quit()

        # start the desktop app once the server is running. It will keep running until the tk mainloop exits (quit menu item usually)
        app.start()
