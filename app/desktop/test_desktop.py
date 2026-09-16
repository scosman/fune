import os
import random
import sys
from unittest.mock import Mock, patch

import pytest
import requests
from PIL import Image
from uvicorn import Config as UvicornConfig

import app.desktop.desktop_server as desktop_server
from app.desktop.desktop import DesktopApp, DesktopServer

TEST_PORT = 8123


@pytest.fixture(autouse=True)
def mock_gui_modules():
    """Mock GUI modules globally to prevent display errors in headless CI."""
    with patch("app.desktop.desktop.tk.Tk"):
        yield


@pytest.fixture
def mock_tk_root():
    """Mock tkinter root window."""
    with patch("app.desktop.desktop.tk.Tk") as mock_tk:
        mock_root = Mock()
        mock_tk.return_value = mock_root
        yield mock_root


@pytest.fixture
def mock_image():
    """Mock PIL Image."""
    with patch("app.desktop.desktop.Image.open") as mock_image_open:
        mock_image_obj = Mock()
        mock_image_open.return_value = mock_image_obj
        yield mock_image_obj


@pytest.fixture
def mock_kiln_tray():
    """Mock KilnTray."""
    with patch("app.desktop.desktop.KilnTray") as mock_tray_class:
        mock_tray = Mock()
        mock_tray_class.return_value = mock_tray
        yield mock_tray


@pytest.fixture
def mock_webbrowser():
    """Mock webbrowser module."""
    with patch("app.desktop.desktop.webbrowser") as mock_wb:
        yield mock_wb


@pytest.fixture
def mock_kiln_menu_item():
    """Mock pystray module."""
    with patch("app.desktop.desktop.KilnMenuItem") as mock_menu_item:
        yield mock_menu_item


class TestDesktopApp:
    """Test the DesktopApp class."""

    def test_init(self, mock_tk_root):
        """Test DesktopApp initialization."""
        app = DesktopApp(port=TEST_PORT)

        assert app.root == mock_tk_root
        assert app.tray is None
        mock_tk_root.title.assert_called_once_with("Kiln")
        mock_tk_root.withdraw.assert_called_once()

    def test_start(self, mock_tk_root, mock_kiln_tray):
        """Test app start method."""
        app = DesktopApp(port=TEST_PORT)

        # Mock the mainloop to prevent hanging
        mock_tk_root.mainloop = Mock()

        with patch.object(app, "run_tray") as mock_run_tray:
            app.start()

            # Verify dock callback is registered
            mock_tk_root.createcommand.assert_called_once()
            command_name, _ = mock_tk_root.createcommand.call_args[0]
            assert command_name == "tk::mac::ReopenApplication"

            # Verify tray is started
            mock_run_tray.assert_called_once()

            # Verify scheduled callbacks
            assert mock_tk_root.after.call_count == 2
            after_calls = mock_tk_root.after.call_args_list
            assert after_calls[0][0] == (200, app.show_studio)
            assert after_calls[1][0] == (200, app.close_splash)

            # Verify mainloop is called
            mock_tk_root.mainloop.assert_called_once()

    def test_quit_app_with_tray_and_root(self, mock_tk_root, mock_kiln_tray):
        """Test quit_app when both tray and root exist."""
        app = DesktopApp(port=TEST_PORT)
        app.tray = mock_kiln_tray

        app.quit_app()

        mock_kiln_tray.stop.assert_called_once()
        mock_tk_root.destroy.assert_called_once()

    def test_quit_app_no_tray(self, mock_tk_root):
        """Test quit_app when tray is None."""
        app = DesktopApp(port=TEST_PORT)
        app.tray = None

        app.quit_app()

        # Should still call root.destroy
        mock_tk_root.destroy.assert_called_once()

    def test_on_quit_with_root(self, mock_tk_root):
        """Test on_quit when root exists."""
        app = DesktopApp(port=TEST_PORT)

        with patch.object(app, "quit_app") as mock_quit_app:
            app.on_quit()

            # Should schedule quit_app via tk.after
            mock_tk_root.after.assert_called_once_with(100, mock_quit_app)

    def test_on_quit_without_root(self, mock_tk_root):
        """Test on_quit when root is None."""
        app = DesktopApp(port=TEST_PORT)

        with patch.object(app, "quit_app") as mock_quit_app:
            with patch.object(app, "root", None):
                app.on_quit()

                # Should call quit_app directly
                mock_quit_app.assert_called_once()

    def test_show_studio(self, mock_tk_root, mock_webbrowser):
        """Test show_studio opens the correct URL."""
        app = DesktopApp(port=TEST_PORT)

        app.show_studio()

        mock_webbrowser.open.assert_called_once_with(f"http://localhost:{TEST_PORT}")

    def test_resource_path_with_meipass(self, mock_tk_root):
        """Test resource_path when running from PyInstaller bundle."""
        app = DesktopApp(port=TEST_PORT)

        with patch.object(sys, "_MEIPASS", "/bundled/path", create=True):
            result = app.resource_path("icon.png")

            expected = os.path.join("/bundled/path", "icon.png")
            assert result == expected

    def test_resource_path_without_meipass(self, mock_tk_root):
        """Test resource_path when running from source."""
        app = DesktopApp(port=TEST_PORT)

        # Ensure _MEIPASS doesn't exist
        if hasattr(sys, "_MEIPASS"):
            delattr(sys, "_MEIPASS")

        result = app.resource_path("icon.png")

        # Since we're testing, it will use the test file's directory
        assert result.endswith("icon.png")

    def test_run_tray_first_time(
        self, mock_tk_root, mock_image, mock_kiln_tray, mock_kiln_menu_item
    ):
        """Test run_tray when tray doesn't exist yet."""
        app = DesktopApp(port=TEST_PORT)

        with patch.object(app, "resource_path", return_value="taskbar.png"):
            app.run_tray()

            # Verify menu items are created
            assert mock_kiln_menu_item.call_count == 2
            menu_calls = mock_kiln_menu_item.call_args_list

            # First menu item (Open Kiln Studio)
            assert menu_calls[0][0][0] == "Open Kiln Studio"
            assert menu_calls[0][0][1] == app.show_studio

            # Second menu item (Quit)
            assert menu_calls[1][0][0] == "Quit"
            assert menu_calls[1][0][1] == app.on_quit

            # Verify tray is created and started
            assert app.tray == mock_kiln_tray
            mock_kiln_tray.run_detached.assert_called_once()

    def test_run_tray_already_exists(self, mock_tk_root, mock_kiln_tray):
        """Test run_tray when tray already exists."""
        app = DesktopApp(port=TEST_PORT)
        app.tray = mock_kiln_tray

        with patch.object(app, "resource_path") as mock_resource_path:
            app.run_tray()

            # Should return early without creating new tray
            mock_resource_path.assert_not_called()

    @patch("app.desktop.desktop.sys.platform", "win32")
    def test_run_tray_windows_default(
        self, mock_tk_root, mock_image, mock_kiln_tray, mock_kiln_menu_item
    ):
        """Test run_tray sets default=True on Windows."""
        app = DesktopApp(port=TEST_PORT)

        with patch.object(app, "resource_path", return_value="taskbar.png"):
            app.run_tray()

            # Check first menu item has default=True
            menu_calls = mock_kiln_menu_item.call_args_list
            assert menu_calls[0][1]["default"] is True

    @patch("app.desktop.desktop.sys.platform", "darwin")
    def test_run_tray_macos_no_default(
        self, mock_tk_root, mock_image, mock_kiln_tray, mock_kiln_menu_item
    ):
        """Test run_tray sets default=False on macOS."""
        app = DesktopApp(port=TEST_PORT)

        with patch.object(app, "resource_path", return_value="taskbar.png"):
            app.run_tray()

            # Check first menu item has default=False
            menu_calls = mock_kiln_menu_item.call_args_list
            assert menu_calls[0][1]["default"] is False

    @patch("app.desktop.desktop.sys.platform", "linux")
    def test_run_tray_linux_icon_scaling(
        self, mock_tk_root, mock_image, mock_kiln_menu_item
    ):
        """Test run_tray scales the tray icon down on Linux."""
        app = DesktopApp(port=TEST_PORT)

        with (
            patch.object(app, "resource_path", return_value="taskbar.png"),
            patch("app.desktop.desktop.KilnTray") as mock_kiln_tray_class,
        ):
            app.run_tray()

            mock_image.resize.assert_called_once()
            mock_image.resize.assert_called_once_with(
                (24, 24), Image.Resampling.LANCZOS
            )
            resized_image = mock_image.resize.return_value
            mock_kiln_tray_class.assert_called_once()
            assert mock_kiln_tray_class.call_args.args[1] is resized_image

    @patch("app.desktop.desktop.sys.platform", "win32")
    def test_run_tray_windows_no_icon_scaling(
        self, mock_tk_root, mock_image, mock_kiln_tray, mock_kiln_menu_item
    ):
        """Test run_tray does not scale the tray icon on Windows."""
        app = DesktopApp(port=TEST_PORT)

        with patch.object(app, "resource_path", return_value="taskbar.png"):
            app.run_tray()

            mock_image.resize.assert_not_called()

    def test_close_splash_with_pyi_splash(self, mock_tk_root):
        """Test close_splash when pyi_splash is available."""
        app = DesktopApp(port=TEST_PORT)

        mock_pyi_splash = Mock()
        with patch.dict("sys.modules", {"pyi_splash": mock_pyi_splash}):
            app.close_splash()

            mock_pyi_splash.close.assert_called_once()

    def test_close_splash_without_pyi_splash(self, mock_tk_root):
        """Test close_splash when pyi_splash is not available."""
        app = DesktopApp(port=TEST_PORT)

        # Should not raise an exception
        app.close_splash()


class TestDesktopServer:
    """Test the DesktopServer class."""

    def test_init(self, mock_tk_root):
        """Test DesktopServer initialization."""
        app = DesktopApp(port=TEST_PORT)
        config = Mock(spec=UvicornConfig)

        server = DesktopServer(app, config)

        assert server.app == app
        assert server.config == config

    def test_run_in_thread_calls_app_on_quit(self, mock_tk_root):
        """Test that run_in_thread calls app.on_quit when context exits."""
        app = DesktopApp(port=TEST_PORT)
        config = Mock(spec=UvicornConfig)
        server = DesktopServer(app, config)

        with patch.object(app, "on_quit") as mock_on_quit:
            with patch(
                "app.desktop.desktop.ThreadedServer.run_in_thread"
            ) as mock_super_run:
                # Mock the super context manager
                mock_context = Mock()
                mock_super_run.return_value.__enter__ = Mock(return_value=mock_context)
                mock_super_run.return_value.__exit__ = Mock(return_value=None)

                with server.run_in_thread():
                    pass

                mock_on_quit.assert_called_once()


def test_desktop_app_server():
    """Test the desktop app server integration (existing test)."""
    # random port between 9000 and 12000
    port = random.randint(9000, 12000)
    config = desktop_server.server_config(port=port, host="127.0.0.1")
    uni_server = desktop_server.ThreadedServer(config=config)
    with uni_server.run_in_thread():
        r = requests.get("http://127.0.0.1:{}/ping".format(port))
        assert r.status_code == 200
