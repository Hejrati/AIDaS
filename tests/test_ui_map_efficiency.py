import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import customtkinter as ctk

from aidas.app import AIDaSApp


class MapEfficiencyTests(unittest.TestCase):
    def test_remapped_control_is_settled_once_per_interface(self):
        app = AIDaSApp.__new__(AIDaSApp)
        app.interface_mode = "Classic"
        control = Mock(spec=ctk.CTkBaseClass)
        event = SimpleNamespace(widget=control)
        with patch("aidas.app.refresh_interface_widgets") as refresh:
            app._on_interface_widget_mapped(event)
            app._on_interface_widget_mapped(event)
            refresh.assert_called_once_with(control, recursive=False)
            app.interface_mode = "Modern"
            app._on_interface_widget_mapped(event)
            self.assertEqual(refresh.call_count, 2)

    def test_native_map_does_not_scan_application(self):
        app = AIDaSApp.__new__(AIDaSApp)
        with patch.object(AIDaSApp, "_queue_interface_widget_refresh") as refresh:
            app._on_interface_widget_mapped(SimpleNamespace(widget=object()))
            refresh.assert_not_called()


if __name__ == "__main__":
    unittest.main()
