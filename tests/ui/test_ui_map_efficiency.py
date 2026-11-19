import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import customtkinter as ctk

from aidas.app import AIDaSApp


class MapEfficiencyTests(unittest.TestCase):
    def test_only_root_remap_resumes_hidden_window_paints(self):
        app = AIDaSApp.__new__(AIDaSApp)
        app._startup_finished = True
        app._aidas_defer_canvas_updates = False
        surface = Mock()
        app._aidas_deferred_surfaces = {surface}
        app._on_window_unmapped(".child")
        self.assertFalse(app._aidas_defer_canvas_updates)
        app._on_window_unmapped(app._w)
        app._on_window_unmapped(app._w)
        self.assertTrue(app._aidas_defer_canvas_updates)
        app._on_interface_widget_mapped(SimpleNamespace(widget=object()))
        self.assertTrue(app._aidas_defer_canvas_updates)
        with patch("aidas.ui.rendering._schedule") as schedule:
            app._on_interface_widget_mapped(SimpleNamespace(widget=app))
            app._on_interface_widget_mapped(SimpleNamespace(widget=app))
            schedule.assert_called_once_with(surface)
        self.assertFalse(app._aidas_deferred_surfaces)
        self.assertFalse(app._aidas_defer_canvas_updates)

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
