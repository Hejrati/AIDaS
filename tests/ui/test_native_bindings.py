from __future__ import annotations

import tkinter as tk
from types import SimpleNamespace
import unittest
from unittest import mock

from aidas.ui.bindings import unbind_callback


class NativeBindingTests(unittest.TestCase):
    def test_newer_tkinter_removes_only_the_requested_registration(self):
        widget = SimpleNamespace(_w=".owner", _unbind=mock.Mock())
        unbind_callback(widget, "<Destroy>", "123callback")
        widget._unbind.assert_called_once_with(("bind", ".owner", "<Destroy>"), "123callback")

    def test_older_tkinter_keeps_other_python_and_tcl_handlers(self):
        script = ('if {"[123keep %W]" == "break"} break\n'
                  'if {"[123remove %W]" == "break"} break\n'
                  'if {"[123remove_suffix %W]" == "break"} break\n'
                  'set ::native_handler 1\n')
        interpreter = mock.Mock()
        interpreter.call.return_value = script
        widget = SimpleNamespace(_w=".owner", tk=interpreter, deletecommand=mock.Mock())
        unbind_callback(widget, "<Configure>", "123remove")
        retained = script.replace('if {"[123remove %W]" == "break"} break\n', '')
        self.assertEqual(interpreter.call.call_args_list, [
            mock.call("bind", ".owner", "<Configure>"),
            mock.call("bind", ".owner", "<Configure>", retained),
        ])
        widget.deletecommand.assert_called_once_with("123remove")

    def test_global_registration_and_empty_callback_id(self):
        interpreter = mock.Mock()
        interpreter.call.return_value = 'if {"[123remove %W]" == "break"} break\n'
        widget = SimpleNamespace(_w=".owner", tk=interpreter, deletecommand=mock.Mock())
        unbind_callback(widget, "<MouseWheel>", "123remove", bindtag="all")
        interpreter.call.assert_called_with("bind", "all", "<MouseWheel>", "")
        interpreter.reset_mock()
        unbind_callback(widget, "<MouseWheel>", None)
        interpreter.call.assert_not_called()

    def test_real_native_widget_retains_unrelated_callback_after_cleanup(self):
        try:
            root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(str(exc))
        try:
            root.withdraw()
            called = []
            first = root.bind("<<OwnedBindingTest>>", lambda event: called.append("first"), add="+")
            removed = root.bind("<<OwnedBindingTest>>", lambda event: called.append("removed"), add="+")
            last = root.bind("<<OwnedBindingTest>>", lambda event: called.append("last"), add="+")
            unbind_callback(root, "<<OwnedBindingTest>>", removed)
            script = root.bind("<<OwnedBindingTest>>")
            self.assertIn(first, script)
            self.assertIn(last, script)
            self.assertNotIn(removed, script)
            root.update()
            root.event_generate("<<OwnedBindingTest>>")
            self.assertEqual(called, ["first", "last"])
            self.assertEqual(root.tk.call("info", "commands", removed), "")
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
