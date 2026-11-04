"""Native callback dispatch retains behavior without unused event conversion."""

from types import SimpleNamespace
import tkinter as tk
import unittest
from unittest import mock

from aidas.ui.bindings import bind_all_widget_event, bind_native_callback, bind_widget_event, unbind_callback
from aidas.ui.directory_completion import DirectoryCompletion


class RawBindingTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(str(exc))
        self.root.attributes("-alpha", 0)
        self.root.geometry("640x400+0+0")
        self.root.update()
        self.addCleanup(self.root.destroy)

    def test_zero_argument_delivery_break_and_owned_cleanup_preserve_other_handlers(self):
        root = self.root
        called = []
        with mock.patch.object(root, "_substitute", side_effect=AssertionError("Unused event conversion")) as substitute:
            owned = bind_native_callback(root, "<<RawCallbackProbe>>", lambda: called.append("owned") or "break")
        late = root.bind("<<RawCallbackProbe>>", lambda event: called.append("late"), add="+")

        root.event_generate("<<RawCallbackProbe>>", when="now")

        self.assertEqual(called, ["owned"])
        substitute.assert_not_called()
        script = root.bind("<<RawCallbackProbe>>")
        legacy_tk = mock.Mock()
        legacy_tk.call.return_value = script
        legacy = SimpleNamespace(_w=root._w, tk=legacy_tk, deletecommand=mock.Mock())
        unbind_callback(legacy, "<<RawCallbackProbe>>", owned)
        retained = legacy_tk.call.call_args.args[-1]
        self.assertIn(late, retained)
        self.assertNotIn(owned, retained)
        legacy.deletecommand.assert_called_once_with(owned)

        unbind_callback(root, "<<RawCallbackProbe>>", owned)
        root.event_generate("<<RawCallbackProbe>>", when="now")
        self.assertEqual(called, ["owned", "late"])
        self.assertEqual(root.tk.call("info", "commands", owned), "")
        unbind_callback(root, "<<RawCallbackProbe>>", late)

    def test_global_widget_delivery_resolves_the_original_widget_without_full_event_fields(self):
        root = self.root
        entry = tk.Entry(root)
        entry.pack()
        root.update()
        received = []
        with mock.patch.object(root, "_substitute", side_effect=AssertionError("Unused event conversion")) as substitute:
            owned = bind_all_widget_event(root, "<<WidgetCallbackProbe>>", lambda event: received.append(event.widget))

        entry.event_generate("<<WidgetCallbackProbe>>", when="now")

        self.assertEqual(received, [entry])
        substitute.assert_not_called()
        unbind_callback(root, "<<WidgetCallbackProbe>>", owned, bindtag="all")
        entry.event_generate("<<WidgetCallbackProbe>>", when="now")
        self.assertEqual(received, [entry])
        self.assertEqual(root.tk.call("info", "commands", owned), "")

    def test_directory_owner_and_anchor_events_keep_text_and_cleanup_without_converting_siblings(self):
        root = self.root
        holder = tk.Frame(root)
        holder.pack(fill="x")
        entry = tk.Entry(holder)
        entry.pack(fill="x")
        entry.insert(0, "C:/Données/分析")
        sibling = tk.Label(root, text="Other content")
        sibling.pack()
        root.update()
        with mock.patch.object(root, "_substitute", side_effect=AssertionError("Unused event conversion")) as substitute:
            completion = DirectoryCompletion(entry)
        self.addCleanup(completion.destroy)
        completion.hide = mock.Mock(wraps=completion.hide)
        completion._visible = mock.Mock(return_value=True)
        completion._anchor_geometry = completion._entry_geometry()

        sibling.event_generate("<Configure>", when="now")
        sibling.event_generate("<Unmap>", when="now")
        completion.hide.assert_not_called()
        root.event_generate("<Configure>", when="now")
        completion.hide.assert_called_once_with()
        completion.hide.reset_mock()
        completion._anchor_geometry = (0, 0, 0, 0)
        holder.event_generate("<Configure>", when="now")
        completion.hide.assert_called_once_with()
        completion.hide.reset_mock()
        holder.event_generate("<Unmap>", when="now")
        self.assertTrue(completion.hide.called)
        self.assertEqual(entry.get(), "C:/Données/分析")
        substitute.assert_not_called()

        owned = [binding for widget, _event, binding in completion._bindings if widget is root]
        unrelated = mock.Mock()
        extra = root.bind("<Configure>", unrelated, add="+")
        completion.destroy()
        root.event_generate("<Configure>", when="now")
        unrelated.assert_called_once()
        self.assertTrue(all(not root.tk.call("info", "commands", binding) for binding in owned))
        unbind_callback(root, "<Configure>", extra)


class WidgetPathFallbackTests(unittest.TestCase):
    def test_unknown_native_widget_keeps_tkinter_string_fallback(self):
        callbacks = []
        received = []
        owner = SimpleNamespace(
            _w=".owner", tk=mock.Mock(),
            register=lambda callback: callbacks.append(callback) or "callback",
            nametowidget=mock.Mock(side_effect=KeyError("native child")),
        )
        bind_widget_event(owner, "<Map>", lambda event: received.append(event.widget))

        callbacks[0](".owner.native-child")

        self.assertEqual(received, [".owner.native-child"])


if __name__ == "__main__":
    unittest.main()
