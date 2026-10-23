from __future__ import annotations

from pathlib import Path
import tempfile
import threading
import time
import tkinter as tk
from tkinter import ttk
import unittest
from unittest import mock

import customtkinter as ctk

from aidas.ui.path_entry import bind_path_entry
from aidas.ui.theme import (
    COLOR_PAIRS,
    apply_appearance_mode,
    get_interface_mode,
    resolve_color,
    set_interface_mode,
)


class _QuietRoot(ctk.CTk):
    _deactivate_windows_window_header_manipulation = True


class DirectoryCompletionInteractionTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = _QuietRoot()
            self.addCleanup(self.root.destroy)
            self.root.attributes("-alpha", 0)
            self.root.geometry("720x320+0+0")
            self.root.update()
        except tk.TclError as exc:
            self.skipTest(f"A transparent Tk test window is unavailable: {exc}")
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        for name in ("Alpha", "alpine", "Beta folder"):
            (self.folder / name).mkdir()
        (self.folder / "alpha.txt").write_text("not a directory", encoding="utf-8")

    def _wait_for(self, predicate, message="The completion UI did not settle"):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            self.root.update()
            if predicate():
                return
            time.sleep(0.005)
        self.assertTrue(predicate(), message)

    def _entry(self, kind, command=None):
        entry = kind(self.root)
        entry.pack(fill="x", padx=12, pady=12)
        bind_path_entry(entry, command)
        target = getattr(entry, "_entry", entry)
        self.root.update()
        target.focus_force()
        self.root.update()
        return entry, target

    @staticmethod
    def _visible(controller):
        popup = controller.popup
        return popup is not None and popup.winfo_exists() and popup.winfo_viewable()

    def _complete(self, entry, target, text, expected_names):
        target.focus_force()
        self.root.update()
        target.event_generate("<Escape>")
        self.root.update()
        target.delete(0, "end")
        target.insert(0, text)
        target.icursor("end")
        controller = entry._path_completion
        controller.refresh_now()
        self._wait_for(
            lambda: [name for name, _path in controller.suggestions] == expected_names
            and self._visible(controller),
            f"Expected directory suggestions {expected_names!r}",
        )
        return controller

    def test_native_and_ctk_inputs_suggest_matching_directories_without_taking_focus(self):
        for kind in (ttk.Entry, ctk.CTkEntry):
            with self.subTest(kind=kind.__name__):
                submit = mock.Mock()
                entry, target = self._entry(kind, submit)
                controller = self._complete(entry, target, str(self.folder / "al"), ["Alpha", "alpine"])

                self.assertEqual(
                    [Path(path) for _name, path in controller.suggestions],
                    [self.folder / "Alpha", self.folder / "alpine"],
                )
                self.assertTrue(controller.popup.overrideredirect())
                self.assertIs(self.root.focus_get(), target)
                self.assertIsNone(self.root.grab_current())
                submit.assert_not_called()
                entry.destroy()

    def test_windows_initialization_and_reverse_tab_do_not_require_the_x11_keysym(self):
        if self.root.tk.call("tk", "windowingsystem") != "win32":
            self.skipTest("Windows uses Shift-Tab without the X11 ISO_Left_Tab keysym")
        for kind in (ttk.Entry, ctk.CTkEntry):
            with self.subTest(kind=kind.__name__):
                entry = kind(self.root)
                entry.pack(fill="x", padx=12, pady=12)
                target = getattr(entry, "_entry", entry)
                native_bind = target.bind

                def bind_without_x11(sequence=None, *args, **kwargs):
                    if sequence == "<ISO_Left_Tab>":
                        raise tk.TclError('bad event type or keysym "ISO_Left_Tab"')
                    return native_bind(sequence, *args, **kwargs)

                with mock.patch.object(target, "bind", side_effect=bind_without_x11):
                    bind_path_entry(entry, mock.Mock())
                self.root.update()
                controller = self._complete(entry, target, str(self.folder / "al"), ["Alpha", "alpine"])
                target.event_generate("<Tab>")
                target.event_generate("<Tab>")
                target.event_generate("<Shift-Tab>")
                self.root.update()
                self.assertEqual(Path(target.get()), self.folder / "Alpha")
                self.assertTrue(self._visible(controller))
                entry.destroy()

    def test_tab_completes_first_or_highlighted_directory_without_submitting(self):
        for kind in (ttk.Entry, ctk.CTkEntry):
            with self.subTest(kind=kind.__name__):
                submit = mock.Mock()
                entry, target = self._entry(kind, submit)
                controller = self._complete(entry, target, str(self.folder / "al"), ["Alpha", "alpine"])
                target.event_generate("<Tab>")
                self.root.update()
                self.assertEqual(Path(target.get()), self.folder / "Alpha")
                self.assertIs(self.root.focus_get(), target)
                submit.assert_not_called()
                target.event_generate("<Tab>")
                self.root.update()
                self.assertEqual(Path(target.get()), self.folder / "alpine")
                target.event_generate("<Shift-Tab>")
                self.root.update()
                self.assertEqual(Path(target.get()), self.folder / "Alpha")

                controller = self._complete(entry, target, str(self.folder / "al"), ["Alpha", "alpine"])
                target.event_generate("<Down>")
                target.event_generate("<Down>")
                self.root.update()
                selected = controller.listbox.curselection()
                self.assertTrue(selected)
                expected = Path(controller.suggestions[selected[0]][1])
                target.event_generate("<Tab>")
                self.root.update()
                self.assertEqual(Path(target.get()), expected)
                self.assertIs(self.root.focus_get(), target)
                submit.assert_not_called()
                entry.destroy()

    def test_enter_confirms_the_highlighted_directory_then_a_second_enter_loads_it(self):
        for kind in (ttk.Entry, ctk.CTkEntry):
            with self.subTest(kind=kind.__name__):
                submitted = []
                entry, target = self._entry(kind, lambda: submitted.append(target.get()))
                controller = self._complete(entry, target, str(self.folder / "al"), ["Alpha", "alpine"])
                target.event_generate("<Down>")
                self.root.update()
                selected = controller.listbox.curselection()
                self.assertTrue(selected)
                expected = Path(controller.suggestions[selected[0]][1])

                target.event_generate("<Return>")
                self.root.update()

                self.assertEqual(Path(target.get()), expected)
                self.assertEqual(submitted, [])
                self.assertFalse(self._visible(controller))
                self.assertIs(self.root.focus_get(), target)
                target.event_generate("<Return>")
                self.root.update()
                self.assertEqual([Path(value) for value in submitted], [expected])
                target.delete(0, "end")
                target.insert(0, str(self.folder / "Beta folder"))
                target.event_generate("<Escape>")
                target.event_generate("<Return>")
                self.root.update()
                self.assertEqual(Path(submitted[-1]), self.folder / "Beta folder")
                self.assertEqual(len(submitted), 2)
                entry.destroy()

    def test_enter_after_tab_completion_only_closes_the_dropdown_before_loading(self):
        for kind in (ttk.Entry, ctk.CTkEntry):
            with self.subTest(kind=kind.__name__):
                submit = mock.Mock()
                entry, target = self._entry(kind, submit)
                controller = self._complete(entry, target, str(self.folder / "al"), ["Alpha", "alpine"])
                target.event_generate("<Tab>")
                self.root.update()
                self.assertEqual(Path(target.get()), self.folder / "Alpha")
                self.assertTrue(self._visible(controller))

                target.event_generate("<Return>")
                self.root.update()
                self.assertFalse(self._visible(controller))
                self.assertEqual(Path(target.get()), self.folder / "Alpha")
                submit.assert_not_called()

                target.event_generate("<Return>")
                self.root.update()
                submit.assert_called_once_with()
                entry.destroy()

    def test_enter_without_a_highlight_closes_the_dropdown_and_preserves_the_address(self):
        for kind in (ttk.Entry, ctk.CTkEntry):
            with self.subTest(kind=kind.__name__):
                submit = mock.Mock()
                entry, target = self._entry(kind, submit)
                controller = self._complete(entry, target, str(self.folder), ["Alpha", "alpine", "Beta folder"])
                self.assertFalse(controller.listbox.curselection())

                target.event_generate("<Return>")
                self.root.update()
                self.assertFalse(self._visible(controller))
                self.assertEqual(Path(target.get()), self.folder)
                submit.assert_not_called()

                target.event_generate("<Return>")
                self.root.update()
                submit.assert_called_once_with()
                entry.destroy()

    def test_click_completes_a_suggestion_without_loading_and_escape_dismisses(self):
        for kind in (ttk.Entry, ctk.CTkEntry):
            with self.subTest(kind=kind.__name__):
                submit = mock.Mock()
                entry, target = self._entry(kind, submit)
                controller = self._complete(entry, target, str(self.folder / "al"), ["Alpha", "alpine"])
                listbox = controller.listbox
                bounds = listbox.bbox(1)
                self.assertIsNotNone(bounds)
                x, y, width, height = bounds
                listbox.event_generate("<ButtonPress-1>", x=x + width // 2, y=y + height // 2)
                if listbox.winfo_exists():
                    listbox.event_generate("<ButtonRelease-1>", x=x + width // 2, y=y + height // 2)
                self.root.update()
                self.assertEqual(Path(target.get()), self.folder / "alpine")
                self.assertIs(self.root.focus_get(), target)
                submit.assert_not_called()

                controller = self._complete(entry, target, str(self.folder / "al"), ["Alpha", "alpine"])
                target.event_generate("<Escape>")
                self.root.update()
                self.assertFalse(self._visible(controller))
                self.assertIs(self.root.focus_get(), target)
                self.assertIsNone(self.root.grab_current())
                entry.destroy()

    def test_outside_click_and_entry_destruction_dismiss_without_replacing_owner_bindings(self):
        observed = []
        self.root.bind("<ButtonPress-1>", lambda _event: observed.append("clicked"), add="+")
        original_binding = self.root.bind("<ButtonPress-1>")
        entry, target = self._entry(ttk.Entry, mock.Mock())
        controller = self._complete(entry, target, str(self.folder / "al"), ["Alpha", "alpine"])

        self.root.event_generate("<ButtonPress-1>", x=680, y=250)
        self.root.update()

        self.assertFalse(self._visible(controller))
        self.assertEqual(observed, ["clicked"])
        self.assertEqual(self.root.bind("<ButtonPress-1>").strip(), original_binding.strip())
        controller = self._complete(entry, target, str(self.folder / "al"), ["Alpha", "alpine"])
        popup = controller.popup
        entry.destroy()
        self.root.update()
        self.assertFalse(popup.winfo_exists())
        self.assertEqual(self.root.bind("<ButtonPress-1>").strip(), original_binding.strip())

    def test_completion_preserves_a_modal_owners_existing_grab(self):
        dialog = tk.Toplevel(self.root)
        dialog.geometry("500x140+10+10")
        entry = ttk.Entry(dialog)
        entry.pack(fill="x", padx=12, pady=12)
        bind_path_entry(entry, mock.Mock())
        self.root.update()
        dialog.grab_set()
        entry.focus_force()
        self.root.update()

        controller = self._complete(entry, entry, str(self.folder / "al"), ["Alpha", "alpine"])

        self.assertIs(self.root.grab_current(), dialog)
        entry.event_generate("<Escape>")
        self.root.update()
        self.assertFalse(self._visible(controller))
        self.assertIs(self.root.grab_current(), dialog)
        dialog.grab_release()
        dialog.destroy()

    def test_clipboard_only_output_entries_do_not_create_directory_completions(self):
        for kind in (ttk.Entry, ctk.CTkEntry):
            with self.subTest(kind=kind.__name__):
                entry, target = self._entry(kind)
                target.insert(0, str(self.folder / "al"))
                target.event_generate("<KeyRelease>")
                self.root.update()
                self.assertIsNone(getattr(entry, "_path_completion", None))
                self.assertIsNotNone(entry._path_context_menu)
                entry.destroy()

    def test_typing_starts_completion_and_missing_or_disabled_inputs_hide_it(self):
        for kind in (ttk.Entry, ctk.CTkEntry):
            with self.subTest(kind=kind.__name__):
                entry, target = self._entry(kind, mock.Mock())
                target.insert(0, str(self.folder / "al"))
                target.event_generate("<KeyRelease-l>")
                controller = entry._path_completion
                self._wait_for(
                    lambda: [name for name, _path in controller.suggestions] == ["Alpha", "alpine"]
                    and self._visible(controller),
                )
                target.delete(0, "end")
                target.insert(0, str(self.folder / "does not exist"))
                target.event_generate("<KeyRelease-t>")
                self._wait_for(lambda: not controller.suggestions and not self._visible(controller))
                self._complete(entry, target, str(self.folder / "al"), ["Alpha", "alpine"])
                entry.configure(state="disabled")
                controller.refresh_now()
                self._wait_for(lambda: not self._visible(controller))
                entry.destroy()

    def test_live_theme_changes_keep_the_popup_and_listbox_on_the_menu_palette(self):
        previous_interface = get_interface_mode()
        previous_mode = ctk.get_appearance_mode()
        previous_source = ctk.AppearanceModeTracker.appearance_mode_set_by
        try:
            set_interface_mode("Modern", redraw=False)
            entry, target = self._entry(ctk.CTkEntry, mock.Mock())
            controller = self._complete(entry, target, str(self.folder / "al"), ["Alpha", "alpine"])
            for mode in ("Light", "Dark", "Light"):
                with self.subTest(mode=mode):
                    apply_appearance_mode(mode, root=self.root, style=ttk.Style(self.root))
                    self.root.update()
                    expected = lambda token: resolve_color(COLOR_PAIRS[token], mode)
                    self.assertEqual(controller.popup.cget("background"), expected("menu_surface"))
                    self.assertEqual(controller.listbox.cget("background"), expected("menu_surface"))
                    self.assertEqual(controller.listbox.cget("foreground"), expected("menu_text"))
                    self.assertEqual(controller.listbox.cget("selectbackground"), expected("menu_hover"))
                    self.assertEqual(controller.listbox.cget("selectforeground"), expected("menu_hover_text"))
            entry.destroy()
        finally:
            set_interface_mode(previous_interface, redraw=False)
            ctk.set_appearance_mode("System" if previous_source == "system" else previous_mode)

    def test_late_background_results_do_not_replace_a_newer_path_or_reopen_after_escape(self):
        entry, target = self._entry(ttk.Entry, mock.Mock())
        controller = entry._path_completion
        started = threading.Event()
        release = threading.Event()
        completed = threading.Event()
        old_text = str(self.folder / "al")

        def lookup(text):
            if text == old_text:
                started.set()
                release.wait(timeout=3)
                completed.set()
                return [("Alpha", str(self.folder / "Alpha"))]
            return [("Beta folder", str(self.folder / "Beta folder"))]

        try:
            with mock.patch("aidas.ui.directory_completion.directory_completions", side_effect=lookup):
                target.insert(0, old_text)
                controller.refresh_now()
                self._wait_for(started.is_set)
                target.delete(0, "end")
                target.insert(0, str(self.folder / "Be"))
                controller.refresh_now()
                self._wait_for(lambda: self._visible(controller)
                               and [name for name, _path in controller.suggestions] == ["Beta folder"])
                release.set()
                self._wait_for(completed.is_set)
                self.root.update()
                self.assertEqual([name for name, _path in controller.suggestions], ["Beta folder"])
                self.assertEqual(target.get(), str(self.folder / "Be"))
                target.event_generate("<Escape>")
                self.root.update()
                self.assertFalse(self._visible(controller))
                self.assertEqual(controller.suggestions, [])
        finally:
            release.set()
            entry.destroy()

    def test_focus_change_and_owner_unmap_close_the_popup(self):
        entry, target = self._entry(ttk.Entry, mock.Mock())
        other = ttk.Button(self.root, text="Other action")
        other.pack()
        controller = self._complete(entry, target, str(self.folder / "al"), ["Alpha", "alpine"])
        other.focus_force()
        self._wait_for(lambda: not self._visible(controller))

        controller = self._complete(entry, target, str(self.folder / "al"), ["Alpha", "alpine"])
        self.root.withdraw()
        self.root.update()
        self.assertFalse(self._visible(controller))
        self.assertIsNone(self.root.grab_current())

    def test_thin_scrollbar_appears_on_hover_for_long_lists_and_wheel_scrolling_works(self):
        names = [f"folder {index:02d}" for index in range(24)]
        for name in names:
            (self.folder / name).mkdir()
        for kind in (ttk.Entry, ctk.CTkEntry):
            with self.subTest(kind=kind.__name__):
                entry, target = self._entry(kind, mock.Mock())
                controller = self._complete(entry, target, str(self.folder / "folder"), names)
                popup, scrollbar = controller.popup, controller.scrollbar
                self.assertFalse(scrollbar.winfo_ismapped())
                popup.event_generate("<Enter>")
                self._wait_for(lambda: scrollbar.winfo_ismapped())
                self.assertLessEqual(scrollbar.winfo_width(), 12)
                self.assertIs(self.root.focus_get(), target)

                before = controller.listbox.yview()[0]
                controller.listbox.event_generate("<MouseWheel>", delta=-120)
                self._wait_for(lambda: controller.listbox.yview()[0] > before)

                outside = (popup.winfo_rootx() + popup.winfo_width() + 20, popup.winfo_rooty())
                with mock.patch.object(popup, "winfo_pointerxy", return_value=outside):
                    popup.event_generate("<Leave>")
                    self._wait_for(lambda: not scrollbar.winfo_ismapped())
                self.assertTrue(self._visible(controller))

                controller = self._complete(entry, target, str(self.folder / "al"), ["Alpha", "alpine"])
                controller.popup.event_generate("<Enter>")
                self.root.update()
                self.assertEqual(controller.listbox.yview(), (0.0, 1.0))
                self.assertFalse(controller.scrollbar.winfo_ismapped())
                self.assertIs(self.root.focus_get(), target)
                entry.destroy()

    def test_dropdown_wheel_does_not_reach_global_scroll_handlers(self):
        names = [f"folder {index:02d}" for index in range(24)]
        for name in names:
            (self.folder / name).mkdir()
        observed = mock.Mock()
        sequences = ("<MouseWheel>", "<Button-4>", "<Button-5>")
        bindings = [(sequence, tk.Misc.bind_all(self.root, sequence, observed, add="+"))
                    for sequence in sequences]
        try:
            for kind in (ttk.Entry, ctk.CTkEntry):
                with self.subTest(kind=kind.__name__):
                    entry, target = self._entry(kind, mock.Mock())
                    controller = self._complete(entry, target, str(self.folder / "folder"), names)
                    popup = controller.popup
                    popup.event_generate("<Enter>")
                    self._wait_for(lambda: controller.scrollbar.winfo_ismapped())
                    origin = (popup.winfo_rootx(), popup.winfo_rooty())

                    for widget in (controller.listbox, controller.scrollbar._canvas):
                        controller.listbox.yview_moveto(0)
                        self.root.update()
                        widget.event_generate("<MouseWheel>", delta=-120)
                        self.root.update()
                        self.assertGreater(controller.listbox.yview()[0], 0)
                        observed.assert_not_called()

                    # Neither popup padding nor either scroll boundary leaks a gesture.
                    popup.event_generate("<MouseWheel>", delta=-120)
                    for position, delta in ((0, 120), (1, -120)):
                        controller.listbox.yview_moveto(position)
                        self.root.update()
                        controller.listbox.event_generate("<MouseWheel>", delta=delta)
                    for sequence in sequences[1:]:
                        for widget in (controller.listbox, controller.scrollbar._canvas, popup):
                            widget.event_generate(sequence)
                    self.root.update()
                    observed.assert_not_called()
                    self.assertTrue(self._visible(controller))
                    self.assertEqual((popup.winfo_rootx(), popup.winfo_rooty()), origin)
                    self.assertIs(self.root.focus_get(), target)

                    controller.hide()
                    target.event_generate("<MouseWheel>", delta=-120)
                    self.root.update()
                    observed.assert_called_once()
                    observed.reset_mock()
                    entry.destroy()
        finally:
            for sequence, binding in bindings:
                self.root._unbind(("bind", "all", sequence), binding)

    def test_external_textvariable_updates_refresh_suggestions_and_destroy_removes_the_trace(self):
        for kind in (ttk.Entry, ctk.CTkEntry):
            with self.subTest(kind=kind.__name__):
                variable = tk.StringVar(master=self.root)
                entry = kind(self.root, textvariable=variable)
                entry.pack(fill="x", padx=12, pady=12)
                bind_path_entry(entry, mock.Mock())
                target = getattr(entry, "_entry", entry)
                self.root.update()
                target.focus_force()
                self.root.update()
                controller = entry._path_completion

                variable.set(str(self.folder / "al"))
                self._wait_for(lambda: self._visible(controller)
                               and [name for name, _path in controller.suggestions] == ["Alpha", "alpine"])
                variable.set(str(self.folder / "Be"))
                self._wait_for(lambda: self._visible(controller)
                               and [name for name, _path in controller.suggestions] == ["Beta folder"])
                self.assertEqual(variable.get(), str(self.folder / "Be"))
                self.assertIs(self.root.focus_get(), target)
                variable.set("")
                self._wait_for(lambda: not controller.suggestions and not self._visible(controller))
                entry.destroy()
                with mock.patch.object(self.root, "report_callback_exception") as callback_error:
                    variable.set(str(self.folder / "al"))
                    self.root.update()
                    callback_error.assert_not_called()

    def test_typing_and_pasting_with_a_textvariable_use_global_tcl_scope(self):
        for kind in (ttk.Entry, ctk.CTkEntry):
            with self.subTest(kind=kind.__name__):
                variable = tk.StringVar(master=self.root, value=str(self.folder) + "/")
                entry = kind(self.root, textvariable=variable)
                entry.pack(fill="x", padx=12, pady=12)
                bind_path_entry(entry, mock.Mock())
                target = getattr(entry, "_entry", entry)
                self.root.update()
                target.focus_force()
                target.icursor("end")
                self.root.update()
                controller = entry._path_completion

                with mock.patch.object(self.root, "report_callback_exception") as callback_error:
                    # These run Tk's Entry class bindings, including its Tcl procedures.
                    for key in ("a", "l"):
                        target.event_generate(f"<KeyPress-{key}>")
                        target.event_generate(f"<KeyRelease-{key}>")
                    self._wait_for(lambda: self._visible(controller)
                                   and [name for name, _path in controller.suggestions] == ["Alpha", "alpine"])
                    self.assertEqual(Path(variable.get()), self.folder / "al")
                    callback_error.assert_not_called()

                    target.selection_range(0, "end")
                    target.clipboard_clear()
                    target.clipboard_append(str(self.folder / "Be"))
                    target.event_generate("<<Paste>>")
                    self._wait_for(lambda: self._visible(controller)
                                   and [name for name, _path in controller.suggestions] == ["Beta folder"])
                    self.assertEqual(Path(variable.get()), self.folder / "Be")
                    callback_error.assert_not_called()
                entry.destroy()


if __name__ == "__main__":
    unittest.main()
