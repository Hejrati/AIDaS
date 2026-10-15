from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


class SettingsScrollingTests(unittest.TestCase):
    def _run_tk_check(self, body):
        # CustomTkinter retains process-wide appearance and scaling callbacks.
        # Each check owns one isolated root with real, invisible Tk geometry.
        setup = textwrap.dedent(
            """
            import sys
            import tkinter as tk
            import customtkinter as ctk
            from aidas.ui import theme
            from aidas.ui.scrolling import DeferredScrollbar, SettingsScrollableFrame

            ctk.set_widget_scaling(1.0)
            ctk.set_window_scaling(1.0)
            root = ctk.CTk()
            root.withdraw()
            root.attributes('-alpha', 0.0)
            root.geometry('500x320+10000+10000')
            callback_errors = []
            root.report_callback_exception = lambda *args: callback_errors.append(args)
            root.tk.eval('set ::aidas_bg_errors {}')
            root.tk.eval('proc bgerror {message} {lappend ::aidas_bg_errors $message}')

            def settle():
                root.update_idletasks()
                root.update()

            def wait_for_callbacks():
                ready = tk.BooleanVar(root, False)
                root.after(100, ready.set, True)
                root.wait_variable(ready)
                settle()

            def assert_no_callback_errors():
                assert not callback_errors, callback_errors
                errors = root.tk.call('set', '::aidas_bg_errors')
                assert not errors, errors
            """
        )
        script = (
            setup
            + "\ntry:\n"
            + textwrap.indent(textwrap.dedent(body), "    ")
            + "\nfinally:\n    root.destroy()\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_scrollbar_coalesces_updates_without_changing_thumb_ranges(self):
        self._run_tk_check(
            """
            scrollbar = DeferredScrollbar(root, height=240)
            scrollbar.pack(fill='y')
            root.deiconify()
            settle()
            draws = []
            original_draw = scrollbar._draw

            def record_draw(*args, **kwargs):
                draws.append(scrollbar.get())
                return original_draw(*args, **kwargs)

            scrollbar._draw = record_draw
            for number in range(30):
                scrollbar.set(str(number / 100), str(number / 100 + 0.24))
            expected = (0.29, 0.53)
            assert scrollbar.get() == expected, scrollbar.get()
            assert not draws, 'Scroll updates must not enter rendering synchronously.'
            root.update_idletasks()
            assert draws == [expected], draws

            for _ in range(30):
                scrollbar.set(*expected)
            root.update_idletasks()
            assert draws == [expected], 'An unchanged thumb must not repaint.'

            # A draw must leave event-loop ownership with Tk. Nested idle flushes
            # repeatedly repaint all settings controls while geometry settles.
            def reject_nested_idle_flush():
                raise AssertionError('Scrollbar rendering called update_idletasks.')

            scrollbar._canvas.update_idletasks = reject_nested_idle_flush
            original_draw()
            settle()
            assert scrollbar.get() == expected
            assert scrollbar._canvas._aidas_surface_photo is not None

            scrollbar.set(0.4, 0.6)
            scrollbar.destroy()
            wait_for_callbacks()
            assert_no_callback_errors()
            """
        )

    def test_scroll_preserves_geometry_and_refreshes_only_changed_content(self):
        self._run_tk_check(
            """
            panel = SettingsScrollableFrame(root, width=450, height=280)
            panel.pack(fill='both', expand=True)
            panel.grid_columnconfigure(0, weight=1)
            for row in range(24):
                ctk.CTkLabel(panel, text=f'Setting {row}', height=36).grid(
                    row=row, column=0, sticky='ew', padx=8, pady=2)
            root.deiconify()
            settle()
            canvas = panel._parent_canvas
            initial_region = tuple(float(value) for value in canvas.cget('scrollregion').split())
            assert initial_region[3] > canvas.winfo_height()
            assert isinstance(panel._scrollbar, DeferredScrollbar)
            region_updates = []
            original_configure = canvas.configure

            def record_configure(*args, **kwargs):
                if 'scrollregion' in kwargs:
                    region_updates.append(kwargs['scrollregion'])
                return original_configure(*args, **kwargs)

            canvas.configure = record_configure
            initial_view = canvas.yview()
            for number in range(20):
                canvas.yview_moveto(0.1 + number / 40)
                settle()
            assert canvas.yview()[0] > initial_view[0]
            assert not region_updates, 'Scrolling alone must not recompute content geometry.'
            assert panel._scrollbar.get() == canvas.yview()

            ctk.CTkLabel(panel, text='Added setting', height=400).grid(
                row=24, column=0, sticky='ew')
            settle()
            expanded_region = tuple(float(value) for value in canvas.cget('scrollregion').split())
            assert expanded_region[3] > initial_region[3]
            assert region_updates, 'New content must extend the scroll region.'

            for appearance, scale in [('Dark', 1.0), ('Light', 1.25), ('Dark', 1.0)]:
                ctk.set_appearance_mode(appearance)
                ctk.set_widget_scaling(scale)
                root.geometry('620x380+10000+10000')
                settle()
                assert panel.winfo_width() == canvas.winfo_width(), (
                    panel.winfo_width(), canvas.winfo_width())
                assert canvas.winfo_height() > 100
                assert panel._scrollbar.get() == canvas.yview()
                assert panel._scrollbar._canvas._aidas_surface_photo is not None
            assert_no_callback_errors()
            """
        )

    def test_mousewheel_bindings_stay_local_and_are_removed_on_reopen(self):
        self._run_tk_check(
            """
            sequences = ('<MouseWheel>', '<Button-4>', '<Button-5>',
                         '<KeyPress-Shift_L>', '<KeyPress-Shift_R>',
                         '<KeyRelease-Shift_L>', '<KeyRelease-Shift_R>')
            global_bindings = {sequence: root.bind_all(sequence) for sequence in sequences}
            local_bindings = {sequence: root.bind(sequence) for sequence in sequences}
            unrelated_calls = []
            root.bind_all('<MouseWheel>', lambda event: unrelated_calls.append(event.widget), add='+')
            global_bindings['<MouseWheel>'] = root.bind_all('<MouseWheel>')
            other = ctk.CTkToplevel(root)
            other.withdraw()
            other.attributes('-alpha', 0.0)
            other.geometry('200x120+10000+10000')
            other_label = tk.Label(other, text='Other window')
            other_label.pack()
            other.deiconify()
            root.deiconify()

            for _ in range(3):
                panel = SettingsScrollableFrame(root, width=450, height=280)
                panel.pack(fill='both', expand=True)
                for row in range(24):
                    label = tk.Label(panel, text=f'Setting {row}', height=2)
                    label.grid(row=row, column=0, sticky='ew')
                settle()
                for sequence, binding in global_bindings.items():
                    assert root.bind_all(sequence) == binding, sequence
                before = panel._parent_canvas.yview()
                if sys.platform.startswith('linux'):
                    label.event_generate('<Button-5>')
                else:
                    label.event_generate('<MouseWheel>', delta=-120)
                settle()
                after = panel._parent_canvas.yview()
                assert after[0] > before[0], (before, after)

                if sys.platform.startswith('linux'):
                    other_label.event_generate('<Button-5>')
                else:
                    other_label.event_generate('<MouseWheel>', delta=-120)
                settle()
                assert panel._parent_canvas.yview() == after

                # Closing before queued paint runs must cancel its callback and
                # remove only this panel's handlers from the containing window.
                panel._scrollbar.set(0.3, 0.5)
                panel.destroy()
                wait_for_callbacks()
                for sequence, binding in global_bindings.items():
                    assert root.bind_all(sequence) == binding, sequence
                for sequence, binding in local_bindings.items():
                    assert root.bind(sequence) == binding, sequence
                assert_no_callback_errors()

            if not sys.platform.startswith('linux'):
                assert other_label in unrelated_calls, 'Existing app wheel handlers must survive.'
            other.destroy()
            """
        )

    def test_destroying_the_settings_window_cancels_queued_work(self):
        self._run_tk_check(
            """
            root.deiconify()
            global_wheel_binding = root.bind_all('<MouseWheel>')
            for _ in range(3):
                window = ctk.CTkToplevel(root)
                window.withdraw()
                window.attributes('-alpha', 0.0)
                window.geometry('460x280+10000+10000')
                panel = SettingsScrollableFrame(window)
                panel.pack(fill='both', expand=True)
                for row in range(20):
                    tk.Label(panel, text=f'Setting {row}', height=2).grid(row=row, column=0)
                window.deiconify()
                settle()
                panel._scrollbar.set(0.3, 0.5)
                pending_redraw = panel._scrollbar._redraw_after_id
                assert pending_redraw is not None
                binding_commands = [binding_id for _, _, binding_id in panel._scoped_bindings]
                assert binding_commands, 'The settings window must own its wheel handlers.'

                window.destroy()
                assert not panel.winfo_exists()
                assert pending_redraw not in root.tk.call('after', 'info')
                for command in binding_commands:
                    assert not root.tk.call('info', 'commands', command), command
                assert root.bind_all('<MouseWheel>') == global_wheel_binding
                wait_for_callbacks()
                assert_no_callback_errors()
            """
        )


if __name__ == "__main__":
    unittest.main()
