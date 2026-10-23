from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


class Step3ResultZoomUITests(unittest.TestCase):
    def test_close_stays_visible_when_maximized_and_resized_at_different_scales(self):
        # A fresh interpreter isolates CTk's process-wide scaling callbacks.
        # Real mapped widgets are necessary to reproduce image request sizes
        # taking all the available height away from the Close footer.
        script = textwrap.dedent(
            """
            import sys
            import time
            import tkinter as tk
            from unittest import mock

            import customtkinter as ctk
            from PIL import Image, ImageChops, ImageTk

            from aidas.core.display import work_area_bounds
            from aidas.steps.step3_flatten import Step3ResultZoomDialog
            from aidas.ui.theme import COLOR_PAIRS, resolve_color, set_interface_mode

            scale = float(sys.argv[1])
            set_interface_mode('Modern', redraw=False)
            ctk.set_widget_scaling(scale)
            ctk.set_window_scaling(scale)

            class QuietRoot(ctk.CTk):
                _deactivate_windows_window_header_manipulation = True

            root = QuietRoot()
            root.interface_mode = 'Modern'
            root.withdraw()
            root.attributes('-alpha', 0.0)
            root.geometry('800x600+10000+10000')
            root.deiconify()
            root.update_idletasks()
            # CTk queues native icon and DPI min/max initialization. Let the
            # owner finish startup before creating transient viewers, as the
            # application does, so those jobs cannot replace their HWNDs.
            started = time.monotonic()
            deadline = started + 3
            while True:
                root.update_idletasks()
                root.update()
                pending = [str(root.tk.call('after', 'info', job)[0])
                           for job in root.tk.call('after', 'info')]
                initializing = any('_windows_set_titlebar_icon' in job or '_set_scaled_min_max' in job
                                   for job in pending)
                if time.monotonic() - started >= 0.25 and not initializing:
                    break
                assert time.monotonic() < deadline, 'Owner startup did not settle.'
                time.sleep(0.005)
            callback_errors = []
            root.report_callback_exception = lambda *error: callback_errors.append(error)

            original_init = tk.Toplevel.__init__

            def invisible_init(window, *args, **kwargs):
                original_init(window, *args, **kwargs)
                window.withdraw()
                window.attributes('-alpha', 0.0)

            def settle():
                # More than one pass exercises configure events caused by a
                # new PhotoImage and any deferred redraws queued by resizing.
                for _ in range(4):
                    root.update_idletasks()
                    root.update()

            def maximize(dialog):
                bar = getattr(dialog, '_aidas_dialog_title_bar', None)
                if bar is not None:
                    bar.controller.toggle_maximize()
                    return
                try:
                    dialog.state('zoomed')
                except tk.TclError:
                    try:
                        dialog.attributes('-zoomed', True)
                    except tk.TclError:
                        pass

            def restore(dialog):
                bar = getattr(dialog, '_aidas_dialog_title_bar', None)
                if bar is None:
                    dialog.state('normal')
                else:
                    bar.controller.restore()
                deadline = time.monotonic() + 3
                while True:
                    settle()
                    transitioning = bar is not None and bar.controller._normal_state_transition
                    if dialog.state() == 'normal' and not transitioning:
                        return
                    assert time.monotonic() < deadline, 'Native restore did not settle.'
                    time.sleep(0.005)

            def resize(dialog, width, height):
                bar = getattr(dialog, '_aidas_dialog_title_bar', None)
                if bar is None:
                    dialog.geometry(f'{width}x{height}+10000+10000')
                else:
                    # Client-edge dragging uses physical Windows dimensions;
                    # Tk wm geometry still assumes its removed native frame.
                    assert bar.controller.resize_window(10000, 10000, width, height)

            def assert_layout(dialog):
                settle()
                footer = dialog.footer
                button = dialog.close_button
                label = dialog.image_label
                assert button.winfo_ismapped(), 'Close must remain mapped.'
                assert footer.winfo_ismapped(), 'Footer must remain mapped.'
                client_bottom = dialog.winfo_rooty() + dialog.winfo_height()
                button_bottom = button.winfo_rooty() + button.winfo_height()
                footer_bottom = footer.winfo_rooty() + footer.winfo_height()
                assert button.winfo_rooty() >= dialog.winfo_rooty(), 'Close is above client area.'
                assert button_bottom <= client_bottom, 'Close is clipped below the client area.'
                assert footer_bottom <= client_bottom, 'Footer is clipped below the client area.'
                assert label.winfo_rooty() + label.winfo_height() <= footer.winfo_rooty(), 'Image overlaps Close footer.'
                assert label.winfo_width() > 1 and label.winfo_height() > 1

                photo = ImageTk.getimage(dialog._photo).convert('RGB')
                assert photo.size == (label.winfo_width(), label.winfo_height()), 'Image must fit current viewport.'
                background = Image.new('RGB', photo.size, resolve_color(COLOR_PAIRS['surface']))
                bounds = ImageChops.difference(photo, background).getbbox()
                assert bounds is not None, 'The diagnostic image disappeared.'
                source_width, source_height = dialog._source_image.size
                ratio = min(photo.width / source_width, photo.height / source_height)
                expected_width = round(source_width * ratio)
                expected_height = round(source_height * ratio)
                left, top, right, bottom = bounds
                assert abs((right - left) - expected_width) <= 2, 'Diagnostic image width was distorted.'
                assert abs((bottom - top) - expected_height) <= 2, 'Diagnostic image height was distorted.'
                assert abs(left - (photo.width - expected_width) / 2) <= 2, 'Image is not horizontally centered.'
                assert abs(top - (photo.height - expected_height) / 2) <= 2, 'Image is not vertically centered.'
                return (dialog.winfo_width(), dialog.winfo_height(), label.winfo_width(), label.winfo_height())

            try:
                # Include a small work area even on a large developer display:
                # CI may have less room than the requested 1000x700 client.
                from contextlib import nullcontext
                bounds_patch = (
                    mock.patch('aidas.steps.step3_flatten.work_area_bounds', return_value=(0, 0, 1024, 720))
                    if sys.argv[2] == 'small' else nullcontext()
                )
                with mock.patch.object(tk.Toplevel, '__init__', invisible_init), bounds_patch:
                    for source_size in ((3600, 2400), (1200, 3600)):
                        source = Image.new('RGB', source_size, (201, 53, 16))
                        dialog = Step3ResultZoomDialog(root, source, 'Hidden diagnostic viewer test')
                        # The viewer must fit before the user chooses to
                        # maximize or resize it through the window manager.
                        assert_layout(dialog)
                        restore(dialog)
                        for width, height in ((1000, 700), (720, 480), (640, 480)):
                            minimum = dialog.minsize()
                            maximum = dialog.maxsize()
                            width = max(minimum[0], min(width, maximum[0]))
                            height = max(minimum[1], min(height, maximum[1]))
                            resize(dialog, width, height)
                            before = assert_layout(dialog)
                            assert before[0:2] == (width, height), 'Image changed the requested window size.'
                            photo = dialog._photo
                            for _ in range(4):
                                dialog._redraw()
                                settle()
                            assert assert_layout(dialog) == before, 'Redraw caused the viewport to grow.'
                            assert dialog._photo is photo, 'An unchanged viewport unnecessarily recreated the image.'
                        maximize(dialog)
                        assert_layout(dialog)
                        if dialog.state() == 'zoomed':
                            left, top, right, bottom = work_area_bounds(dialog, monitor_window=dialog)
                            assert dialog.winfo_rootx() >= left, 'Maximized client area is left of the usable screen.'
                            assert dialog.winfo_rooty() >= top, 'Maximized client area is above the usable screen.'
                            assert dialog.winfo_rootx() + dialog.winfo_width() <= right, 'Maximized client area extends beyond the usable screen.'
                            assert dialog.winfo_rooty() + dialog.winfo_height() <= bottom, 'Maximized client area overlaps the taskbar.'
                            assert dialog.close_button.winfo_rooty() + dialog.close_button.winfo_height() <= bottom, 'Maximized Close is behind the taskbar.'
                        restore(dialog)
                        resize(dialog, 720, 480)
                        assert_layout(dialog)
                        dialog._queue_redraw()
                        assert dialog._redraw_after_id is not None
                        dialog.close_button.invoke()
                        settle()
                        assert not dialog.winfo_exists(), 'Close did not destroy the viewer.'
                        assert dialog._redraw_after_id is None, 'Close left a queued redraw.'
                        assert dialog._photo is None and dialog._source_image is None, 'Close retained diagnostic images.'

                    dialog = Step3ResultZoomDialog(root, Image.new('RGB', (3600, 2400)), 'Hidden Escape test')
                    settle()
                    # Window managers can refuse focus for an invisible
                    # window. Invoke its registered binding as Tk would.
                    binding = dialog.bind('<Escape>')
                    assert binding, 'Escape must remain a close shortcut.'
                    command = binding.split('[', 1)[1].split(' ', 1)[0]
                    dialog.tk.call(command, *(['0'] * len(dialog._subst_format)))
                    settle()
                    assert not dialog.winfo_exists(), 'Escape did not destroy the viewer.'
                    assert not callback_errors, callback_errors
            finally:
                root.destroy()
            """
        )
        for scale, work_area in ((1.0, 'actual'), (1.25, 'actual'), (2.0, 'actual'), (1.0, 'small')):
            with self.subTest(scale=scale, work_area=work_area):
                result = subprocess.run(
                    [sys.executable, "-c", script, str(scale), work_area],
                    cwd=Path(__file__).resolve().parents[1],
                    capture_output=True,
                    text=True,
                    timeout=90,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_initial_window_and_close_fit_monitor_work_area_without_forced_maximize(self):
        script = textwrap.dedent(
            """
            import sys
            import time
            import tkinter as tk
            from unittest import mock

            import customtkinter as ctk
            from PIL import Image

            from aidas.steps import step3_flatten as module
            from aidas.ui.theme import set_interface_mode

            scale = float(sys.argv[1])
            set_interface_mode('Modern', redraw=False)
            ctk.set_widget_scaling(scale)
            ctk.set_window_scaling(scale)

            class QuietRoot(ctk.CTk):
                _deactivate_windows_window_header_manipulation = True

            root = QuietRoot()
            root.interface_mode = 'Modern'
            root.withdraw()
            root.attributes('-alpha', 0.0)
            root.geometry('800x600+10000+10000')
            root.deiconify()
            root.update_idletasks()
            started = time.monotonic()
            deadline = started + 3
            while True:
                root.update_idletasks()
                root.update()
                pending = [str(root.tk.call('after', 'info', job)[0])
                           for job in root.tk.call('after', 'info')]
                initializing = any('_windows_set_titlebar_icon' in job or '_set_scaled_min_max' in job
                                   for job in pending)
                if time.monotonic() - started >= 0.25 and not initializing:
                    break
                assert time.monotonic() < deadline, 'Owner startup did not settle.'
                time.sleep(0.005)
            original_init = tk.Toplevel.__init__
            original_state = tk.Toplevel.state
            original_attributes = tk.Toplevel.attributes
            state_requests = []
            attribute_requests = []

            def invisible_init(window, *args, **kwargs):
                original_init(window, *args, **kwargs)
                window.withdraw()
                window.attributes('-alpha', 0.0)

            def recorded_state(window, newstate=None):
                state_requests.append((window, newstate))
                return original_state(window, newstate)

            def recorded_attributes(window, *args):
                attribute_requests.append((window, args))
                return original_attributes(window, *args)

            try:
                # Record native calls rather than suppressing maximization:
                # the old test hid the production error by forcing normal.
                actual_bounds = module.work_area_bounds(root)
                with mock.patch.object(tk.Toplevel, '__init__', invisible_init), mock.patch.object(tk.Toplevel, 'state', recorded_state), mock.patch.object(tk.Toplevel, 'attributes', recorded_attributes):
                    for bounds in (
                        actual_bounds,
                        (0, 0, 1000, 660),  # Bottom taskbar.
                        (0, 60, 1000, 720),  # Top taskbar.
                        (80, 0, 1000, 720),  # Left taskbar.
                        (0, 0, 920, 720),  # Right taskbar.
                        (100, 50, 900, 650),
                        (100, 50, 740, 530),
                        (100, 50, 1100, 530),
                        (-1280, 40, -480, 640),
                    ):
                        with mock.patch.object(module, 'work_area_bounds', return_value=bounds):
                            dialog = module.Step3ResultZoomDialog(root, Image.new('RGB', (3600, 2400)), 'Hidden initial fit test')
                            for _ in range(4):
                                root.update_idletasks()
                                root.update()
                            assert not any(window is dialog and state == 'zoomed' for window, state in state_requests), 'The constructor must not force native maximization.'
                            assert not any(window is dialog and args and args[0] == '-zoomed' and len(args) > 1 and args[1] for window, args in attribute_requests), 'The constructor must not request the platform maximization fallback.'
                            assert dialog.state() == 'normal', 'The viewer must initially open as a bounded normal window.'
                            left, top, right, bottom = bounds
                            frame_left = max(0, dialog.winfo_rootx() - dialog.winfo_x())
                            assert dialog.winfo_x() >= left, 'The native left border is outside the work area.'
                            assert dialog.winfo_y() >= top, 'The native title bar is outside the work area.'
                            assert dialog.winfo_rootx() >= left, 'Initial image viewer is left of the work area.'
                            assert dialog.winfo_rooty() >= top, 'Initial image viewer is above the work area.'
                            assert dialog.winfo_rootx() + dialog.winfo_width() <= right, 'Initial image viewer extends right of the work area.'
                            assert dialog.winfo_rooty() + dialog.winfo_height() <= bottom, 'Initial image viewer extends below the work area.'
                            assert dialog.winfo_rootx() + dialog.winfo_width() + frame_left <= right, 'The native right border is outside the work area.'
                            assert dialog.winfo_rooty() + dialog.winfo_height() + frame_left <= bottom, 'The native bottom border overlaps the taskbar.'
                            assert dialog.close_button.winfo_rooty() + dialog.close_button.winfo_height() <= bottom, 'Close is below the usable screen.'
                            assert dialog.close_button.winfo_ismapped()
                            assert dialog.image_label.winfo_rooty() + dialog.image_label.winfo_height() <= dialog.footer.winfo_rooty(), 'The image overlaps the Close footer.'
                            dialog.destroy()
            finally:
                root.destroy()
            """
        )
        for scale in (1.0, 1.25, 2.0):
            with self.subTest(scale=scale):
                result = subprocess.run(
                    [sys.executable, "-c", script, str(scale)],
                    cwd=Path(__file__).resolve().parents[1],
                    capture_output=True,
                    text=True,
                    timeout=90,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
