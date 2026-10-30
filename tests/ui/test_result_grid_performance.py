from pathlib import Path
from types import SimpleNamespace
import threading
import time
import tkinter as tk
from tkinter import ttk
import unittest
from unittest.mock import Mock, patch

from PIL import Image, ImageDraw, ImageTk

from aidas.steps import step3_flatten as module


class ResultGridPerformanceTests(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.root.attributes("-alpha", 0)
        self.root.geometry("1000x700+10000+10000")
        self.errors = []
        self.root.report_callback_exception = lambda *error: self.errors.append(error)
        self.frame = module.Step3Frame.__new__(module.Step3Frame)
        self.frame.plot_holder = ttk.Frame(self.root)
        self.frame.plot_holder.pack(fill="both", expand=True)
        self.frame.canvas = None
        self.frame._result_zoom_dialog = None
        self.frame._open_result_zoom = Mock()
        self.images = []
        self.reads = []
        self.read_threads = []
        for size in ((1200, 900), (1600, 520)):
            image = Image.new("RGB", size, "white")
            draw = ImageDraw.Draw(image)
            box = (60, 50, size[0] - 30, size[1] - 55)
            draw.rectangle(box, outline="black")
            draw.line([(x, 100 + x % 257) for x in range(60, size[0] - 30)], fill="blue")
            self.images.append(image)
        self.root.deiconify()
        self.settle()

    def tearDown(self):
        self.root.destroy()
        self.assertFalse(self.errors, self.errors)

    def settle(self):
        deadline = time.monotonic() + 10
        while True:
            self.root.update_idletasks()
            self.root.update()
            canvas = self.frame.canvas
            if not isinstance(canvas, module.ResultGridCanvas) or not canvas.winfo_exists():
                return
            if canvas._refresh_after_id is None and all(
                index in canvas._pages and canvas._pages[index].get("ready", True)
                for index in canvas._visible_pages
            ):
                return
            if time.monotonic() >= deadline:
                self.fail("Visible result previews did not finish rendering within 10 seconds")
            time.sleep(0.005)

    def open_grid(self, count=12, sided=False, *, on_load=None, wait=True):
        images_by_filename = {
            filename: self.images[index]
            for index, (_title, filename) in enumerate(module.Step3Frame.RESULT_IMAGE_ROWS)
        }

        def load(folder, filename):
            if on_load is not None:
                on_load(folder, filename)
            image = images_by_filename[filename].copy()
            self.reads.append((folder, filename, image))
            self.read_threads.append(threading.get_ident())
            return image

        self.frame._load_result_png_from_folder = load
        folders = [Path(f"study-{index}") for index in range(count)]
        if sided:
            folders = [folder / side for folder in folders for side in module.Step3Frame.RESULT_SIDE_ORDER]
        self.frame._render_result_grid(folders)
        if wait:
            self.settle()
        return self.frame.canvas

    def click_action(self, canvas, action):
        left, top, right, bottom = action["bounds"]
        x = round((left + right) / 2 - canvas.canvasx(0))
        y = round((top + bottom) / 2 - canvas.canvasy(0))
        canvas.event_generate("<Motion>", x=x, y=y)
        canvas.event_generate("<Button-1>", x=x, y=y)
        self.settle()

    @staticmethod
    def actions(canvas, index, kind):
        return [action for action in canvas._pages[index]["actions"] if action["kind"] == kind]

    def test_aligned_preview_pixels_match_original_full_quality_rendering(self):
        bounds = [self.frame._result_plot_bounds(image) for image in self.images]
        margins = [self.frame._result_plot_margins(image, box) for image, box in zip(self.images, bounds)]
        margins = tuple(max(values) for values in zip(*margins))
        aspect = (bounds[0][3] - bounds[0][1]) / (bounds[0][2] - bounds[0][0])
        originals = [image.tobytes() for image in self.images]
        for image, box in zip(self.images, bounds):
            with self.subTest(size=image.size):
                label = self.frame._display_preview_image(
                    image, background="#ffffff", plot_bounds=box,
                    plot_margins=margins, plot_box_aspect=aspect, align_top=True,
                )
                self.settle()
                for geometry in ("1000x700", "740x520"):
                    self.root.geometry(geometry + "+10000+10000")
                    self.settle()
                    size = (label.winfo_width(), label.winfo_height())
                    left, top, width, height = self.frame._result_preview_layout(
                        image.size, box, margins, size, aspect,
                    )
                    expected = Image.new("RGB", size, "#ffffff")
                    expected.paste(image.resize((width, height), Image.Resampling.LANCZOS), (left, top))
                    self.assertEqual(ImageTk.getimage(label.preview_photo).convert("RGB").tobytes(), expected.tobytes())
                label.destroy()
        self.assertEqual([image.tobytes() for image in self.images], originals)

    def test_canvas_preview_pixels_and_axes_keep_the_full_quality_layout(self):
        original_pixels = [image.tobytes() for image in self.images]
        canvas = self.open_grid(count=3)
        for geometry in ("1000x700", "740x520"):
            self.root.geometry(geometry + "+10000+10000")
            self.settle()
            previews = canvas._pages[0]["previews"]
            self.assertEqual(len(previews), 2)
            plot_edges = []
            for preview in previews:
                image = preview["image"]
                x, y, width, height = preview["target"]
                target_size = (width, height)
                left, top, fitted_width, fitted_height = self.frame._result_preview_layout(
                    image.size, preview["bounds"], preview["margins"],
                    target_size, preview["aspect"],
                )
                expected = Image.new("RGB", target_size, canvas.cget("background"))
                expected.paste(image.resize((fitted_width, fitted_height), Image.Resampling.LANCZOS), (left, top))
                self.assertEqual(
                    ImageTk.getimage(preview["photo"]).convert("RGB").tobytes(),
                    expected.tobytes(),
                )
                self.assertEqual(canvas.type(preview["item"]), "image")
                self.assertEqual(canvas.coords(preview["item"]), [float(x), float(y)])
                self.assertEqual(canvas.itemcget(preview["item"], "anchor"), "nw")
                bounds_left, bounds_top, bounds_right, bounds_bottom = preview["bounds"]
                plot_edges.append((
                    left + bounds_left * fitted_width / image.width,
                    top + bounds_top * fitted_height / image.height,
                    left + bounds_right * fitted_width / image.width,
                    top + bounds_bottom * fitted_height / image.height,
                ))
            for first, second in zip(plot_edges[0], plot_edges[1]):
                self.assertAlmostEqual(first, second, delta=2)
        self.assertEqual([image.tobytes() for image in self.images], original_pixels)

    def test_appearance_refresh_updates_canvas_palette_without_changing_image_pixels(self):
        with patch.object(module, "resolve_color", side_effect=lambda colors: colors[1]):
            dark_canvas = self.open_grid(count=1)
        self.assertEqual(dark_canvas.cget("background"), module.COLOR_PAIRS["surface"][1])
        with patch.object(module, "resolve_color", side_effect=lambda colors: colors[0]):
            self.frame.refresh_appearance()
            self.settle()
        canvas = self.frame.canvas
        self.assertFalse(dark_canvas.winfo_exists())
        self.assertIsNot(canvas, dark_canvas)
        self.assertEqual(canvas.cget("background"), module.COLOR_PAIRS["surface"][0])
        page = canvas._pages[0]
        for preview in page["previews"]:
            source = preview["image"]
            left, top, width, height = preview["layout"]
            expected = Image.new("RGB", preview["target"][2:], canvas.cget("background"))
            expected.paste(source.resize((width, height), Image.Resampling.LANCZOS), (left, top))
            self.assertEqual(ImageTk.getimage(preview["photo"]).convert("RGB").tobytes(), expected.tobytes())
        path_texts = [
            item for item in page["items"]
            if canvas.type(item) == "text" and "study-0" in canvas.itemcget(item, "text")
        ]
        self.assertTrue(path_texts)
        for item in path_texts:
            self.assertEqual(canvas.itemcget(item, "fill"), module.COLOR_PAIRS["link"][0])

    def test_scrolling_cached_pages_does_not_resize_or_upload_images(self):
        canvas = self.open_grid()
        canvas.yview_moveto(0.07)
        self.settle()
        canvas.yview_moveto(0)
        self.settle()
        page = canvas._pages[0]
        photos = [preview["photo"] for preview in page["previews"]]
        reads = len(self.reads)
        with patch.object(module.ImageTk, "PhotoImage", wraps=ImageTk.PhotoImage) as upload, \
                patch.object(Image.Image, "resize", autospec=True, side_effect=Image.Image.resize) as resize:
            for _ in range(3):
                canvas.yview_moveto(0.07)
                self.settle()
                canvas.yview_moveto(0)
                self.settle()
            for index in range(20):
                canvas.event_generate("<Configure>", x=index, y=-index,
                                      width=canvas.winfo_width(), height=canvas.winfo_height())
            self.settle()
            upload.assert_not_called()
            resize.assert_not_called()
        self.assertEqual(len(self.reads), reads)
        self.assertIs(canvas._pages[0], page)
        self.assertEqual([preview["photo"] for preview in page["previews"]], photos)

    def test_large_batch_loads_visible_subjects_and_bounds_the_page_cache(self):
        canvas = self.open_grid(count=30, sided=True)
        first_subject = canvas._pages[0]
        first_items = set(first_subject["items"])
        self.assertLessEqual(len(self.reads), 8)
        self.assertEqual(len(first_subject["previews"]), 4)
        for fraction in (0.25, 0.5, 0.75, 1.0):
            canvas.yview_moveto(fraction)
            self.settle()
            self.assertLessEqual(len(canvas._pages), canvas.CACHE_PAGES)
            self.assertLessEqual(len(canvas._resize_sources), canvas.CACHE_PAGES)
            self.assertTrue(all(canvas.type(item) != "window" for item in canvas.find_all()))
            self.assertFalse(canvas.winfo_children())
            for index in canvas._visible_pages:
                self.assertIn(index, canvas._pages)
                self.assertEqual(len(canvas._pages[index]["previews"]), 4)
            retained_items = {item for page in canvas._pages.values() for item in page["items"]}
            self.assertTrue(retained_items.issubset(set(canvas.find_all())))
        self.assertIn(29, canvas._pages)
        self.assertAlmostEqual(canvas.yview()[1], 1.0)
        self.assertFalse(first_items.intersection(canvas.find_all()))
        canvas.yview_moveto(0)
        self.settle()
        self.assertIn(0, canvas._pages)
        self.assertIsNot(canvas._pages[0], first_subject)

    def test_zoom_retains_original_resolution_from_lazy_pages(self):
        canvas = self.open_grid()
        canvas.yview_moveto(1)
        self.settle()
        original = canvas._pages[11]["previews"][0]["image"]
        for kind in ("preview", "expand"):
            with self.subTest(action=kind):
                self.frame._open_result_zoom.reset_mock()
                self.click_action(canvas, self.actions(canvas, 11, kind)[0])
                self.frame._open_result_zoom.assert_called_once()
                zoom_image, title = self.frame._open_result_zoom.call_args.args
                self.assertIs(zoom_image, original)
                self.assertEqual(zoom_image.size, self.images[0].size)
                self.assertEqual(zoom_image.tobytes(), self.images[0].tobytes())
                self.assertIn("study-11", title)
                self.assertIn("Vertex", title)

    def test_folder_links_open_each_directory_with_mouse_enter_and_space(self):
        self.frame._open_result_directory = Mock()
        canvas = self.open_grid(count=1, sided=True)
        links = self.actions(canvas, 0, "folder")
        self.assertEqual(len(links), 3)
        expected_paths = {
            str(path.resolve()) for path in (
                Path("study-0"), Path("study-0/temporal"), Path("study-0/nasal"),
            )
        }
        opened_paths = set()
        open_directory = self.frame._open_result_directory
        for link in links:
            open_directory.reset_mock()
            self.click_action(canvas, link)
            open_directory.assert_called_once()
            opened_paths.add(str(Path(open_directory.call_args.args[0]).resolve()))
            for sequence in ("<Return>", "<space>"):
                open_directory.reset_mock()
                canvas._focus_action(link)
                canvas.focus_force()
                self.settle()
                canvas.event_generate(sequence)
                self.settle()
                open_directory.assert_called_once()
        self.assertEqual(opened_paths, expected_paths)

    def test_reversed_side_inputs_keep_folder_headers_and_images_in_the_same_order(self):
        images = {}
        for side_index, side in enumerate(module.Step3Frame.RESULT_SIDE_ORDER):
            for column, (_title, filename) in enumerate(module.Step3Frame.RESULT_IMAGE_ROWS):
                image = self.images[column].copy()
                ImageDraw.Draw(image).rectangle(
                    (75, 75, 125, 125), fill="red" if side_index == 0 else "green",
                )
                images[side, filename] = image

        self.frame._load_result_png_from_folder = lambda folder, filename: images[
            Path(folder).name, filename
        ].copy()
        self.frame._open_result_directory = Mock()
        folders = [Path("side-order-study") / side for side in reversed(module.Step3Frame.RESULT_SIDE_ORDER)]
        self.frame._render_result_grid(folders)
        self.settle()
        canvas = self.frame.canvas
        page = canvas._pages[0]
        texts = [canvas.itemcget(item, "text") for item in page["items"] if canvas.type(item) == "text"]
        for pair_index, side in enumerate(module.Step3Frame.RESULT_SIDE_ORDER):
            side_folder = (Path("side-order-study") / side).resolve()
            self.assertIn(f"{side.title()} — {side_folder}", texts)
            link = next(
                action for action in self.actions(canvas, 0, "folder")
                if len(action["key"]) == 3 and action["key"][1] == pair_index
            )
            self.frame._open_result_directory.reset_mock()
            self.click_action(canvas, link)
            self.frame._open_result_directory.assert_called_once_with(side_folder)
            for column, (title, filename) in enumerate(module.Step3Frame.RESULT_IMAGE_ROWS):
                preview = page["previews"][pair_index * 2 + column]
                self.assertEqual(preview["pair_index"], pair_index)
                self.assertIn(side.title(), preview["title"])
                self.assertIn(title, preview["title"])
                self.assertEqual(preview["image"].tobytes(), images[side, filename].tobytes())

    def test_directory_handler_opens_the_resolved_folder(self):
        folder = Path.cwd()
        self.frame.winfo_toplevel = Mock(return_value=self.root)
        with patch.object(module.os, "startfile", create=True) as startfile, \
                patch.object(module.subprocess, "Popen") as popen, \
                patch.object(module.messagebox, "showerror") as error:
            self.frame._open_result_directory(folder)
        if module.os.name == "nt":
            startfile.assert_called_once_with(str(folder.resolve()))
            popen.assert_not_called()
        else:
            command = "open" if module.sys.platform == "darwin" else "xdg-open"
            popen.assert_called_once_with([command, str(folder.resolve())])
            startfile.assert_not_called()
        error.assert_not_called()

    def test_directory_handler_reports_a_missing_folder_without_opening_it(self):
        self.frame.winfo_toplevel = Mock(return_value=self.root)
        with patch.object(Path, "is_dir", return_value=False), \
                patch.object(module.os, "startfile", create=True) as startfile, \
                patch.object(module.subprocess, "Popen") as popen, \
                patch.object(module.messagebox, "showerror") as error:
            self.frame._open_result_directory(Path("missing-result-folder"))
        startfile.assert_not_called()
        popen.assert_not_called()
        error.assert_called_once()
        self.assertIn("missing-result-folder", error.call_args.args[1])

    def test_reverse_tab_works_when_tk_rejects_the_x11_keysym(self):
        native_bind = tk.Canvas.bind
        rejected = []

        def windows_bind(canvas, sequence=None, func=None, add=None):
            if sequence == "<ISO_Left_Tab>":
                rejected.append(sequence)
                raise tk.TclError('bad event type or keysym "ISO_Left_Tab"')
            return native_bind(canvas, sequence, func, add)

        with patch.object(module.ResultGridCanvas, "bind", new=windows_bind):
            canvas = self.open_grid(count=3)
        self.assertEqual(rejected, ["<ISO_Left_Tab>"])
        actions = canvas._pages[0]["actions"]
        canvas.focus_force()
        self.settle()
        canvas._focus_action(actions[1])
        canvas.event_generate("<Shift-Tab>")
        self.settle()
        self.assertEqual(canvas._focus_key, actions[0]["key"])
        self.assertEqual(canvas._on_tab(reverse=True), "break")

    def test_keyboard_traversal_retains_focus_and_scrolls_to_next_subject(self):
        canvas = self.open_grid(count=4)
        page_actions = canvas._pages[0]["actions"]
        self.assertTrue(page_actions)
        self.assertTrue(bool(canvas.cget("takefocus")))
        canvas.focus_force()
        self.settle()
        self.assertEqual(canvas._focus_key, page_actions[0]["key"])
        canvas.event_generate("<Tab>")
        self.settle()
        self.assertEqual(canvas._focus_key, page_actions[1]["key"])
        canvas.event_generate("<Shift-Tab>")
        self.settle()
        self.assertEqual(canvas._focus_key, page_actions[0]["key"])
        canvas._focus_action(page_actions[-1])
        canvas.event_generate("<Tab>")
        self.settle()
        self.assertIn(1, canvas._pages)
        self.assertEqual(canvas._focus_key, canvas._pages[1]["actions"][0]["key"])
        self.assertGreater(canvas.yview()[0], 0)
        self.assertIn(1, canvas._visible_pages)

    def test_resizing_preserves_zoom_sources_and_reuses_unchanged_previews(self):
        canvas = self.open_grid(count=8)
        page = canvas._pages[0]
        original_pixels = [preview["image"].tobytes() for preview in page["previews"]]
        old_size = canvas._page_size
        self.root.geometry("740x520+10000+10000")
        self.settle()
        self.assertNotEqual(canvas._page_size, old_size)
        self.assertLessEqual(len(canvas._pages), canvas.CACHE_PAGES)
        previews = canvas._pages[0]["previews"]
        self.assertEqual([preview["image"].tobytes() for preview in previews], original_pixels)
        for preview in previews:
            self.assertEqual(ImageTk.getimage(preview["photo"]).size, preview["target"][2:])
        reads = len(self.reads)
        with patch.object(module.ImageTk, "PhotoImage", wraps=ImageTk.PhotoImage) as upload:
            for _ in range(10):
                canvas.event_generate("<Configure>", width=canvas.winfo_width(), height=canvas.winfo_height())
            self.settle()
            upload.assert_not_called()
        self.assertEqual(len(self.reads), reads)

    def test_image_preparation_uses_a_worker_and_tk_uploads_stay_on_main_thread(self):
        main_thread = threading.get_ident()
        upload_threads, canvas_threads, resize_threads = [], [], []
        original_photo = ImageTk.PhotoImage
        original_create_image = tk.Canvas.create_image
        original_resize = Image.Image.resize

        def upload(*args, **kwargs):
            upload_threads.append(threading.get_ident())
            return original_photo(*args, **kwargs)

        def create_image(canvas, *args, **kwargs):
            canvas_threads.append(threading.get_ident())
            return original_create_image(canvas, *args, **kwargs)

        def resize(image, size, *args, **kwargs):
            if image.size in {source.size for source in self.images}:
                resize_threads.append(threading.get_ident())
            return original_resize(image, size, *args, **kwargs)

        with patch.object(module.ImageTk, "PhotoImage", side_effect=upload), \
                patch.object(tk.Canvas, "create_image", autospec=True, side_effect=create_image), \
                patch.object(Image.Image, "resize", autospec=True, side_effect=resize):
            canvas = self.open_grid(count=3)
            self.assertTrue(all(canvas._pages[index]["ready"] for index in canvas._visible_pages))
        self.assertTrue(self.read_threads)
        self.assertTrue(resize_threads)
        self.assertNotIn(main_thread, self.read_threads)
        self.assertNotIn(main_thread, resize_threads)
        self.assertEqual(set(upload_threads), {main_thread})
        self.assertEqual(set(canvas_threads), {main_thread})

    def test_pending_worker_does_not_block_tk_and_cannot_install_after_teardown(self):
        started, release = threading.Event(), threading.Event()

        def block_load(_folder, _filename):
            started.set()
            if not release.wait(timeout=5):
                raise TimeoutError("Test image loader was not released")

        canvas = self.open_grid(count=20, on_load=block_load, wait=False)
        try:
            deadline = time.monotonic() + 5
            while not started.is_set():
                self.root.update_idletasks()
                self.root.update()
                if time.monotonic() >= deadline:
                    self.fail("Image worker did not start")
                time.sleep(0.005)
            future = canvas._render_future[2]
            callback = Mock()
            self.root.after_idle(callback)
            self.root.update_idletasks()
            callback.assert_called_once()
            self.assertTrue(canvas._pages)
            self.assertFalse(any(page["ready"] for page in canvas._pages.values()))
            for fraction in (0.25, 0.5, 0.75, 1.0):
                canvas.yview_moveto(fraction)
                self.root.update_idletasks()
                self.root.update()
                self.assertLessEqual(len(canvas._pages), canvas.CACHE_PAGES)
                self.assertLessEqual(len(canvas._render_requests), len(canvas._visible_pages))
                self.assertTrue(set(canvas._render_requests).issubset(canvas._visible_pages))
            with patch.object(canvas, "_install_previews", wraps=canvas._install_previews) as install:
                canvas.scroll_mousewheel(SimpleNamespace(delta=-120, num=None))
                canvas.event_generate("<Configure>", width=740, height=520)
                self.frame._clear_plot_holder()
                release.set()
                future.result(timeout=5)
                self.settle()
                install.assert_not_called()
            self.assertIsNone(canvas._poll_after_id)
            self.assertIsNone(canvas._refresh_after_id)
            self.assertIsNone(canvas._wheel_after_id)
            self.assertFalse(canvas._render_requests)
            self.assertFalse(canvas._pages)
        finally:
            release.set()

    def test_resizing_during_image_preparation_installs_only_current_size(self):
        started, release = threading.Event(), threading.Event()

        def block_load(_folder, _filename):
            started.set()
            if not release.wait(timeout=5):
                raise TimeoutError("Test image loader was not released")

        canvas = self.open_grid(count=3, on_load=block_load, wait=False)
        try:
            deadline = time.monotonic() + 5
            while not started.is_set():
                self.root.update_idletasks()
                self.root.update()
                if time.monotonic() >= deadline:
                    self.fail("Image worker did not start")
                time.sleep(0.005)
            generation = canvas._generation
            old_size = canvas._page_size
            self.root.geometry("740x520+10000+10000")
            self.root.update_idletasks()
            self.root.update()
            self.assertGreater(canvas._generation, generation)
            self.assertNotEqual(canvas._page_size, old_size)
            release.set()
            self.settle()
            for index in canvas._visible_pages:
                page = canvas._pages[index]
                self.assertTrue(page["ready"])
                for preview in page["previews"]:
                    x, y, width, height = preview["target"]
                    self.assertLessEqual(x + width, canvas.winfo_width())
                    self.assertEqual(ImageTk.getimage(preview["photo"]).size, (width, height))
                    self.assertEqual(canvas.coords(preview["item"]), [float(x), float(y)])
        finally:
            release.set()

    def test_wheel_burst_matches_native_travel_with_one_viewport_update(self):
        canvas = self.open_grid()
        reference = tk.Canvas(self.root, borderwidth=0, highlightthickness=0)
        reference.place(x=0, y=0, width=canvas.winfo_width(), height=canvas.winfo_height())
        reference.configure(scrollregion=canvas.cget("scrollregion"))
        self.settle()
        with patch.object(canvas, "yview_moveto", wraps=canvas.yview_moveto) as move:
            for delta in (-120, -120, 120, -120, -120, 120, -120):
                canvas.scroll_mousewheel(SimpleNamespace(delta=delta, num=None))
                reference.yview_scroll(-3 if delta > 0 else 3, "units")
            move.assert_not_called()
            self.settle()
            self.assertEqual(move.call_count, 1)
        self.assertEqual(canvas.yview(), reference.yview())
        canvas.scroll_mousewheel(SimpleNamespace(delta=-120, num=None))
        canvas.yview_moveto(0.5)
        self.settle()
        self.assertAlmostEqual(canvas.yview()[0], 0.5, places=3)
        self.assertIsNone(canvas._wheel_after_id)
        reference.destroy()

    def test_replacing_view_cancels_queued_scrolling_and_preview_draws(self):
        canvas = self.open_grid()
        canvas.scroll_mousewheel(SimpleNamespace(delta=-120, num=None))
        self.assertIsNotNone(canvas._wheel_after_id)
        canvas.event_generate("<Configure>", width=740, height=520)
        self.assertIsNotNone(canvas._refresh_after_id)
        self.frame._clear_plot_holder()
        self.settle()
        self.assertIsNone(canvas._wheel_after_id)
        self.assertIsNone(canvas._refresh_after_id)
        self.assertFalse(canvas._pages)
        label = self.frame._display_preview_image(self.images[0])
        label.destroy()
        self.settle()


if __name__ == "__main__":
    unittest.main()
