from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

import customtkinter as ctk

from aidas.steps.step1_resize_raw import SDB_PREF_KEY, Step1Frame


class _QuietRoot(ctk.CTk):
    _deactivate_windows_window_header_manipulation = True


class Step1DirectoryEntryTests(unittest.TestCase):
    def setUp(self):
        self.root = _QuietRoot()
        self.root.attributes("-alpha", 0)
        self.root.geometry("1100x800+0+0")
        self.preferences = mock.Mock()
        self.preferences.get.side_effect = lambda _key, default=None: default
        with mock.patch.object(Step1Frame, "refresh_sdb_list"):
            self.frame = Step1Frame(self.root, preferences=self.preferences)
        self.frame.pack(fill="both", expand=True)
        self.root.update()
        self.temporary = tempfile.TemporaryDirectory()
        self.original = Path(self.temporary.name)
        self.target = self.original / "new folder with spaces"
        self.target.mkdir()
        self.frame.set_sdb_directory(str(self.original))
        self.preferences.set.reset_mock()
        self.entry = self.frame.sdb_dir_entry._entry
        self.entry.focus_force()
        self.root.update()

    def tearDown(self):
        self.root.destroy()
        self.temporary.cleanup()

    def _type_address(self, address):
        self.entry.delete(0, "end")
        self.entry.insert(0, address)

    def test_editing_is_a_draft_and_enter_commits_then_scans_the_valid_folder(self):
        self._type_address(f'  "{self.target}"  ')
        self.assertEqual(self.frame.sdb_dir_var.get(), str(self.original))
        self.preferences.set.assert_not_called()
        with mock.patch.object(self.frame, "refresh_sdb_list") as scan:
            self.entry.event_generate("<Return>")
            self.root.update()
        self.assertEqual(self.frame.sdb_dir_var.get(), str(self.target.resolve()))
        self.assertEqual(self.frame.sdb_dir_edit_var.get(), str(self.target.resolve()))
        self.preferences.set.assert_called_once_with(SDB_PREF_KEY, str(self.target.resolve()))
        scan.assert_called_once_with(preview_first=True)

    def test_invalid_address_keeps_the_active_folder_and_does_not_scan(self):
        file = self.original / "image.sdb"
        file.touch()
        for address in ("", str(self.original / "missing"), str(file)):
            with self.subTest(address=address):
                self._type_address(address)
                with (
                    mock.patch("aidas.steps.step1_resize_raw.messagebox.showerror") as error,
                    mock.patch.object(self.frame, "refresh_sdb_list") as scan,
                ):
                    self.entry.event_generate("<Return>")
                    self.root.update()
                error.assert_called_once()
                scan.assert_not_called()
                self.assertEqual(self.frame.sdb_dir_var.get(), str(self.original))
                self.assertEqual(self.frame.sdb_dir_edit_var.get(), address)
        self.preferences.set.assert_not_called()

    def test_first_enter_accepts_suggestion_and_second_enter_commits_and_scans(self):
        controller = self.frame.sdb_dir_entry._path_completion
        with mock.patch.object(self.frame, "refresh_sdb_list") as scan:
            self._type_address(str(self.original / "new"))
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                self.root.update()
                if (controller.popup is not None and controller.popup.winfo_viewable()
                        and [name for name, _path in controller.suggestions] == [self.target.name]):
                    break
                time.sleep(0.005)
            self.assertEqual([name for name, _path in controller.suggestions], [self.target.name])
            self.assertTrue(controller.popup.winfo_viewable())
            self.entry.event_generate("<Down>")
            self.root.update()
            self.assertEqual(controller.listbox.curselection(), (0,))

            self.entry.event_generate("<Return>")
            self.root.update()

            self.assertEqual(Path(self.frame.sdb_dir_edit_var.get()).resolve(), self.target.resolve())
            self.assertEqual(self.frame.sdb_dir_var.get(), str(self.original))
            self.assertFalse(controller.popup.winfo_ismapped())
            self.assertIs(self.root.focus_get(), self.entry)
            self.preferences.set.assert_not_called()
            scan.assert_not_called()

            self.entry.event_generate("<Return>")
            self.root.update()

            self.assertEqual(self.frame.sdb_dir_var.get(), str(self.target.resolve()))
            self.preferences.set.assert_called_once_with(SDB_PREF_KEY, str(self.target.resolve()))
            scan.assert_called_once_with(preview_first=True)

    def test_browse_updates_the_address_and_starts_the_existing_scan_flow(self):
        self._type_address("unfinished draft")
        with (
            mock.patch("aidas.steps.step1_resize_raw.filedialog.askdirectory", return_value=str(self.target)),
            mock.patch.object(self.frame, "refresh_sdb_list") as scan,
        ):
            self.frame._browse_sdb_dir()
        self.assertEqual(self.frame.sdb_dir_var.get(), str(self.target))
        self.assertEqual(self.frame.sdb_dir_edit_var.get(), str(self.target))
        scan.assert_called_once_with(preview_first=True)

    def _open_scrolling_folder_completion(self):
        sidebar = self.frame.sidebar
        spacer = ctk.CTkFrame(sidebar.content, height=1200)
        spacer.pack(fill="x")
        self.root.update()
        sidebar.refresh_scrollregion(settle=True)
        sidebar.canvas.yview_moveto(0)
        self.root.update()
        self.assertLess(sidebar.canvas.yview()[1], 1.0)
        for index in range(25):
            (self.original / f"wheel {index:02d}").mkdir()
        self.entry.focus_force()
        self.root.update()
        self._type_address(str(self.original / "wheel"))
        controller = self.frame.sdb_dir_entry._path_completion
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            self.root.update()
            if (controller.popup is not None and controller.popup.winfo_viewable()
                    and len(controller.suggestions) == 25):
                break
            time.sleep(0.005)
        self.assertEqual(len(controller.suggestions), 25)
        self.assertTrue(controller.popup.winfo_viewable())
        self.assertLess(controller.listbox.yview()[1], 1.0)
        return sidebar, controller

    def test_wheeling_folder_suggestions_scrolls_only_the_dropdown_and_preserves_its_anchor(self):
        sidebar, controller = self._open_scrolling_folder_completion()
        sidebar_view = sidebar.canvas.yview()
        list_view = controller.listbox.yview()
        entry_position = (self.entry.winfo_rootx(), self.entry.winfo_rooty())
        popup_geometry = (
            controller.popup.winfo_rootx(), controller.popup.winfo_rooty(),
            controller.popup.winfo_width(), controller.popup.winfo_height(),
        )

        with mock.patch.object(sidebar, "_contains_scroll_area_pointer", return_value=True):
            controller.listbox.event_generate("<MouseWheel>", delta=-120, x=5, y=5)
            self.root.update()

        self.assertGreater(controller.listbox.yview()[0], list_view[0])
        self.assertEqual(sidebar.canvas.yview(), sidebar_view)
        self.assertEqual((self.entry.winfo_rootx(), self.entry.winfo_rooty()), entry_position)
        self.assertEqual(
            (controller.popup.winfo_rootx(), controller.popup.winfo_rooty(),
             controller.popup.winfo_width(), controller.popup.winfo_height()),
            popup_geometry,
        )
        self.assertTrue(controller.popup.winfo_ismapped())

    def test_scrolling_the_sidebar_dismisses_suggestions_when_the_address_moves(self):
        sidebar, controller = self._open_scrolling_folder_completion()
        entry_position = (self.entry.winfo_rootx(), self.entry.winfo_rooty())
        draft = self.frame.sdb_dir_edit_var.get()

        sidebar.canvas.yview_scroll(1, "units")
        self.root.update()

        self.assertNotEqual((self.entry.winfo_rootx(), self.entry.winfo_rooty()), entry_position)
        self.assertFalse(controller.popup.winfo_ismapped())
        self.assertEqual(self.frame.sdb_dir_edit_var.get(), draft)
        self.assertEqual(self.frame.sdb_dir_var.get(), str(self.original))
        self.preferences.set.assert_not_called()
