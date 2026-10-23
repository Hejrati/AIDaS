"""Cache bundled icon pixels without sharing mutable images or Tk owners."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

from PIL import Image

from aidas.utils import ui_utils


class UIIconCacheTests(unittest.TestCase):
    def setUp(self):
        self._clear_caches()
        self.addCleanup(self._clear_caches)
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "icon.png"
        source = Image.new("RGBA", (3, 1))
        source.putdata(((5, 5, 5, 255), (255, 0, 0, 128), (0, 0, 0, 0)))
        source.save(self.path)
        patcher = mock.patch.object(ui_utils, "asset_path", return_value=str(self.path))
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def _clear_caches():
        ui_utils._load_ctk_icon_variants.cache_clear()
        ui_utils._load_ctk_icon_source.cache_clear()

    def test_repeated_sizes_reuse_decoding_and_theme_conversion(self):
        owner = SimpleNamespace()
        with mock.patch.object(ui_utils.Image, "open", wraps=Image.open) as opened, mock.patch.object(
                ui_utils, "_dark_mode_icon", wraps=ui_utils._dark_mode_icon) as darkened:
            small = ui_utils.load_ctk_image(owner, "icon.png", size=20)
            large = ui_utils.load_ctk_image(owner, "icon.png", size=(24, 18))
        self.assertEqual(opened.call_count, 1)
        self.assertEqual(darkened.call_count, 1)
        self.assertEqual(small.cget("size"), (20, 20))
        self.assertEqual(large.cget("size"), (24, 18))
        self.assertEqual(small.cget("light_image").getpixel((0, 0)), (5, 5, 5, 255))
        self.assertEqual(small.cget("dark_image").getpixel((0, 0)), (221, 231, 240, 255))
        self.assertEqual(small.cget("dark_image").getpixel((1, 0)), (255, 0, 0, 128))

    def test_tints_share_decoded_source_and_normalize_list_tuple_keys(self):
        owner = SimpleNamespace()
        with mock.patch.object(ui_utils.Image, "open", wraps=Image.open) as opened, mock.patch.object(
                ui_utils, "_tint_icon", wraps=ui_utils._tint_icon) as tinted:
            ui_utils.load_ctk_image(owner, "icon.png")
            first = ui_utils.load_ctk_image(owner, "icon.png", tint=["#123456", "#ABCDEF"])
            second = ui_utils.load_ctk_image(owner, "icon.png", tint=("#123456", "#ABCDEF"))
            single = ui_utils.load_ctk_image(owner, "icon.png", tint="#010203")
        self.assertEqual(opened.call_count, 1)
        self.assertEqual(tinted.call_count, 4)
        for icon in (first, second):
            self.assertEqual(icon.cget("light_image").getpixel((1, 0)), (18, 52, 86, 128))
            self.assertEqual(icon.cget("dark_image").getpixel((1, 0)), (171, 205, 239, 128))
        self.assertEqual(single.cget("light_image").getpixel((1, 0)), (1, 2, 3, 128))
        self.assertEqual(single.cget("dark_image").getpixel((1, 0)), (1, 2, 3, 128))

    def test_each_icon_has_its_own_pixels_and_owner(self):
        first_owner, second_owner = SimpleNamespace(), SimpleNamespace()
        first = ui_utils.load_ctk_image(first_owner, "icon.png")
        second = ui_utils.load_ctk_image(second_owner, "icon.png")
        self.assertIsNot(first, second)
        self.assertIs(first._aidas_image_master, first_owner)
        self.assertIs(second._aidas_image_master, second_owner)
        for mode in ("light_image", "dark_image"):
            original = second.cget(mode).tobytes()
            first.cget(mode).paste((0, 255, 0, 255), (0, 0, 3, 1))
            later = ui_utils.load_ctk_image(SimpleNamespace(), "icon.png")
            self.assertEqual(second.cget(mode).tobytes(), original)
            self.assertEqual(later.cget(mode).tobytes(), original)
        self.assertEqual(first_owner._ui_image_refs, [first])
        self.assertEqual(second_owner._ui_image_refs, [second])

    def test_different_paths_do_not_share_pixels_and_failures_are_not_cached(self):
        second_path = self.path.with_name("other.png")
        Image.new("RGBA", (1, 1), "blue").save(second_path)
        with mock.patch.object(ui_utils, "asset_path", side_effect=[str(self.path), str(second_path)]):
            first = ui_utils.load_ctk_image(SimpleNamespace(), "icon.png")
            second = ui_utils.load_ctk_image(SimpleNamespace(), "other.png")
        self.assertEqual(first.cget("light_image").getpixel((0, 0)), (5, 5, 5, 255))
        self.assertEqual(second.cget("light_image").getpixel((0, 0)), (0, 0, 255, 255))

        missing_path = self.path.with_name("missing.png")
        with mock.patch.object(ui_utils, "asset_path", return_value=str(missing_path)):
            with self.assertRaises(FileNotFoundError):
                ui_utils.load_ctk_image(SimpleNamespace(), "missing.png")
            Image.new("RGBA", (1, 1), "red").save(missing_path)
            recovered = ui_utils.load_ctk_image(SimpleNamespace(), "missing.png")
        self.assertEqual(recovered.cget("light_image").getpixel((0, 0)), (255, 0, 0, 255))


if __name__ == "__main__":
    unittest.main()
