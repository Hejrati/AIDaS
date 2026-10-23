"""DPI-aware images attached to their owning Tk interpreter."""

import customtkinter as ctk
from PIL import ImageTk


class OwnedCTkImage(ctk.CTkImage):
    """Keep CTk's image scaling/cache behavior without using Tk's default root.

    CustomTkinter's PhotoImage factories omit ``master``, which selects the
    first Tk interpreter even when the displaying widget belongs to another.
    An image belongs to one interpreter, just like its owning widget.
    """

    def __init__(self, master, **kwargs):
        self._aidas_image_master = master
        super().__init__(**kwargs)

    def _get_scaled_light_photo_image(self, scaled_size):
        if scaled_size not in self._scaled_light_photo_images:
            self._scaled_light_photo_images[scaled_size] = ImageTk.PhotoImage(
                self._light_image.resize(scaled_size), master=self._aidas_image_master,
            )
        return self._scaled_light_photo_images[scaled_size]

    def _get_scaled_dark_photo_image(self, scaled_size):
        if scaled_size not in self._scaled_dark_photo_images:
            self._scaled_dark_photo_images[scaled_size] = ImageTk.PhotoImage(
                self._dark_image.resize(scaled_size), master=self._aidas_image_master,
            )
        return self._scaled_dark_photo_images[scaled_size]
