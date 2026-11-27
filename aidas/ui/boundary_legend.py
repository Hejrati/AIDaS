"""Read-only color key for retinal boundaries and the foveal marker."""

import customtkinter as ctk
from PIL import Image, ImageDraw

from aidas.ui.images import OwnedCTkImage
from aidas.ui.theme import COLOR_PAIRS, TYPOGRAPHY


class BoundaryLegend(ctk.CTkFrame):
    """Show read-only samples of the boundary traces and fovea marker."""

    def __init__(self, master, boundaries, *, fovea_color="#ffd500"):
        super().__init__(master, fg_color="transparent", corner_radius=0)
        ctk.CTkLabel(
            self, text="Boundary legend", anchor="w", text_color=COLOR_PAIRS["text"],
            font=ctk.CTkFont(family=TYPOGRAPHY.family, size=TYPOGRAPHY.body_size,
                           weight=TYPOGRAPHY.semibold_weight),
        ).pack(fill="x", pady=(4, 3))
        font = ctk.CTkFont(family=TYPOGRAPHY.family, size=TYPOGRAPHY.body_size)
        self.rows = {}
        entries = [(name, color, False) for name, color in boundaries]
        entries.append(("Fovea center", fovea_color, True))
        for name, color, fovea in entries:
            row = ctk.CTkLabel(
                self, text=name, image=self._symbol(color, fovea=fovea),
                compound="left", anchor="w", fg_color="transparent", height=24,
                text_color=COLOR_PAIRS["text"], font=font, takefocus=False,
            )
            row.pack(fill="x")
            self.rows[name] = row

    def _symbol(self, color, *, fovea):
        width, height = 32, 18
        scale = 4
        image = Image.new("RGBA", (width * scale, height * scale))
        pen = ImageDraw.Draw(image)
        if fovea:
            x = 14 * scale
            for y in (1, 7, 13):
                pen.line((x, y * scale, x, (y + 4) * scale), fill=color, width=2 * scale)
        else:
            points = [(3, 12), (10, 8), (17, 11), (25, 6)]
            pen.line([(x * scale, y * scale) for x, y in points],
                     fill=color, width=2 * scale, joint="curve")
        return OwnedCTkImage(self, light_image=image, size=(width, height))
