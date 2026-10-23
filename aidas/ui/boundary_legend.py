"""Read-only color key for retinal boundaries and the foveal marker."""

import customtkinter as ctk

from aidas.ui.theme import COLOR_PAIRS, SHAPES, TYPOGRAPHY


class BoundaryLegend(ctk.CTkFrame):
    """Display the map's colors in rounded, non-interactive legend rows."""

    def __init__(self, master, boundaries, *, fovea_color="#ffd500"):
        super().__init__(master, fg_color="transparent", corner_radius=0)
        ctk.CTkLabel(
            self, text="Boundary legend", anchor="w", text_color=COLOR_PAIRS["text"],
            font=ctk.CTkFont(family=TYPOGRAPHY.family, size=TYPOGRAPHY.body_size,
                           weight=TYPOGRAPHY.semibold_weight),
        ).pack(fill="x", pady=(4, 3))
        self.rows = {}
        for name, color, vertical in [
            *((name, color, False) for name, color in boundaries),
            ("Fovea center", fovea_color, True),
        ]:
            row = ctk.CTkFrame(
                self, height=30, fg_color=COLOR_PAIRS["surface_subtle"],
                corner_radius=SHAPES.corner_radius_sm,
                border_width=1, border_color=COLOR_PAIRS["border"],
            )
            row.pack(fill="x", pady=2)
            swatch = ctk.CTkFrame(row, width=40, height=26, fg_color="transparent")
            swatch.pack(side="left", padx=(4, 6), pady=2)
            if vertical:
                for y in (5, 13, 21):
                    ctk.CTkFrame(swatch, width=3, height=5, corner_radius=0,
                                 fg_color=color).place(relx=0.5, y=y, anchor="center")
            else:
                ctk.CTkFrame(swatch, width=24, height=3, corner_radius=1,
                             fg_color=color).place(relx=0.5, rely=0.5, anchor="center")
            ctk.CTkLabel(
                row, text=name, anchor="w", text_color=COLOR_PAIRS["text"],
                font=ctk.CTkFont(family=TYPOGRAPHY.family, size=TYPOGRAPHY.body_size),
            ).pack(side="left", fill="x", expand=True, padx=(0, 8))
            self.rows[name] = row
