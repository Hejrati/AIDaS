"""Fixed-appearance startup splash for AIDaS."""

from __future__ import annotations

from pathlib import Path
import tkinter as tk

import customtkinter as ctk
from PIL import Image

from aidas.core.display import fit_size_to_bounds, work_area_bounds
from aidas.ui.theme import COLOR_PAIRS, TYPOGRAPHY, get_interface_mode, resolve_color
from aidas.ui.windowing import centered_logical_geometry, physical_window_size
from aidas.utils.ui_utils import apply_app_icon_to


SPLASH_APPEARANCE_MODE = "Light"


def _splash_color(name: str) -> str:
    """Resolve one semantic token to the splash's permanent appearance."""

    return resolve_color(COLOR_PAIRS[name], SPLASH_APPEARANCE_MODE)


class SplashWindow(ctk.CTkToplevel):
    """Clear, fixed-light startup window with live progress reporting."""

    WIDTH = 480
    HEIGHT = 548
    MAX_SCREEN_FRACTION = 0.9

    def __init__(
        self,
        parent: tk.Misc,
        *,
        logo_path: str,
        title: str,
        subtitle: str,
        affiliation: str,
        lab_name: str,
        copyright_notice: str,
    ) -> None:
        super().__init__(parent)
        self.withdraw()
        apply_app_icon_to(self)
        self.overrideredirect(True)
        # The splash is undecorated, so its root and full-bleed panel use the
        # same fixed surface. This avoids a dark/contrasting one-pixel seam.
        self.configure(fg_color=_splash_color("surface"))

        bounds = work_area_bounds(self)
        design_width, design_height = physical_window_size(self, self.WIDTH, self.HEIGHT)
        _width, _height, self.scale = fit_size_to_bounds(
            bounds,
            design_width,
            design_height,
            maximum_fraction=self.MAX_SCREEN_FRACTION,
        )
        splash_width = max(1, round(self.WIDTH * self.scale))
        splash_height = max(1, round(self.HEIGHT * self.scale))

        def spacing(value: int) -> int:
            return max(1, round(value * self.scale))

        panel = ctk.CTkFrame(
            self,
            fg_color=_splash_color("surface"),
            corner_radius=0,
            border_width=0,
        )
        panel.pack(fill="both", expand=True)
        panel.grid_columnconfigure(0, weight=1)
        panel.grid_rowconfigure(6, weight=1)
        # Preserve the established vertical geometry without drawing a top
        # accent/boundary against the outer edge.
        panel.grid_rowconfigure(0, minsize=max(3, spacing(4)))

        logo_size = spacing(148)
        logo_file = Path(logo_path)
        if logo_file.is_file():
            with Image.open(logo_file) as logo:
                logo_source = logo.convert("RGBA").copy()
            self.logo_image = ctk.CTkImage(
                light_image=logo_source,
                dark_image=logo_source,
                size=(logo_size, logo_size),
            )
            ctk.CTkLabel(panel, text="", image=self.logo_image).grid(
                row=1,
                column=0,
                pady=(spacing(22), 0),
            )
        else:
            self.logo_image = None
            ctk.CTkLabel(
                panel,
                text="A",
                width=logo_size,
                height=logo_size,
                corner_radius=logo_size // 2,
                fg_color=_splash_color("primary_soft"),
                text_color=_splash_color("primary"),
                font=ctk.CTkFont(
                    family=TYPOGRAPHY.family,
                    size=TYPOGRAPHY.display_size,
                    weight=TYPOGRAPHY.bold_weight,
                ),
            ).grid(row=1, column=0, pady=(spacing(22), 0))

        ctk.CTkLabel(
            panel,
            text=title,
            text_color=_splash_color("text"),
            font=ctk.CTkFont(
                family=TYPOGRAPHY.family,
                size=TYPOGRAPHY.display_size,
                weight=TYPOGRAPHY.bold_weight,
            ),
        ).grid(row=2, column=0, pady=(spacing(6), 0))
        ctk.CTkLabel(
            panel,
            text=subtitle,
            text_color=_splash_color("muted_text"),
            font=ctk.CTkFont(
                family=TYPOGRAPHY.family,
                size=TYPOGRAPHY.subtitle_size,
            ),
        ).grid(row=3, column=0)
        ctk.CTkLabel(
            panel,
            text=affiliation,
            text_color=_splash_color("institution"),
            font=ctk.CTkFont(
                family=TYPOGRAPHY.family,
                size=TYPOGRAPHY.body_size,
                weight=TYPOGRAPHY.semibold_weight,
            ),
        ).grid(row=4, column=0, pady=(spacing(14), 0))
        ctk.CTkLabel(
            panel,
            text=lab_name,
            text_color=_splash_color("text"),
            font=ctk.CTkFont(
                family=TYPOGRAPHY.family,
                size=TYPOGRAPHY.body_size,
            ),
            justify="center",
            wraplength=max(1, splash_width - spacing(48)),
        ).grid(row=5, column=0, sticky="ew", padx=spacing(24), pady=(spacing(3), 0))

        loading_region = ctk.CTkFrame(
            panel,
            fg_color="transparent",
            corner_radius=0,
            border_width=0,
        )
        loading_region.grid(
            row=6,
            column=0,
            sticky="sew",
            padx=spacing(30),
            pady=spacing(16),
        )
        loading_region.grid_columnconfigure(0, weight=1)
        progress_header = ctk.CTkFrame(
            loading_region,
            fg_color="transparent",
            corner_radius=0,
        )
        progress_header.grid(
            row=0,
            column=0,
            sticky="ew",
            padx=spacing(14),
            pady=(spacing(11), spacing(6)),
        )
        progress_header.grid_columnconfigure(0, weight=1)

        self.status_var = tk.StringVar(master=self, value="Starting AIDaS...")
        self.percent_var = tk.StringVar(master=self, value="0%")
        self.status_label = ctk.CTkLabel(
            progress_header,
            textvariable=self.status_var,
            text_color=_splash_color("text"),
            anchor="w",
            justify="left",
            wraplength=max(spacing(120), splash_width - spacing(150)),
            font=ctk.CTkFont(
                family=TYPOGRAPHY.family,
                size=TYPOGRAPHY.body_size,
            ),
        )
        self.status_label.grid(row=0, column=0, sticky="ew", padx=(0, spacing(8)))
        ctk.CTkLabel(
            progress_header,
            textvariable=self.percent_var,
            text_color=_splash_color("primary"),
            width=46,
            anchor="e",
            font=ctk.CTkFont(
                family=TYPOGRAPHY.family,
                size=TYPOGRAPHY.body_size,
                weight=TYPOGRAPHY.semibold_weight,
            ),
        ).grid(row=0, column=1, sticky="e")

        self.progress = ctk.CTkProgressBar(
            loading_region,
            height=max(7, spacing(7)),
            corner_radius=0 if get_interface_mode() == "Classic" else spacing(4),
            fg_color=_splash_color("border"),
            progress_color=_splash_color("primary"),
        )
        self.progress.grid(
            row=1,
            column=0,
            sticky="ew",
            padx=spacing(14),
            pady=(0, spacing(13)),
        )
        self.progress.set(0.0)

        ctk.CTkLabel(
            panel,
            text=copyright_notice,
            text_color=_splash_color("muted_text"),
            font=ctk.CTkFont(
                family=TYPOGRAPHY.family,
                size=TYPOGRAPHY.caption_size,
            ),
            justify="center",
            wraplength=max(1, splash_width - spacing(56)),
        ).grid(row=7, column=0, sticky="ew", padx=spacing(28), pady=(0, spacing(17)))

        self.attributes("-topmost", True)
        self.geometry(centered_logical_geometry(self, splash_width, splash_height))
        self.deiconify()
        self.lift()

    def set_progress(self, value: float, message: str) -> None:
        """Update the visible startup stage and percentage immediately."""

        percent = max(0.0, min(float(value), 100.0))
        self.percent_var.set(f"{percent:.0f}%")
        self.status_var.set(str(message))
        self.progress.set(percent / 100.0)
        self.update_idletasks()


__all__ = ["SPLASH_APPEARANCE_MODE", "SplashWindow"]
