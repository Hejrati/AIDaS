"""Reusable CustomTkinter components for the AIDaS application shell."""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from typing import Callable, Sequence

import customtkinter as ctk
from PIL import Image

from aidas.ui.theme import COLOR_PAIRS, CONTROLS, SHAPES, TYPOGRAPHY
from aidas.utils.ui_utils import HoverToolTip


class AppButton(ctk.CTkButton):
    """A semantic CTk button with a small ttk-state compatibility shim."""

    _PALETTE_OPTIONS = (
        "fg_color",
        "hover_color",
        "text_color",
        "border_color",
        "border_width",
        "background_corner_colors",
    )
    _DISABLED_PALETTE = {
        "fg_color": COLOR_PAIRS["surface_subtle"],
        "hover_color": COLOR_PAIRS["surface_subtle"],
        "border_color": COLOR_PAIRS["border"],
        "border_width": SHAPES.border_width,
    }

    def __init__(self, master, *, variant: str = "secondary", **kwargs):
        legacy_style = str(kwargs.pop("style", ""))
        kwargs.pop("padding", None)
        if kwargs.get("image") is not None:
            kwargs.setdefault("compound", "left")
        if legacy_style in {"Accent.TButton", "AIDaS.Primary.TButton"}:
            variant = "primary"

        options = {
            "height": CONTROLS.height_md,
            "corner_radius": SHAPES.corner_radius_md,
            "border_width": SHAPES.border_width,
            "text_color_disabled": COLOR_PAIRS["disabled_text"],
            "font": ctk.CTkFont(
                family=TYPOGRAPHY.family,
                size=TYPOGRAPHY.body_size,
                weight=TYPOGRAPHY.semibold_weight,
            ),
        }
        if not isinstance(master, ctk.CTkBaseClass):
            # ttk parents expose only their creation-time background to CTk;
            # a dual color prevents stale light corners after switching dark.
            options["bg_color"] = COLOR_PAIRS["surface"]
        options.update(self._variant_palette(variant))
        options.update(kwargs)
        self._variant = variant
        self._enabled_palette = {
            name: options[name]
            for name in self._PALETTE_OPTIONS
            if name in options
        }
        if options.get("state") == "disabled":
            options.update(self._disabled_palette())
        super().__init__(master, **options)

    @staticmethod
    def _variant_palette(variant: str):
        palettes = {
            "primary": {
                "fg_color": COLOR_PAIRS["primary"],
                "hover_color": COLOR_PAIRS["primary_hover"],
                "text_color": COLOR_PAIRS["on_primary"],
                "border_color": COLOR_PAIRS["primary"],
            },
            "danger": {
                "fg_color": COLOR_PAIRS["danger"],
                "hover_color": COLOR_PAIRS["danger_hover"],
                "text_color": COLOR_PAIRS["on_primary"],
                "border_color": COLOR_PAIRS["danger"],
            },
            "success": {
                "fg_color": COLOR_PAIRS["success"],
                "hover_color": COLOR_PAIRS["success_hover"],
                "text_color": COLOR_PAIRS["on_primary"],
                "border_color": COLOR_PAIRS["success"],
            },
            "ghost": {
                "fg_color": "transparent",
                "hover_color": COLOR_PAIRS["primary_soft"],
                "text_color": COLOR_PAIRS["text"],
                "border_color": COLOR_PAIRS["border"],
            },
            "secondary": {
                "fg_color": COLOR_PAIRS["button"],
                "hover_color": COLOR_PAIRS["button_hover"],
                "text_color": COLOR_PAIRS["text"],
                "border_color": COLOR_PAIRS["border_strong"],
            },
        }
        return palettes.get(variant, palettes["secondary"])

    def set_variant(self, variant: str) -> None:
        """Change semantic emphasis while preserving state and geometry."""

        self._variant = variant
        self.configure(**self._variant_palette(variant))

    def _disabled_palette(self):
        """Return a neutral palette while preserving a composite's silhouette."""

        disabled = dict(self._DISABLED_PALETTE)
        # Interface tokens are selected before widgets are built, whereas this
        # class-level compatibility constant is initialized at import time.
        # Resolve the color values here so Classic launches use the v2 palette.
        disabled.update(
            fg_color=COLOR_PAIRS["surface_subtle"],
            hover_color=COLOR_PAIRS["surface_subtle"],
            border_color=COLOR_PAIRS["border"],
            border_width=SHAPES.border_width,
        )
        enabled_corners = self._enabled_palette.get("background_corner_colors")
        if enabled_corners is not None:
            enabled_fill = self._enabled_palette.get("fg_color")
            disabled_fill = disabled["fg_color"]
            disabled["background_corner_colors"] = tuple(
                disabled_fill if corner == enabled_fill else corner
                for corner in enabled_corners
            )
        return disabled

    def configure(self, require_redraw=False, **kwargs):
        """Keep semantic colors reserved for enabled, actionable states."""

        requested_state = kwargs.get("state")
        current_state = getattr(self, "_state", "normal")
        target_state = requested_state if requested_state is not None else current_state

        # Explicit palette changes describe the enabled appearance. This lets
        # callers restyle a disabled button without accidentally making it
        # look actionable before its state changes.
        for name in self._PALETTE_OPTIONS:
            if name in kwargs:
                self._enabled_palette[name] = kwargs[name]

        if target_state == "disabled":
            kwargs.update(self._disabled_palette())
        elif current_state == "disabled" and target_state != "disabled":
            kwargs.update(self._enabled_palette)

        return super().configure(require_redraw, **kwargs)

    config = configure

    def state(self, statespec=None):
        """Support the subset of ttk.Button.state used by workflow panels."""

        if statespec is None:
            return ("disabled",) if self.cget("state") == "disabled" else ()
        disabled = self.cget("state") == "disabled"
        for state in statespec:
            if state == "disabled":
                disabled = True
            elif state == "!disabled":
                disabled = False
        self.configure(state="disabled" if disabled else "normal")
        return ()


class AppSplitButton(ctk.CTkFrame):
    """Primary action with a conventional, two-part split-button shape."""

    # U+25BE is the standard filled dropdown indicator.  The previous U+2304
    # glyph rendered as a wide, curved ``v`` in Segoe UI and looked unrelated
    # to a native menu indicator.
    CHEVRON = "\u25be"
    CHEVRON_FONT_SIZE = 14
    DEFAULT_WIDTH = 198
    SEGMENT_WIDTH = 40

    def __init__(
        self,
        master,
        *,
        text: str,
        command: Callable[[], object],
        options_command: Callable[[], object],
        image=None,
        width: int = DEFAULT_WIDTH,
        height: int = CONTROLS.height_lg,
        bg_color=None,
    ) -> None:
        corner_radius = SHAPES.corner_radius_md
        total_width = max(self.SEGMENT_WIDTH + 48, int(width))
        divider_width = 1
        action_width = max(48, total_width - self.SEGMENT_WIDTH)
        exterior_color = bg_color or COLOR_PAIRS["surface"]

        super().__init__(
            master,
            width=total_width,
            height=height,
            corner_radius=0,
            border_width=0,
            bg_color=exterior_color,
            fg_color="transparent",
        )
        self.grid_propagate(False)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1, minsize=action_width)
        self.grid_columnconfigure(1, weight=0, minsize=self.SEGMENT_WIDTH)

        # CustomTkinter uses this same background-corner technique for its
        # segmented button.  The exterior corners remain rounded while the
        # two inside edges are filled square, producing one clean silhouette
        # instead of two rounded buttons placed beside one another.
        self.action_button = AppButton(
            self,
            text=text,
            variant="primary",
            width=action_width,
            height=height,
            corner_radius=corner_radius,
            border_width=0,
            bg_color=exterior_color,
            background_corner_colors=(
                exterior_color,
                COLOR_PAIRS["primary"],
                COLOR_PAIRS["primary"],
                exterior_color,
            ),
            command=command,
            image=image,
            compound="left",
        )
        self.action_button.grid(row=0, column=0, sticky="nsew")

        self.divider = ctk.CTkFrame(
            self,
            width=divider_width,
            height=max(1, int(height) - 16),
            corner_radius=0,
            fg_color=COLOR_PAIRS["on_primary"],
        )

        self.options_button = AppButton(
            self,
            text=self.CHEVRON,
            variant="primary",
            width=self.SEGMENT_WIDTH,
            height=height,
            corner_radius=corner_radius,
            border_width=0,
            bg_color=exterior_color,
            background_corner_colors=(
                COLOR_PAIRS["primary"],
                exterior_color,
                exterior_color,
                COLOR_PAIRS["primary"],
            ),
            command=options_command,
            anchor="center",
            font=ctk.CTkFont(
                family="Segoe UI Symbol",
                size=self.CHEVRON_FONT_SIZE,
                weight=TYPOGRAPHY.normal_weight,
            ),
        )
        self.options_button.grid(row=0, column=1, sticky="nsew")

        # Overlay the separator at the join so it does not consume a column
        # or cut a one-pixel notch into the top and bottom edges.
        self.divider.place(
            relx=1.0,
            x=-self.SEGMENT_WIDTH,
            rely=0.5,
            anchor="center",
        )
        self.divider.lift()


class WorkflowNavigation(ctk.CTkFrame):
    """Independent workflow buttons without a connecting segmented track."""

    GAP = 6

    def __init__(
        self,
        master,
        *,
        values: Sequence[str],
        command: Callable[[str], None],
        height: int = CONTROLS.height_md,
    ) -> None:
        super().__init__(
            master,
            height=height,
            corner_radius=0,
            border_width=0,
            fg_color="transparent",
        )
        self.grid_propagate(False)
        self._values = tuple(str(value) for value in values)
        self._command = command
        self._value = ""
        self._buttons_dict: dict[str, ctk.CTkButton] = {}

        for index, value in enumerate(self._values):
            button_column = index * 2
            self.grid_columnconfigure(
                button_column,
                weight=1,
                uniform="workflow_step",
            )
            if index < len(self._values) - 1:
                self.grid_columnconfigure(
                    button_column + 1,
                    weight=0,
                    minsize=self.GAP,
                )
            button = ctk.CTkButton(
                self,
                text=value,
                command=lambda selected=value: self._choose(selected),
                height=height,
                corner_radius=SHAPES.corner_radius_md,
                border_width=SHAPES.border_width,
                border_color=COLOR_PAIRS["border_strong"],
                fg_color=COLOR_PAIRS["button"],
                hover_color=COLOR_PAIRS["button_hover"],
                text_color=COLOR_PAIRS["text"],
                font=ctk.CTkFont(
                    family=TYPOGRAPHY.family,
                    size=TYPOGRAPHY.body_size,
                    weight=TYPOGRAPHY.semibold_weight,
                ),
            )
            button.grid(
                row=0,
                column=button_column,
                sticky="nsew",
            )
            self._buttons_dict[value] = button
        self.grid_rowconfigure(0, weight=1)

    def _choose(self, value: str) -> None:
        self.set(value)
        self._command(value)

    def set(self, value: str) -> None:
        """Select one known workflow value without invoking the callback."""

        selected = str(value)
        if selected not in self._buttons_dict or selected == self._value:
            return
        previous = self._buttons_dict.get(self._value)
        if previous is not None:
            self._style_button(previous, selected=False)
        self._value = selected
        self._style_button(self._buttons_dict[selected], selected=True)

    def get(self) -> str:
        return self._value

    @staticmethod
    def _style_button(button: ctk.CTkButton, *, selected: bool) -> None:
        button.configure(
            fg_color=(
                COLOR_PAIRS["primary"]
                if selected
                else COLOR_PAIRS["button"]
            ),
            hover_color=(
                COLOR_PAIRS["primary_hover"]
                if selected
                else COLOR_PAIRS["button_hover"]
            ),
            border_color=(
                COLOR_PAIRS["primary"]
                if selected
                else COLOR_PAIRS["border_strong"]
            ),
            text_color=(
                COLOR_PAIRS["on_primary"]
                if selected
                else COLOR_PAIRS["text"]
            ),
        )


class WorkflowHeader(ctk.CTkFrame):
    """Brand bar, workflow navigation, and appearance control for the app."""

    DEFAULT_STEPS = (
        "1  Load & Crop",
        "2  Segment",
        "3  Flatten",
        "4  MCP/AR",
        "5  Compile",
    )

    def __init__(
        self,
        master,
        *,
        version: str,
        on_step_selected: Callable[[int], None],
        on_settings_selected: Callable[[], None] | None = None,
        on_help_selected: Callable[[], None] | None = None,
        logo_path: str | None = None,
        settings_icon_path: str | None = None,
        help_icon_path: str | None = None,
        step_labels: Sequence[str] | None = None,
    ) -> None:
        super().__init__(
            master,
            height=96,
            corner_radius=0,
            border_width=0,
            fg_color=COLOR_PAIRS["surface"],
        )
        self.grid_columnconfigure(0, weight=1)
        self._on_step_selected = on_step_selected
        self._step_labels = tuple(step_labels or self.DEFAULT_STEPS)

        top = ctk.CTkFrame(self, fg_color="transparent", corner_radius=0)
        top.grid(row=0, column=0, sticky="ew", padx=18, pady=(10, 6))
        top.grid_columnconfigure(1, weight=1)

        logo_holder = ctk.CTkFrame(
            top,
            width=42,
            height=42,
            corner_radius=SHAPES.corner_radius_md,
            fg_color=COLOR_PAIRS["primary_soft"],
        )
        logo_holder.grid(row=0, column=0, rowspan=2, padx=(0, 10))
        logo_holder.grid_propagate(False)
        self.logo_image = None
        if logo_path and Path(logo_path).is_file():
            with Image.open(logo_path) as source:
                logo = source.convert("RGBA").copy()
            self.logo_image = ctk.CTkImage(
                light_image=logo,
                dark_image=logo,
                size=(34, 34),
            )
            ctk.CTkLabel(logo_holder, text="", image=self.logo_image).place(relx=0.5, rely=0.5, anchor="center")
        else:
            ctk.CTkLabel(
                logo_holder,
                text="A",
                text_color=COLOR_PAIRS["primary"],
                font=ctk.CTkFont(
                    family=TYPOGRAPHY.family,
                    size=TYPOGRAPHY.title_size,
                    weight=TYPOGRAPHY.bold_weight,
                ),
            ).place(relx=0.5, rely=0.5, anchor="center")

        title_row = ctk.CTkFrame(top, fg_color="transparent", corner_radius=0)
        title_row.grid(row=0, column=1, sticky="sw")
        ctk.CTkLabel(
            title_row,
            text="AIDaS",
            anchor="w",
            text_color=COLOR_PAIRS["text"],
            font=ctk.CTkFont(
                family=TYPOGRAPHY.family,
                size=TYPOGRAPHY.title_size,
                weight=TYPOGRAPHY.bold_weight,
            ),
        ).pack(side="left")
        ctk.CTkLabel(
            title_row,
            text=f"v{version}",
            width=54,
            height=22,
            corner_radius=11,
            fg_color=COLOR_PAIRS["primary_soft"],
            text_color=COLOR_PAIRS["primary"],
            font=ctk.CTkFont(family=TYPOGRAPHY.family, size=TYPOGRAPHY.caption_size),
        ).pack(side="left", padx=(8, 0))
        subtitle_row = ctk.CTkFrame(
            top,
            fg_color="transparent",
            corner_radius=0,
        )
        subtitle_row.grid(row=1, column=1, sticky="nw")
        ctk.CTkLabel(
            subtitle_row,
            text="OCT image processing workspace",
            anchor="w",
            text_color=COLOR_PAIRS["muted_text"],
            font=ctk.CTkFont(family=TYPOGRAPHY.family, size=TYPOGRAPHY.caption_size),
        ).pack(side="left")
        self.workflow_progress = None
        self.workflow_progress_label = None

        header_actions = ctk.CTkFrame(top, fg_color="transparent", corner_radius=0)
        header_actions.grid(row=0, column=2, rowspan=2, sticky="e")

        def load_header_action_icon(icon_path):
            if not icon_path or not Path(icon_path).is_file():
                return None
            with Image.open(icon_path) as source:
                icon = source.convert("RGBA").copy()
            return ctk.CTkImage(
                light_image=icon,
                dark_image=icon,
                size=(24, 24),
            )

        self.settings_image = load_header_action_icon(settings_icon_path)
        self.help_image = load_header_action_icon(help_icon_path)
        self.settings_button = ctk.CTkButton(
            header_actions,
            text="" if self.settings_image is not None else "\ue713",
            image=self.settings_image,
            command=on_settings_selected,
            state="normal" if on_settings_selected is not None else "disabled",
            width=36,
            height=36,
            corner_radius=SHAPES.corner_radius_md,
            border_width=SHAPES.border_width,
            border_color=COLOR_PAIRS["border_strong"],
            fg_color=COLOR_PAIRS["button"],
            hover_color=COLOR_PAIRS["button_hover"],
            text_color=COLOR_PAIRS["text"],
            font=ctk.CTkFont(family="Segoe Fluent Icons", size=21),
            anchor="center",
        )
        self.settings_button.pack(side="left", padx=(0, 4))
        self.help_button = ctk.CTkButton(
            header_actions,
            text="" if self.help_image is not None else "\ue897",
            image=self.help_image,
            command=on_help_selected,
            state="normal" if on_help_selected is not None else "disabled",
            width=36,
            height=36,
            corner_radius=SHAPES.corner_radius_md,
            border_width=SHAPES.border_width,
            border_color=COLOR_PAIRS["primary"],
            fg_color=COLOR_PAIRS["button"],
            hover_color=COLOR_PAIRS["primary_soft"],
            text_color=COLOR_PAIRS["primary"],
            font=ctk.CTkFont(
                family="Segoe Fluent Icons",
                size=21,
            ),
            anchor="center",
        )
        self.help_button.pack(side="left")
        self.help_tooltip = HoverToolTip(
            self.help_button,
            "Open the AIDaS workflow tutorial",
        )

        self.navigation = WorkflowNavigation(
            self,
            values=list(self._step_labels),
            command=self._navigate,
            height=34,
        )
        self.navigation.grid(row=1, column=0, sticky="ew", padx=18, pady=(0, 10))
        self.select_step(0)

    def _navigate(self, label: str) -> None:
        try:
            index = self._step_labels.index(label)
        except ValueError:
            return
        self._on_step_selected(index)

    def select_step(self, index: int) -> None:
        if 0 <= int(index) < len(self._step_labels):
            selected_index = int(index)
            self.navigation.set(self._step_labels[selected_index])
            self._set_workflow_progress(selected_index)

    def _set_workflow_progress(self, index: int) -> None:
        """(Legacy) Kept for compatibility. Updates are now handled by WorkflowProgressStrip."""
        pass


class AppStatusBar(ctk.CTkFrame):
    """Compact application-wide status bar with an activity indicator."""

    def __init__(self, master, *, text: str) -> None:
        super().__init__(
            master,
            height=30,
            corner_radius=0,
            border_width=0,
            fg_color=COLOR_PAIRS["surface_subtle"],
        )
        self.pack_propagate(False)
        ctk.CTkLabel(
            self,
            text="\u25cf",
            width=18,
            text_color=COLOR_PAIRS["success"],
            font=ctk.CTkFont(family=TYPOGRAPHY.family, size=10),
        ).pack(side="left", padx=(10, 0))
        self.label = ctk.CTkLabel(
            self,
            text=text,
            anchor="w",
            text_color=COLOR_PAIRS["muted_text"],
            font=ctk.CTkFont(family=TYPOGRAPHY.family, size=TYPOGRAPHY.caption_size),
        )
        self.label.pack(side="left", fill="x", expand=True, padx=(2, 12))

    def set_text(self, text: str) -> None:
        self.label.configure(text=text)


def _resolve_color(widget, color):
    if hasattr(widget, "_apply_appearance_mode"):
        return widget._apply_appearance_mode(color)
    if isinstance(color, (list, tuple)):
        mode = ctk.get_appearance_mode()
        return color[0] if mode.lower() == "light" else color[1]
    return color

class ProgressCircle(tk.Canvas):
    NUMBER_FONT_FAMILY = TYPOGRAPHY.family
    NUMBER_FONT_WEIGHT = "bold"

    @staticmethod
    def default_number_font_size(size: int) -> int:
        """Return the numeral size used by workflow progress markers."""

        return max(9, int(size) // 3)

    def __init__(self, master, size=24, text="", bg_color=None, font_size=None):
        self.size = size
        self.bg_color_ref = bg_color
        self.font_size = (
            font_size
            if font_size is not None
            else self.default_number_font_size(size)
        )
        bg = _resolve_color(master, bg_color) if bg_color else "#000000"
        super().__init__(master, width=size, height=size, bg=bg, highlightthickness=0)
        self.oval_id = self.create_oval(1, 1, size-1, size-1, fill="", outline="")
        self.text_id = self.create_text(
            size/2, size/2,
            text=text,
            font=(
                self.NUMBER_FONT_FAMILY,
                self.font_size,
                self.NUMBER_FONT_WEIGHT,
            ),
            fill="black"
        )

    def update_state(self, fill_color, text_color, text):
        f_color = _resolve_color(self.master, fill_color)
        t_color = _resolve_color(self.master, text_color)
        if self.bg_color_ref:
            self.configure(bg=_resolve_color(self.master, self.bg_color_ref))
        self.itemconfig(self.oval_id, fill=f_color, outline=f_color)
        self.itemconfig(self.text_id, fill=t_color, text=text)


class WorkflowProgressStrip(ctk.CTkFrame):
    """Show completed workflow milestones independently of tab selection."""

    DEFAULT_STEPS = (
        "1  Load & Crop",
        "2  Segment",
        "3  Flatten",
        "4  MCP/AR",
        "5  Compile",
    )

    def __init__(
        self,
        master,
        *,
        step_labels: Sequence[str] | None = None,
        height: int = 40,
    ) -> None:
        super().__init__(
            master,
            height=height,
            corner_radius=0,
            border_width=0,
            fg_color=COLOR_PAIRS["primary_soft"],
        )
        self.pack_propagate(False)
        self.grid_propagate(False)
        self._step_labels = tuple(step_labels or self.DEFAULT_STEPS)
        self._completion_animation_job = None
        self._completion_animation_overlay = None

        self.center_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.center_frame.place(relx=0.5, rely=0.5, anchor="center")

        self.circles_frame = ctk.CTkFrame(self.center_frame, fg_color="transparent")
        self.circles_frame.pack(side="left", padx=(0, 15))
        
        self.circles = []
        for i in range(len(self._step_labels)):
            circle = ProgressCircle(
                self.circles_frame,
                size=24,
                text=str(i + 1),
                bg_color=COLOR_PAIRS["primary_soft"],
            )
            circle.pack(side="left", padx=4)
            self.circles.append(circle)

        self.workflow_progress_label = ctk.CTkLabel(
            self.center_frame,
            text="",
            width=340,
            anchor="w",
            text_color=COLOR_PAIRS["text"],
            font=ctk.CTkFont(
                family=TYPOGRAPHY.family,
                size=TYPOGRAPHY.body_size + 2,
                weight=TYPOGRAPHY.bold_weight,
            ),
        )
        self.workflow_progress_label.pack(side="left", padx=(10, 0))

    def set_completed_steps(self, completed_steps: int) -> None:
        """Render progress from the number of successfully completed steps."""

        total = max(1, len(self._step_labels))
        completed = min(total, max(0, int(completed_steps)))
        active_index = completed if completed < total else None
        
        for i, circle in enumerate(self.circles):
            if i < completed:
                circle.update_state(
                    fill_color=COLOR_PAIRS["success"],
                    text_color=COLOR_PAIRS["on_primary"],
                    text="✓"
                )
            elif i == active_index:
                circle.update_state(
                    fill_color=COLOR_PAIRS["primary"],
                    text_color=COLOR_PAIRS["on_primary"],
                    text=str(i + 1)
                )
            else:
                circle.update_state(
                    fill_color=COLOR_PAIRS["surface_subtle"],
                    text_color=COLOR_PAIRS["muted_text"],
                    text=str(i + 1)
                )

        if completed == total:
            label = f"Workflow complete: {total} of {total} steps"
        else:
            current = completed + 1
            step_name = self._step_labels[active_index]
            step_name = step_name.split("  ", 1)[-1] if "  " in step_name else step_name
            label = f"Step {current} of {total}: {step_name}"
        self.workflow_progress_label.configure(text=label)

    def animate_completion(self, completed_index: int) -> None:
        """Show a brief oversized check when a workflow milestone completes."""

        try:
            index = int(completed_index)
        except (TypeError, ValueError):
            return
        if index < 0 or index >= len(self.circles):
            return

        if self._completion_animation_job is not None:
            try:
                self.after_cancel(self._completion_animation_job)
            except tk.TclError:
                pass
            self._completion_animation_job = None
        if self._completion_animation_overlay is not None:
            try:
                self._completion_animation_overlay.destroy()
            except tk.TclError:
                pass

        self.update_idletasks()
        circle = self.circles[index]
        x = circle.winfo_rootx() - self.winfo_rootx() + (circle.winfo_width() // 2)
        y = circle.winfo_rooty() - self.winfo_rooty() + (circle.winfo_height() // 2)
        overlay = ctk.CTkLabel(
            self,
            text="✓",
            fg_color="transparent",
            text_color=COLOR_PAIRS["success"],
            font=ctk.CTkFont(
                family=TYPOGRAPHY.family,
                size=10,
                weight=TYPOGRAPHY.bold_weight,
            ),
        )
        overlay.place(x=x, y=y, anchor="center")
        self._completion_animation_overlay = overlay

        frames = (10, 16, 23, 29, 25, 20, 16)

        def render(frame_index: int = 0) -> None:
            if self._completion_animation_overlay is not overlay:
                return
            if frame_index >= len(frames):
                try:
                    overlay.destroy()
                except tk.TclError:
                    pass
                self._completion_animation_overlay = None
                self._completion_animation_job = None
                return
            overlay.configure(
                font=ctk.CTkFont(
                    family=TYPOGRAPHY.family,
                    size=frames[frame_index],
                    weight=TYPOGRAPHY.bold_weight,
                )
            )
            self._completion_animation_job = self.after(
                65,
                lambda: render(frame_index + 1),
            )

        render()

    def select_step(self, index: int) -> None:
        """Compatibility alias for callers that supply a zero-based step index."""

        self.set_completed_steps(index)


__all__ = [
    "AppButton",
    "AppSplitButton",
    "AppStatusBar",
    "ProgressCircle",
    "WorkflowHeader",
    "WorkflowNavigation",
    "WorkflowProgressStrip",
]
