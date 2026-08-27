"""In-app tutorial for the four-stage AIDaS processing workflow."""

from __future__ import annotations

from dataclasses import dataclass
import time
import tkinter as tk
from typing import Callable

import customtkinter as ctk
from PIL import Image, ImageDraw

from aidas.core.display import work_area_bounds
from aidas.ui.components import AppButton
from aidas.ui.theme import (
    COLOR_PAIRS,
    CONTROLS,
    SHAPES,
    TYPOGRAPHY,
    resolve_color,
)
from aidas.ui.windowing import (
    centered_logical_geometry,
    logical_window_size,
    physical_window_size,
    synchronize_window_chrome,
)
from aidas.utils.ui_utils import HoverToolTip, apply_app_icon_to


StagePointGroups = tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
]


@dataclass(frozen=True)
class TutorialPage:
    """One independently navigable page in the workflow tutorial."""

    key: str
    navigation_label: str
    title: str
    purpose: str
    input_summary: str
    function_summary: str
    output_summary: str
    completion_check: str
    stage_points: StagePointGroups = ((), (), ())
    tips: tuple[str, ...] = ()
    step_index: int | None = None


TUTORIAL_PAGES: tuple[TutorialPage, ...] = (
    TutorialPage(
        key="overview",
        navigation_label="Workflow overview",
        title="From raw OCT scans to ISez measurements",
        purpose=(
            "AIDaS keeps selected subject folder(s) moving through four connected stages. "
            "Each handoff opens the exact folders completed by the previous step, "
            "while every stage can also scan a parent folder when work is resumed later."
        ),
        input_summary="Nested subject folders containing raw .sdb OCT scans",
        function_summary="Crop, annotate, flatten, then quantify retinal profiles",
        output_summary="Analyze images, diagnostic plots, ROI stacks, and Excel measurements",
        completion_check=(
            "The workflow is complete when every selected Step 4 folder contains "
            "ROI_to_move_stck.tif, MAX_Stack.tif, and a measurement workbook. New "
            "runs write rr_MCPAR.xlsx; legacy Results.xlsx workbooks are also recognized."
        ),
        tips=(
            "Use the green progress rows, completed lists, tab counters, and status bars to see what is ready at each stage.",
            "The navigation on the left opens any tutorial page; the button below can take you directly to that app step.",
        ),
    ),
    TutorialPage(
        key="step1",
        navigation_label="Step 1 - Load & Crop",
        title="Step 1 - Load, resize, and crop raw OCT images",
        purpose=(
            "Step 1 decodes raw SDB data, lets you isolate the retinal region, and "
            "normalizes the crop into the Analyze image used by segmentation."
        ),
        input_summary="A parent folder with one or more nested .sdb image folders",
        function_summary="Decode raw pixels, select an ROI, and scale the crop",
        output_summary="One folder-level two-slice 16-bit light.hdr and light.img pair",
        completion_check=(
            "Each folder you want to segment contains both light.hdr and light.img, "
            "and its queue row is shown as completed."
        ),
        stage_points=(
            (
                "Select the parent directory; AIDaS discovers its nested .sdb image folders.",
                "Choose a discovered folder and image. The first image previews automatically; enable Exclude saved cropped folders to show only unfinished work.",
                "Verify Width, Height, Offset, and Little-endian. Height, Offset, and byte order reread the source; Width crops or pads the preview and becomes the decode width for the next image.",
                "Define the ROI by dragging on the image or entering X, Y, Width, and Height; Default Region and Entire Image are available as presets.",
            ),
            (
                "Choose Crop & Scale to create the Target view. AIDaS repeats the crop three times horizontally and once vertically; Undo returns to the editable Source view.",
                "Choose Save to write or replace the current result, mark the queue row complete, and open the next SDB image automatically.",
            ),
            (
                "Each processed folder receives a two-slice, 16-bit light.hdr and light.img Analyze pair.",
                "Completed queue rows identify the folders that are ready for segmentation.",
                "Go to Step 2 opens those completed folders in a preloaded review; confirm the selected rows and choose Continue.",
            ),
        ),
        tips=(
            "If the preview looks shifted or noisy, recheck the byte offset, dimensions, and byte order before cropping.",
            "Save is enabled only after Crop & Scale has produced a target image.",
        ),
        step_index=0,
    ),
    TutorialPage(
        key="step2",
        navigation_label="Step 2 - Annotate",
        title="Step 2 - Segment, review, and annotate retinal boundaries",
        purpose=(
            "Step 2 identifies the six retinal layer boundaries and foveal center "
            "required by flattening. AI supplies a starting trace; every result remains editable."
        ),
        input_summary="Step 1 Light Analyze pairs, loaded from selected folders",
        function_summary="Predict or trace six boundaries, set the fovea, and assign image sides",
        output_summary="Nasal and temporal folders containing 16-bit Light and 8-bit Light_MARKED pairs",
        completion_check=(
            "All six boundaries and the foveal line have been reviewed, the side "
            "orientation is correct, and each nasal/temporal folder contains Light and Light_MARKED Analyze pairs."
        ),
        stage_points=(
            (
                "Use the Step 1 handoff or Select folders to segment, then review the discovered Light images and choose the ready rows.",
                "Folders that already contain Light_MARKED are identified and skipped.",
                "For each image, drag the foveal line or enter Center X.",
                "Choose the exact side layout—Left: Temporal | Right: Nasal or Left: Nasal | Right: Temporal—and Confirm. Skip omits the image; Exit cancels the batch.",
            ),
            (
                "After the final confirmation, AIDaS runs AI segmentation using a compatible DirectML GPU or the displayed shared CPU-core budget.",
                "Review RNFL-Vitreous, GCL-RNFL, INL-IPL, ONL-OPL, ELM, and RPE; Revert and redraw an AI trace or add at least two points for a missing trace, then choose Done. Clear removes boundary traces but keeps the foveal line.",
            ),
            (
                "Save or Save all commits the reviewed results; unsaved AI results and edits exist only in the open review tabs.",
                "AIDaS creates nasal and temporal child folders with normalized two-slice, 2133-pixel-wide 16-bit Light and 8-bit Light_MARKED pairs, mirrored according to the selected orientation.",
                "Go to Step 3 preflights, saves, and hands the completed nasal and temporal folders to the R batch review.",
            ),
        ),
        tips=(
            "The completed and incomplete lists are the authoritative boundary checklist.",
            "Step 2 normalizes saved images to two slices and 2133 pixels wide so Step 3 receives consistent geometry.",
        ),
        step_index=1,
    ),
    TutorialPage(
        key="step3",
        navigation_label="Step 3 - Flatten",
        title="Step 3 - Flatten the retina with the R workflow",
        purpose=(
            "Step 3 uses the fovea and boundary markers to sample perpendicular to "
            "the RPE, align retinal profiles, flatten the volume, and create the layer-analysis outputs."
        ),
        input_summary="Matching 16-bit Light and 8-bit Light_MARKED Analyze pairs",
        function_summary="Run the main and output R scripts, monitor progress, and review diagnostics",
        output_summary="_flat_LIGHT.hdr/_flat_LIGHT.img, R workspaces, plots, profiles, and logs",
        completion_check=(
            "A successful R batch result contains both _thickness_vs_distance_from_fovea_DARK.txt "
            "and _thickness_vs_distance_from_fovea_LIGHT.txt. The previews should show a plausible "
            "foveal vertex and tissue borders; every folder handed to Step 4 must also contain a "
            "complete _flat_LIGHT Analyze pair."
        ),
        stage_points=(
            (
                "If required, run Set up R and packages; the workflow requires R 3.3.1 with the bundled AnalyzeFMRI and RNiftyReg dependencies.",
                "Use the Step 2 handoff or Select folders to flatten.",
                "AIDaS verifies matching readable 16-bit Light and 8-bit Light_MARKED pairs with identical slice, height, and width dimensions; folders with existing RData are locked and skipped.",
                "Select ready folders, set Batch size and the per-script timeout (240 minutes by default), and choose Parallel or Sequential execution. The core limit is shared with active Step 2 CPU work.",
            ),
            (
                "Run selected folders and follow each folder’s progress and log. Stop terminates active R process trees and cancels queued folders; failed or timed-out folders remain identified.",
                "Review DARK_MARKED_find_vertex and _tissueBorders__DARK in the result tabs, or use Load R results to reopen a completed folder.",
            ),
            (
                "Each valid result contains a complete _flat_LIGHT.hdr and _flat_LIGHT.img Analyze pair.",
                "The run also produces R workspaces, diagnostic plots, profiles, logs, _thickness_vs_distance_from_fovea_DARK.txt, and _thickness_vs_distance_from_fovea_LIGHT.txt.",
                "Go to Step 4 becomes available for valid flattened folders and transfers them to ROI analysis.",
            ),
        ),
        tips=(
            "The fovea marker value is 243 and the RPE marker value is 255; the technical diagram shown in the idle Step 3 workspace explains minimum marker coverage and depth geometry.",
            "The selected R-script versions are managed in Settings; the Process panel starts setup, scanning, and result loading.",
        ),
        step_index=2,
    ),
    TutorialPage(
        key="step4",
        navigation_label="Step 4 - Analyze",
        title="Step 4 - Analyze ISez profiles and build measurements",
        purpose=(
            "Step 4 measures the inner-segment ellipsoid zone across 20 peripheral "
            "ROIs and one foveal ROI, then reproduces the lab's MATLAB/ImageJ-style outputs."
        ),
        input_summary="Completed Step 3 _flat_LIGHT Analyze pairs, using slice 0",
        function_summary="Select profile bounds for 21 ROIs and calculate shape measurements",
        output_summary="ROI_to_move_stck.tif, MAX_Stack.tif, and rr_MCPAR.xlsx",
        completion_check=(
            "Every selected folder tab is marked Done and contains "
            "ROI_to_move_stck.tif, MAX_Stack.tif, and a measurement workbook. "
            "New builds write rr_MCPAR.xlsx."
        ),
        stage_points=(
            (
                "Use the Step 3 handoff or Select folders for ROI; the scanner selects complete _flat_LIGHT pairs and locks folders whose final outputs already exist.",
                "Open a folder tab; Step 4 analyzes slice 0 of the flattened stack.",
                "Select one of 21 ROIs from the table or overview grid: 20 peripheral 120-column bands plus the foveal ROI.",
                "Click or enter the profile Start and End positions. AIDaS orders and clamps the values before applying the band bounds.",
            ),
            (
                "AIDaS calculates the rotated and rescaled ISez profile, saves a new two-point selection, and advances to the next ROI; clearing a result makes that ROI selectable again.",
                "Review the overview plots and Major, Minor, Angle, Circ., AR, Round, and Solidity measurements. Use the detail editor to revise a completed ROI; Apply changes commits it, while closing without applying discards it.",
            ),
            (
                "When all 21 ROIs are complete, Build stack becomes available and marks the folder tab Done after a successful build.",
                "The build writes ROI_to_move_stck.tif, MAX_Stack.tif, and rr_MCPAR.xlsx, then advances to the next incomplete folder.",
                "Optional Compile measurements combines LE/RE rrMCP/AR results with Step 3 ELM-RPE and ONL exports in LE_RE_rrMCPAR_ELMRPE_ONL.xlsx.",
            ),
        ),
        tips=(
            "Build stack remains disabled until every one of the 21 ROI rows is complete.",
            "Revising a completed ROI updates its measurements without advancing, so the change can be verified immediately.",
            "The folder scanner also accepts legacy Results.xlsx or Results_org.xlsx as the measurement workbook when deciding that earlier work is complete.",
        ),
        step_index=3,
    ),
)


def tutorial_page_index_for_step(step_index: int) -> int:
    """Return the tutorial page associated with a zero-based workflow step."""

    try:
        step_index = int(step_index)
    except (TypeError, ValueError):
        return 0
    for page_index, page in enumerate(TUTORIAL_PAGES):
        if page.step_index == step_index:
            return page_index
    return 0


@dataclass(frozen=True)
class _PipelineAnimationState:
    active_stage: int
    connector_progress: tuple[float, float]
    stage_progress: float = 0.0


INPUT_HOLD_MS = 5200.0
CONNECTOR_TRAVEL_MS = 900.0
PROCESS_HOLD_MS = 3200.0
OUTPUT_HOLD_MS = 4300.0
PIPELINE_CYCLE_MS = (
    INPUT_HOLD_MS
    + CONNECTOR_TRAVEL_MS
    + PROCESS_HOLD_MS
    + CONNECTOR_TRAVEL_MS
    + OUTPUT_HOLD_MS
)


def _smoothstep(value: float) -> float:
    value = max(0.0, min(1.0, float(value)))
    return value * value * (3.0 - (2.0 * value))


def _pipeline_orientation(logical_width: int) -> str:
    """Choose a readable card layout for the available logical width."""

    return "horizontal" if int(logical_width) >= 650 else "vertical"


def _pipeline_animation_state(elapsed_ms: float) -> _PipelineAnimationState:
    """Return the deterministic stage and connector state for one animation tick."""

    elapsed = max(0.0, float(elapsed_ms)) % PIPELINE_CYCLE_MS

    if elapsed < INPUT_HOLD_MS:
        return _PipelineAnimationState(
            0,
            (0.0, 0.0),
            elapsed / INPUT_HOLD_MS,
        )
    elapsed -= INPUT_HOLD_MS
    if elapsed < CONNECTOR_TRAVEL_MS:
        progress = _smoothstep(elapsed / CONNECTOR_TRAVEL_MS)
        return _PipelineAnimationState(0, (progress, 0.0), 1.0)
    elapsed -= CONNECTOR_TRAVEL_MS
    if elapsed < PROCESS_HOLD_MS:
        return _PipelineAnimationState(
            1,
            (1.0, 0.0),
            elapsed / PROCESS_HOLD_MS,
        )
    elapsed -= PROCESS_HOLD_MS
    if elapsed < CONNECTOR_TRAVEL_MS:
        progress = _smoothstep(elapsed / CONNECTOR_TRAVEL_MS)
        return _PipelineAnimationState(1, (1.0, progress), 1.0)
    elapsed -= CONNECTOR_TRAVEL_MS
    return _PipelineAnimationState(
        2,
        (1.0, 1.0),
        min(1.0, elapsed / OUTPUT_HOLD_MS),
    )


class _WorkflowOverviewMap(ctk.CTkFrame):
    """Static responsive map of the four AIDaS workflow steps."""

    TWO_COLUMN_BREAKPOINT = 480
    STEP_CARDS = (
        (
            "Load & Crop",
            "Decode raw .sdb scans and create the light.hdr/light.img Analyze pair.",
            "primary",
            "primary_soft",
        ),
        (
            "Annotate",
            "Place the fovea and six boundaries, then save nasal and temporal Light_MARKED.",
            "warning",
            "warning_soft",
        ),
        (
            "Flatten",
            "Run the R scripts and diagnostics to generate the _flat_LIGHT Analyze pair.",
            "accent",
            "accent_soft",
        ),
        (
            "Analyze",
            "Measure 21 ROIs and build the TIFF stacks and Excel measurements.",
            "success",
            "success_soft",
        ),
    )

    def __init__(self, master, *, available_width: int = 650) -> None:
        super().__init__(
            master,
            fg_color=COLOR_PAIRS["surface_subtle"],
            corner_radius=SHAPES.corner_radius_lg,
            border_width=SHAPES.border_width,
            border_color=COLOR_PAIRS["border"],
        )
        self._layout: str | None = None
        self._available_width: int | None = None
        self._cards: list[ctk.CTkFrame] = []
        self._summary_labels: list[ctk.CTkLabel] = []
        self.grid_columnconfigure(0, weight=1)
        self.card_host = ctk.CTkFrame(self, fg_color="transparent", corner_radius=0)
        self.card_host.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

        for step_index, (title, summary, strong_color, soft_color) in enumerate(
            self.STEP_CARDS,
            start=1,
        ):
            card = ctk.CTkFrame(
                self.card_host,
                width=1,
                fg_color=COLOR_PAIRS["surface"],
                corner_radius=SHAPES.corner_radius_md,
                border_width=SHAPES.border_width,
                border_color=COLOR_PAIRS[strong_color],
            )
            card.grid_columnconfigure(1, weight=1)
            ctk.CTkLabel(
                card,
                text=f"{step_index:02d}",
                width=38,
                height=38,
                corner_radius=SHAPES.corner_radius_sm,
                fg_color=COLOR_PAIRS[soft_color],
                text_color=COLOR_PAIRS[strong_color],
                font=ctk.CTkFont(
                    family=TYPOGRAPHY.family,
                    size=TYPOGRAPHY.subtitle_size,
                    weight=TYPOGRAPHY.bold_weight,
                ),
            ).grid(row=0, column=0, sticky="nw", padx=(10, 8), pady=(10, 4))
            ctk.CTkLabel(
                card,
                text=f"STEP {step_index}\n{title}",
                anchor="w",
                justify="left",
                text_color=COLOR_PAIRS[strong_color],
                font=ctk.CTkFont(
                    family=TYPOGRAPHY.family,
                    size=TYPOGRAPHY.caption_size,
                    weight=TYPOGRAPHY.bold_weight,
                ),
            ).grid(row=0, column=1, sticky="ew", padx=(0, 10), pady=(9, 4))
            summary_label = ctk.CTkLabel(
                card,
                text=summary,
                anchor="nw",
                justify="left",
                wraplength=150,
                text_color=COLOR_PAIRS["text"],
                font=ctk.CTkFont(
                    family=TYPOGRAPHY.family,
                    size=TYPOGRAPHY.body_size,
                ),
            )
            summary_label.grid(
                row=1,
                column=0,
                columnspan=2,
                sticky="nsew",
                padx=10,
                pady=(2, 10),
            )
            self._cards.append(card)
            self._summary_labels.append(summary_label)

        self.set_available_width(available_width)

    def set_available_width(self, logical_width: int) -> None:
        """Lay out four step cards without introducing horizontal overflow."""

        logical_width = max(220, int(logical_width))
        if logical_width == self._available_width:
            return
        self._available_width = logical_width
        layout = "grid" if logical_width >= self.TWO_COLUMN_BREAKPOINT else "vertical"
        if layout != self._layout:
            for card in self._cards:
                card.grid_forget()
            for column in range(2):
                self.card_host.grid_columnconfigure(
                    column,
                    weight=0,
                    minsize=0,
                    uniform="",
                )
            if layout == "grid":
                for index, card in enumerate(self._cards):
                    row, column = divmod(index, 2)
                    card.grid(
                        row=row,
                        column=column,
                        sticky="nsew",
                        padx=3,
                        pady=3,
                    )
                self.card_host.grid_columnconfigure(
                    0,
                    weight=1,
                    uniform="overview_step",
                )
                self.card_host.grid_columnconfigure(
                    1,
                    weight=1,
                    uniform="overview_step",
                )
            else:
                for index, card in enumerate(self._cards):
                    card.grid(
                        row=index,
                        column=0,
                        sticky="nsew",
                        padx=3,
                        pady=3,
                    )
                self.card_host.grid_columnconfigure(0, weight=1)
            self._layout = layout

        columns = 2 if layout == "grid" else 1
        card_width = max(110, (logical_width - 34) // columns)
        wraplength = max(90, card_width - 20)
        for label in self._summary_labels:
            label.configure(wraplength=wraplength)

    @staticmethod
    def stop() -> None:
        """Match the animated visualizer lifecycle contract; no timer exists."""


class _WorkflowPipeline(ctk.CTkFrame):
    """Responsive Input/Process/Output story with stage-specific tutorial points."""

    ANIMATION_INTERVAL_MS = 50
    HIDDEN_RETRY_MS = 250
    RESIZE_DEBOUNCE_MS = 60
    MIN_CONNECTOR_SIZE = 48
    MAX_CONNECTOR_SIZE = 80
    HORIZONTAL_DIAGRAM_HEIGHT = 106
    HORIZONTAL_DETAIL_ROW_HEIGHT = 294
    STAGE_LABELS = ("INPUT", "PROCESS", "OUTPUT")
    STAGE_ROLES = ("SOURCE", "TRANSFORM", "RESULT")
    STAGE_STRONG_COLORS = ("primary", "warning", "success")
    STAGE_SOFT_COLORS = ("primary_soft", "warning_soft", "success_soft")

    def __init__(
        self,
        master,
        summaries: tuple[str, str, str],
        stage_points: StagePointGroups,
        completion_check: str,
        tips: tuple[str, ...] = (),
        *,
        available_width: int = 650,
    ) -> None:
        if len(summaries) != 3:
            raise ValueError("The workflow pipeline requires exactly three summaries")
        if len(stage_points) != 3 or any(not group for group in stage_points):
            raise ValueError(
                "The workflow pipeline requires non-empty Input, Process, and Output points"
            )
        if not completion_check.strip():
            raise ValueError("The workflow pipeline requires a completion check")
        super().__init__(
            master,
            fg_color=COLOR_PAIRS["surface_subtle"],
            corner_radius=SHAPES.corner_radius_lg,
            border_width=SHAPES.border_width,
            border_color=COLOR_PAIRS["border"],
        )
        self._summaries = summaries
        self._stage_points: StagePointGroups = (
            tuple(stage_points[0]),
            tuple(stage_points[1]),
            tuple(stage_points[2]),
        )
        self._completion_check = completion_check
        self._tips = tuple(tips)
        self._orientation: str | None = None
        self._summary_wraplength: int | None = None
        self._connector_size: int | None = None
        initial_width = max(220, int(available_width))
        self._pending_logical_width = initial_width
        self._viewport_logical_width = initial_width
        self._applied_layout_width: int | None = None
        self._stage_points_orientation: str | None = None
        self._stage_point_wraplength: int | None = None
        self._context_wraplength: int | None = None
        self._animation_after_id = None
        self._resize_after_id = None
        self._destroying = False
        self._stopped = False
        self._paused = False
        self._hidden_since: float | None = None
        self._animation_started_at = time.monotonic()
        self._paused_elapsed_ms = 0.0
        self._animation_state = _pipeline_animation_state(0.0)
        self._rendered_active_stage: int | None = None
        self._rendered_paused: bool | None = None
        self._rendered_reveal: tuple[int, int] | None = None
        self._theme_binding_widget = None
        self._theme_binding_id = None
        self._cards: list[ctk.CTkFrame] = []
        self._title_labels: list[ctk.CTkLabel] = []
        self._summary_labels: list[ctk.CTkLabel] = []
        self._icon_tiles: list[ctk.CTkFrame] = []
        self._icon_canvases: list[tk.Canvas] = []
        self._stage_group_frames: list[ctk.CTkFrame] = []
        self._stage_point_hosts: list[ctk.CTkFrame] = []
        self._stage_point_widgets: list[
            list[tuple[ctk.CTkFrame, ctk.CTkLabel, ctk.CTkLabel]]
        ] = []
        self._context_labels: list[ctk.CTkLabel] = []
        self._connector_canvases: list[tk.Canvas] = []
        self._connector_items: list[dict[str, object]] = [{}, {}]
        self._connector_geometry: list[tuple[float, float, float, float] | None] = [
            None,
            None,
        ]

        self._build_ui()
        self._apply_responsive_layout(initial_width)
        self._redraw_stage_icons()
        self._apply_animation_state(self._animation_state)
        self.bind("<Destroy>", self._on_destroy_event, add="+")
        try:
            root = self._root()
            self._theme_binding_widget = root
            self._theme_binding_id = root.bind(
                "<<AIDaSThemeChanged>>",
                self._on_theme_changed,
                add="+",
            )
        except (AttributeError, tk.TclError):
            self._theme_binding_widget = None
            self._theme_binding_id = None
        self.start()

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        header = ctk.CTkFrame(self, fg_color="transparent", corner_radius=0)
        header.grid(row=0, column=0, sticky="ew", padx=10, pady=(4, 0))
        header.grid_columnconfigure(0, weight=1)
        self.motion_status = ctk.CTkLabel(
            header,
            text="●  INPUT ACTIVE",
            anchor="w",
            text_color=COLOR_PAIRS["primary"],
            font=ctk.CTkFont(
                family=TYPOGRAPHY.family,
                size=TYPOGRAPHY.caption_size,
                weight=TYPOGRAPHY.semibold_weight,
            ),
        )
        self.motion_status.grid(row=0, column=0, sticky="w")
        self._refresh_animation_icons()
        self.animation_button = AppButton(
            header,
            text="",
            image=self._pause_animation_icon,
            variant="ghost",
            width=CONTROLS.height_md,
            height=CONTROLS.height_md,
            corner_radius=SHAPES.corner_radius_sm,
            border_spacing=0,
            command=self._toggle_animation,
        )
        self.animation_button.grid(row=0, column=1, sticky="e")
        self.animation_button.image = self._pause_animation_icon
        self._animation_tooltip = HoverToolTip(
            self.animation_button,
            "Pause animation",
        )
        self.animation_button._canvas.configure(takefocus=1)
        self.animation_button.bind(
            "<Return>",
            self._toggle_animation_from_keyboard,
            add="+",
        )
        self.animation_button.bind(
            "<space>",
            self._toggle_animation_from_keyboard,
            add="+",
        )
        self.animation_button.bind("<FocusIn>", self._show_animation_focus, add="+")
        self.animation_button.bind("<FocusOut>", self._hide_animation_focus, add="+")

        self.diagram = ctk.CTkFrame(self, fg_color="transparent", corner_radius=0)
        self.diagram.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 4))

        for index, summary in enumerate(self._summaries):
            card = ctk.CTkFrame(
                self.diagram,
                width=1,
                fg_color=COLOR_PAIRS["surface"],
                corner_radius=SHAPES.corner_radius_md,
                border_width=2,
                border_color=COLOR_PAIRS["border"],
            )
            card.grid_columnconfigure(1, weight=1)
            card.grid_rowconfigure(1, weight=1)
            icon_tile = ctk.CTkFrame(
                card,
                width=38,
                height=38,
                fg_color=COLOR_PAIRS[self.STAGE_SOFT_COLORS[index]],
                corner_radius=SHAPES.corner_radius_sm,
                border_width=SHAPES.border_width,
                border_color=COLOR_PAIRS[self.STAGE_STRONG_COLORS[index]],
            )
            icon_tile.grid_propagate(False)
            icon_tile.pack_propagate(False)
            icon_canvas = tk.Canvas(
                icon_tile,
                width=1,
                height=1,
                highlightthickness=0,
                bd=0,
                takefocus=0,
            )
            icon_canvas.pack(fill="both", expand=True, padx=4, pady=4)
            icon_canvas.bind(
                "<Configure>",
                lambda event, stage_index=index: self._redraw_stage_icon(
                    stage_index,
                    event,
                ),
                add="+",
            )
            icon_tile.grid(
                row=0,
                column=0,
                sticky="n",
                padx=(8, 6),
                pady=(8, 4),
            )
            title_label = ctk.CTkLabel(
                card,
                text=f"0{index + 1}  {self.STAGE_LABELS[index]}",
                anchor="w",
                text_color=COLOR_PAIRS[self.STAGE_STRONG_COLORS[index]],
                font=ctk.CTkFont(
                    family=TYPOGRAPHY.family,
                    size=TYPOGRAPHY.caption_size,
                    weight=TYPOGRAPHY.bold_weight,
                ),
            )
            title_label.grid(
                row=0,
                column=1,
                sticky="ew",
                padx=(0, 8),
                pady=(8, 4),
            )
            summary_label = ctk.CTkLabel(
                card,
                text=summary,
                anchor="nw",
                justify="left",
                wraplength=150,
                text_color=COLOR_PAIRS["text"],
                font=ctk.CTkFont(
                    family=TYPOGRAPHY.family,
                    size=TYPOGRAPHY.body_size,
                ),
            )
            summary_label.grid(
                row=1,
                column=0,
                columnspan=2,
                sticky="nsew",
                padx=8,
                pady=(0, 7),
            )
            self._cards.append(card)
            self._title_labels.append(title_label)
            self._summary_labels.append(summary_label)
            self._icon_tiles.append(icon_tile)
            self._icon_canvases.append(icon_canvas)

        for index in range(2):
            canvas = tk.Canvas(
                self.diagram,
                width=self.MIN_CONNECTOR_SIZE,
                height=120,
                highlightthickness=0,
                bd=0,
                takefocus=0,
            )
            canvas.bind(
                "<Configure>",
                lambda event, connector_index=index: self._redraw_connector(
                    connector_index,
                    event,
                ),
                add="+",
            )
            self._connector_canvases.append(canvas)

        self.stage_detail_shell = ctk.CTkFrame(
            self,
            fg_color=COLOR_PAIRS["surface_elevated"],
            corner_radius=SHAPES.corner_radius_md,
            border_width=2,
            border_color=COLOR_PAIRS["primary"],
        )
        self.stage_detail_shell.grid(
            row=2,
            column=0,
            sticky="ew",
            padx=10,
            pady=(2, 8),
        )
        self.stage_detail_shell.grid_columnconfigure(0, weight=1)
        self.stage_detail_shell.grid_rowconfigure(0, weight=1)

        for stage_index, points in enumerate(self._stage_points):
            strong_color = COLOR_PAIRS[self.STAGE_STRONG_COLORS[stage_index]]
            soft_color = COLOR_PAIRS[self.STAGE_SOFT_COLORS[stage_index]]
            group_frame = ctk.CTkFrame(
                self.stage_detail_shell,
                fg_color=COLOR_PAIRS["surface_elevated"],
                corner_radius=0,
            )
            group_frame.grid(row=0, column=0, sticky="nsew")
            group_frame.grid_columnconfigure(0, weight=1)
            group_frame.grid_rowconfigure(1, weight=1)
            ctk.CTkLabel(
                group_frame,
                text=f"{self.STAGE_LABELS[stage_index]} · {len(points)} POINTS",
                anchor="w",
                text_color=strong_color,
                font=ctk.CTkFont(
                    family=TYPOGRAPHY.family,
                    size=TYPOGRAPHY.caption_size,
                    weight=TYPOGRAPHY.bold_weight,
                ),
            ).grid(row=0, column=0, sticky="ew", padx=7, pady=(5, 2))
            point_host = ctk.CTkFrame(
                group_frame,
                fg_color="transparent",
                corner_radius=0,
            )
            point_host.grid(row=1, column=0, sticky="nsew", padx=4, pady=(0, 4))
            widgets: list[tuple[ctk.CTkFrame, ctk.CTkLabel, ctk.CTkLabel]] = []
            for point_number, point in enumerate(points, start=1):
                point_card = ctk.CTkFrame(
                    point_host,
                    fg_color=COLOR_PAIRS["surface"],
                    corner_radius=SHAPES.corner_radius_sm,
                    border_width=2,
                    border_color=COLOR_PAIRS["border"],
                )
                point_card.grid_columnconfigure(1, weight=1)
                point_card.grid_rowconfigure(0, weight=1)
                badge = ctk.CTkLabel(
                    point_card,
                    text=str(point_number),
                    width=22,
                    height=22,
                    corner_radius=11,
                    fg_color=soft_color,
                    text_color=strong_color,
                    font=ctk.CTkFont(
                        family=TYPOGRAPHY.family,
                        size=TYPOGRAPHY.caption_size,
                        weight=TYPOGRAPHY.bold_weight,
                    ),
                )
                badge.grid(row=0, column=0, sticky="nw", padx=(5, 4), pady=4)
                label = ctk.CTkLabel(
                    point_card,
                    text=point,
                    anchor="nw",
                    justify="left",
                    wraplength=250,
                    text_color=COLOR_PAIRS["muted_text"],
                    font=ctk.CTkFont(
                        family=TYPOGRAPHY.family,
                        size=TYPOGRAPHY.caption_size,
                    ),
                )
                label.grid(
                    row=0,
                    column=1,
                    sticky="nsew",
                    padx=(0, 5),
                    pady=4,
                )
                widgets.append((point_card, badge, label))
            self._stage_group_frames.append(group_frame)
            self._stage_point_hosts.append(point_host)
            self._stage_point_widgets.append(widgets)

        self.context_card = ctk.CTkFrame(
            self.stage_detail_shell,
            fg_color=COLOR_PAIRS["surface"],
            corner_radius=SHAPES.corner_radius_sm,
            border_width=SHAPES.border_width,
            border_color=COLOR_PAIRS["border"],
        )
        self.context_card.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            self.context_card,
            text="READY WHEN",
            anchor="w",
            text_color=COLOR_PAIRS["success"],
            font=ctk.CTkFont(
                family=TYPOGRAPHY.family,
                size=TYPOGRAPHY.caption_size,
                weight=TYPOGRAPHY.bold_weight,
            ),
        ).grid(row=0, column=0, sticky="ew", padx=8, pady=(6, 1))
        self._completion_label = ctk.CTkLabel(
            self.context_card,
            text=self._completion_check,
            anchor="nw",
            justify="left",
            wraplength=220,
            text_color=COLOR_PAIRS["text"],
            font=ctk.CTkFont(
                family=TYPOGRAPHY.family,
                size=TYPOGRAPHY.caption_size,
            ),
        )
        self._completion_label.grid(
            row=1,
            column=0,
            sticky="ew",
            padx=8,
            pady=(0, 5),
        )
        self._context_labels.append(self._completion_label)

        self._tips_label: ctk.CTkLabel | None = None
        if self._tips:
            ctk.CTkLabel(
                self.context_card,
                text="KEY NOTES",
                anchor="w",
                text_color=COLOR_PAIRS["primary"],
                font=ctk.CTkFont(
                    family=TYPOGRAPHY.family,
                    size=TYPOGRAPHY.caption_size,
                    weight=TYPOGRAPHY.bold_weight,
                ),
            ).grid(row=2, column=0, sticky="ew", padx=8, pady=(1, 1))
            self._tips_label = ctk.CTkLabel(
                self.context_card,
                text="\n".join(f"\u2022  {tip}" for tip in self._tips),
                anchor="nw",
                justify="left",
                wraplength=220,
                text_color=COLOR_PAIRS["muted_text"],
                font=ctk.CTkFont(
                    family=TYPOGRAPHY.family,
                    size=TYPOGRAPHY.caption_size,
                ),
            )
            self._tips_label.grid(
                row=3,
                column=0,
                sticky="ew",
                padx=8,
                pady=(0, 6),
            )
            self._context_labels.append(self._tips_label)

        self._stage_group_frames[0].tkraise()

    @staticmethod
    def _render_animation_icon(kind: str, color: str) -> Image.Image:
        """Render one anti-aliased play/pause glyph at four times display size."""

        source_size = CONTROLS.icon_size * 4
        icon = Image.new("RGBA", (source_size, source_size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(icon)
        if kind == "pause":
            draw.rounded_rectangle((21, 15, 31, 57), radius=4, fill=color)
            draw.rounded_rectangle((41, 15, 51, 57), radius=4, fill=color)
        elif kind == "play":
            draw.polygon(((24, 13), (24, 59), (58, 36)), fill=color)
        else:
            raise ValueError(f"Unknown animation icon: {kind}")
        return icon

    @classmethod
    def _build_animation_icon(cls, kind: str) -> ctk.CTkImage:
        color_pair = COLOR_PAIRS["primary"]
        if isinstance(color_pair, str):
            light_color = dark_color = color_pair
        else:
            light_color, dark_color = color_pair
        return ctk.CTkImage(
            light_image=cls._render_animation_icon(kind, light_color),
            dark_image=cls._render_animation_icon(kind, dark_color),
            size=(CONTROLS.icon_size, CONTROLS.icon_size),
        )

    def _refresh_animation_icons(self) -> None:
        self._pause_animation_icon = self._build_animation_icon("pause")
        self._play_animation_icon = self._build_animation_icon("play")
        if hasattr(self, "animation_button"):
            self._sync_animation_control()

    def _sync_animation_control(self) -> None:
        icon = (
            self._play_animation_icon
            if self._paused
            else self._pause_animation_icon
        )
        tooltip = "Play animation" if self._paused else "Pause animation"
        self.animation_button.image = icon
        self.animation_button.configure(image=icon)
        if hasattr(self, "_animation_tooltip"):
            self._animation_tooltip.text = tooltip

    def replace_content(
        self,
        summaries: tuple[str, str, str],
        stage_points: StagePointGroups,
        completion_check: str,
        tips: tuple[str, ...],
    ) -> bool:
        """Reuse the fixed 4/2/3 pipeline when another tutorial step opens."""

        if (
            len(summaries) != len(self._summary_labels)
            or tuple(map(len, stage_points))
            != tuple(map(len, self._stage_point_widgets))
            or not completion_check.strip()
            or bool(tips) != (self._tips_label is not None)
        ):
            return False

        self._summaries = tuple(summaries)
        self._stage_points = tuple(tuple(group) for group in stage_points)
        self._completion_check = completion_check
        self._tips = tuple(tips)
        for label, summary in zip(self._summary_labels, self._summaries):
            label.configure(text=summary)
        for widgets, points in zip(self._stage_point_widgets, self._stage_points):
            for (_point_card, _badge, label), point in zip(widgets, points):
                label.configure(text=point)
        self._completion_label.configure(text=self._completion_check)
        if self._tips_label is not None:
            self._tips_label.configure(
                text="\n".join(f"\u2022  {tip}" for tip in self._tips)
            )

        self._cancel_after_job("_animation_after_id")
        self._paused = False
        self._hidden_since = None
        self._animation_started_at = time.monotonic()
        self._paused_elapsed_ms = 0.0
        self._animation_state = _pipeline_animation_state(0.0)
        self._rendered_active_stage = None
        self._rendered_paused = None
        self._rendered_reveal = None
        self._sync_animation_control()
        self._apply_animation_state(self._animation_state)
        self.start()
        return True

    def _layout_stage_points(self, orientation: str, logical_width: int) -> None:
        horizontal = orientation == "horizontal"
        if horizontal:
            context_width = max(230, min(285, int(logical_width * 0.35)))
            points_width = max(340, int(logical_width) - context_width - 38)
            columns = 2
            self.stage_detail_shell.grid_columnconfigure(0, weight=3, minsize=0)
            self.stage_detail_shell.grid_columnconfigure(
                1,
                weight=2,
                minsize=context_width,
            )
            self.stage_detail_shell.grid_rowconfigure(
                0,
                weight=1,
                minsize=self.HORIZONTAL_DETAIL_ROW_HEIGHT,
            )
            self.stage_detail_shell.grid_rowconfigure(1, weight=0, minsize=0)
            context_wrap = max(175, context_width - 22)
        else:
            points_width = max(180, int(logical_width) - 28)
            columns = 1
            self.stage_detail_shell.grid_columnconfigure(0, weight=1, minsize=0)
            self.stage_detail_shell.grid_columnconfigure(1, weight=0, minsize=0)
            self.stage_detail_shell.grid_rowconfigure(0, weight=1, minsize=0)
            self.stage_detail_shell.grid_rowconfigure(1, weight=0, minsize=0)
            context_wrap = max(180, int(logical_width) - 48)

        if context_wrap != self._context_wraplength:
            self._context_wraplength = context_wrap
            for label in self._context_labels:
                label.configure(wraplength=context_wrap)

        card_width = max(150, points_width // columns)
        wraplength = max(120, card_width - 48)
        if wraplength != self._stage_point_wraplength:
            self._stage_point_wraplength = wraplength
            for widgets in self._stage_point_widgets:
                for _point_card, _badge, label in widgets:
                    label.configure(wraplength=wraplength)

        if orientation == self._stage_points_orientation:
            return
        self._stage_points_orientation = orientation
        if horizontal:
            self.context_card.grid(
                row=0,
                column=1,
                sticky="nsew",
                padx=(3, 5),
                pady=5,
            )
        else:
            self.context_card.grid(
                row=1,
                column=0,
                sticky="ew",
                padx=5,
                pady=(1, 5),
            )

        for point_host, widgets in zip(
            self._stage_point_hosts,
            self._stage_point_widgets,
        ):
            for column in range(2):
                point_host.grid_columnconfigure(
                    column,
                    weight=0,
                    minsize=0,
                    uniform="",
                )
            for row in range(4):
                point_host.grid_rowconfigure(
                    row,
                    weight=0,
                    minsize=0,
                    uniform="",
                )
            for index, (point_card, _badge, label) in enumerate(widgets):
                point_card.grid_forget()
                row, column = divmod(index, columns)
                point_card.grid(
                    row=row,
                    column=column,
                    sticky="nsew",
                    padx=2,
                    pady=2,
                )
            for column in range(columns):
                point_host.grid_columnconfigure(
                    column,
                    weight=1,
                    uniform="stage_point",
                )
            row_count = (len(widgets) + columns - 1) // columns
            for row in range(row_count):
                point_host.grid_rowconfigure(
                    row,
                    weight=1,
                    uniform="stage_point_row",
                )

    def _apply_stage_reveal(self, stage_index: int, progress: float) -> None:
        progress = max(0.0, min(1.0, float(progress)))
        widgets = self._stage_point_widgets[stage_index]
        revealed_count = min(len(widgets), int(progress * len(widgets)) + 1)
        render_key = (stage_index, revealed_count)
        if render_key == self._rendered_reveal:
            return
        strong_color = COLOR_PAIRS[self.STAGE_STRONG_COLORS[stage_index]]
        soft_color = COLOR_PAIRS[self.STAGE_SOFT_COLORS[stage_index]]
        self._stage_group_frames[stage_index].tkraise()
        for index, (point_card, badge, label) in enumerate(widgets):
            revealed = index < revealed_count
            current = index == revealed_count - 1
            point_card.configure(
                fg_color=soft_color if revealed else COLOR_PAIRS["surface"],
                border_color=(
                    strong_color if current else COLOR_PAIRS["border"]
                ),
            )
            badge.configure(
                fg_color=strong_color if revealed else COLOR_PAIRS["surface_subtle"],
                text_color=(
                    COLOR_PAIRS["on_primary"]
                    if revealed
                    else COLOR_PAIRS["muted_text"]
                ),
            )
            label.configure(
                text_color=(
                    COLOR_PAIRS["text"]
                    if revealed
                    else COLOR_PAIRS["muted_text"]
                ),
            )
        self._rendered_reveal = render_key

    def set_available_width(self, logical_width: int) -> None:
        """Receive the scroll viewport width, which is authoritative for overflow."""

        available_width = max(220, int(logical_width))
        if available_width == self._viewport_logical_width:
            return
        self._viewport_logical_width = available_width
        self._queue_responsive_layout(available_width)

    def _queue_responsive_layout(self, logical_width: int) -> None:
        if self._destroying or self._stopped:
            return
        logical_width = max(220, int(logical_width))
        if logical_width == self._applied_layout_width:
            self._cancel_after_job("_resize_after_id")
            self._pending_logical_width = logical_width
            return
        if (
            logical_width == self._pending_logical_width
            and self._resize_after_id is not None
        ):
            return
        self._pending_logical_width = logical_width
        self._cancel_after_job("_resize_after_id")
        try:
            self._resize_after_id = self.after(
                self.RESIZE_DEBOUNCE_MS,
                self._apply_pending_layout,
            )
        except tk.TclError:
            self._resize_after_id = None

    def _apply_pending_layout(self) -> None:
        self._resize_after_id = None
        if self._destroying or self._stopped:
            return
        self._apply_responsive_layout(self._pending_logical_width)

    def _apply_responsive_layout(self, logical_width: int) -> None:
        logical_width = max(220, int(logical_width))
        if logical_width == self._applied_layout_width:
            return
        orientation = _pipeline_orientation(logical_width)
        orientation_changed = orientation != self._orientation
        if orientation_changed:
            for widget in (*self._cards, *self._connector_canvases):
                widget.grid_forget()
            for column in range(5):
                self.diagram.grid_columnconfigure(
                    column,
                    weight=0,
                    minsize=0,
                    uniform="",
                )
            for row in range(5):
                self.diagram.grid_rowconfigure(row, weight=0, minsize=0)
            self._configure_card_content_layout(orientation)

        if orientation == "horizontal":
            connector_size = max(
                self.MIN_CONNECTOR_SIZE,
                min(self.MAX_CONNECTOR_SIZE, (int(logical_width) - 540) // 2),
            )
            if orientation_changed:
                for index, card in enumerate(self._cards):
                    column = index * 2
                    card.grid(row=0, column=column, sticky="nsew", pady=2)
                    self.diagram.grid_columnconfigure(
                        column,
                        weight=1,
                        uniform="pipeline_stage",
                    )
                for index, canvas in enumerate(self._connector_canvases):
                    column = (index * 2) + 1
                    canvas.grid(row=0, column=column, sticky="nsew", padx=3, pady=2)
                self.diagram.grid_rowconfigure(
                    0,
                    weight=1,
                    minsize=self.HORIZONTAL_DIAGRAM_HEIGHT,
                )
            if orientation_changed or connector_size != self._connector_size:
                for index, canvas in enumerate(self._connector_canvases):
                    column = (index * 2) + 1
                    canvas.configure(height=100, width=connector_size)
                    self.diagram.grid_columnconfigure(
                        column,
                        weight=0,
                        minsize=connector_size,
                    )
                self._connector_size = connector_size
            card_width = max(
                120,
                (int(logical_width) - (connector_size * 2) - 38) // 3,
            )
            wraplength = max(110, card_width - 24)
        else:
            if orientation_changed:
                self.diagram.grid_columnconfigure(0, weight=1)
                for index, card in enumerate(self._cards):
                    card.grid(
                        row=index * 2,
                        column=0,
                        sticky="ew",
                        padx=2,
                        pady=2,
                    )
                for index, canvas in enumerate(self._connector_canvases):
                    canvas.configure(height=20, width=1)
                    canvas.grid(
                        row=(index * 2) + 1,
                        column=0,
                        sticky="ew",
                        padx=2,
                    )
                self._connector_size = None
            wraplength = max(180, int(logical_width) - 100)

        self._orientation = orientation
        if wraplength != self._summary_wraplength:
            self._summary_wraplength = wraplength
            for label in self._summary_labels:
                label.configure(wraplength=wraplength)
        self._layout_stage_points(orientation, logical_width)
        self._applied_layout_width = logical_width

    def _configure_card_content_layout(self, orientation: str) -> None:
        """Use compact side-by-side copy only in the narrow vertical timeline."""

        for icon_tile, title_label, summary_label in zip(
            self._icon_tiles,
            self._title_labels,
            self._summary_labels,
        ):
            icon_tile.grid_forget()
            title_label.grid_forget()
            summary_label.grid_forget()
            if orientation == "horizontal":
                icon_tile.configure(width=38, height=38)
                title_label.configure(height=22)
                icon_tile.grid(
                    row=0,
                    column=0,
                    sticky="n",
                    padx=(8, 6),
                    pady=(8, 4),
                )
                title_label.grid(
                    row=0,
                    column=1,
                    sticky="ew",
                    padx=(0, 8),
                    pady=(8, 4),
                )
                summary_label.grid(
                    row=1,
                    column=0,
                    columnspan=2,
                    sticky="nsew",
                    padx=8,
                    pady=(0, 7),
                )
                continue
            icon_tile.configure(width=36, height=36)
            title_label.configure(height=18)
            icon_tile.grid(
                row=0,
                column=0,
                rowspan=2,
                sticky="n",
                padx=(7, 7),
                pady=5,
            )
            title_label.grid(
                row=0,
                column=1,
                sticky="ew",
                padx=(0, 7),
                pady=(5, 0),
            )
            summary_label.grid(
                row=1,
                column=1,
                sticky="nsew",
                padx=(0, 7),
                pady=(0, 5),
            )

    def _redraw_stage_icons(self) -> None:
        for index in range(len(self._icon_canvases)):
            self._redraw_stage_icon(index)

    def _redraw_stage_icon(self, index: int, event=None) -> None:
        if self._destroying or index >= len(self._icon_canvases):
            return
        canvas = self._icon_canvases[index]
        strong_pair = COLOR_PAIRS[self.STAGE_STRONG_COLORS[index]]
        soft_pair = COLOR_PAIRS[self.STAGE_SOFT_COLORS[index]]
        strong = resolve_color(strong_pair)
        soft = resolve_color(soft_pair)
        try:
            width = max(1, int(getattr(event, "width", canvas.winfo_width())))
            height = max(1, int(getattr(event, "height", canvas.winfo_height())))
            self._icon_tiles[index].configure(
                fg_color=soft_pair,
                border_color=strong_pair,
            )
            canvas.configure(background=soft)
            canvas.delete("all")
            scale = max(0.1, min(width, height) / 36.0)
            offset_x = (width - (36.0 * scale)) / 2.0
            offset_y = (height - (36.0 * scale)) / 2.0
            def x(value: float) -> float:
                return offset_x + (float(value) * scale)

            def y(value: float) -> float:
                return offset_y + (float(value) * scale)

            stroke = max(1, round(2.0 * scale))
            if index == 0:
                canvas.create_polygon(
                    x(4),
                    y(11),
                    x(14),
                    y(11),
                    x(18),
                    y(15),
                    x(32),
                    y(15),
                    x(32),
                    y(29),
                    x(4),
                    y(29),
                    fill="",
                    outline=strong,
                    width=stroke,
                    joinstyle=tk.ROUND,
                )
                canvas.create_line(
                    x(4),
                    y(15),
                    x(32),
                    y(15),
                    fill=strong,
                    width=stroke,
                )
            elif index == 1:
                for line_y, knob_x in ((9, 14), (18, 25), (27, 11)):
                    canvas.create_line(
                        x(5),
                        y(line_y),
                        x(31),
                        y(line_y),
                        fill=strong,
                        width=stroke,
                        capstyle=tk.ROUND,
                    )
                    canvas.create_oval(
                        x(knob_x - 3),
                        y(line_y - 3),
                        x(knob_x + 3),
                        y(line_y + 3),
                        fill=soft,
                        outline=strong,
                        width=stroke,
                    )
            else:
                canvas.create_polygon(
                    x(7),
                    y(4),
                    x(24),
                    y(4),
                    x(31),
                    y(11),
                    x(31),
                    y(32),
                    x(7),
                    y(32),
                    fill="",
                    outline=strong,
                    width=stroke,
                    joinstyle=tk.ROUND,
                )
                canvas.create_line(
                    x(24),
                    y(4),
                    x(24),
                    y(11),
                    x(31),
                    y(11),
                    fill=strong,
                    width=stroke,
                )
                canvas.create_line(
                    x(12),
                    y(21),
                    x(17),
                    y(26),
                    x(26),
                    y(16),
                    fill=strong,
                    width=stroke,
                    capstyle=tk.ROUND,
                    joinstyle=tk.ROUND,
                )
        except tk.TclError:
            return

    def _redraw_connector(self, index: int, event=None) -> None:
        if self._destroying or index >= len(self._connector_canvases):
            return
        canvas = self._connector_canvases[index]
        try:
            width = max(2, int(getattr(event, "width", canvas.winfo_width())))
            height = max(2, int(getattr(event, "height", canvas.winfo_height())))
            background = resolve_color(COLOR_PAIRS["surface_subtle"])
            canvas.configure(background=background)
            canvas.delete("all")
            if self._orientation == "horizontal":
                geometry = (5.0, height / 2.0, width - 5.0, height / 2.0)
            else:
                geometry = (width / 2.0, 4.0, width / 2.0, height - 4.0)
            x1, y1, x2, y2 = geometry
            track_color = resolve_color(COLOR_PAIRS["border_strong"])
            progress_color = resolve_color(
                COLOR_PAIRS[("primary", "success")[index]]
            )
            soft_color = resolve_color(
                COLOR_PAIRS[("primary_soft", "success_soft")[index]]
            )
            track = canvas.create_line(
                x1,
                y1,
                x2,
                y2,
                fill=track_color,
                width=4,
                capstyle=tk.ROUND,
            )
            progress = canvas.create_line(
                x1,
                y1,
                x1,
                y1,
                fill=progress_color,
                width=5,
                capstyle=tk.ROUND,
                state="hidden",
            )
            for x, y in ((x1, y1), (x2, y2)):
                canvas.create_oval(
                    x - 4,
                    y - 4,
                    x + 4,
                    y + 4,
                    fill=background,
                    outline=track_color,
                    width=2,
                )
            trails = [
                canvas.create_oval(0, 0, 0, 0, fill=progress_color, outline="", state="hidden")
                for _trail in range(2)
            ]
            halo = canvas.create_oval(
                0,
                0,
                0,
                0,
                fill=soft_color,
                outline="",
                state="hidden",
            )
            packet = canvas.create_oval(
                0,
                0,
                0,
                0,
                fill=progress_color,
                outline=background,
                width=1,
                state="hidden",
            )
            self._connector_geometry[index] = geometry
            self._connector_items[index] = {
                "track": track,
                "progress": progress,
                "trails": trails,
                "halo": halo,
                "packet": packet,
            }
            self._update_connector(index, self._animation_state.connector_progress[index])
        except tk.TclError:
            self._connector_geometry[index] = None
            self._connector_items[index] = {}

    def _update_connector(self, index: int, progress: float) -> None:
        geometry = self._connector_geometry[index]
        items = self._connector_items[index]
        if geometry is None or not items:
            return
        canvas = self._connector_canvases[index]
        x1, y1, x2, y2 = geometry
        progress = max(0.0, min(1.0, float(progress)))
        packet_x = x1 + ((x2 - x1) * progress)
        packet_y = y1 + ((y2 - y1) * progress)
        moving = 0.0 < progress < 1.0
        try:
            progress_item = items["progress"]
            canvas.coords(progress_item, x1, y1, packet_x, packet_y)
            canvas.itemconfigure(
                progress_item,
                state="normal" if progress > 0.0 else "hidden",
            )
            for trail_index, trail_item in enumerate(items["trails"]):
                trail_progress = max(0.0, progress - ((trail_index + 1) * 0.1))
                trail_x = x1 + ((x2 - x1) * trail_progress)
                trail_y = y1 + ((y2 - y1) * trail_progress)
                radius = 2.5 - (trail_index * 0.5)
                canvas.coords(
                    trail_item,
                    trail_x - radius,
                    trail_y - radius,
                    trail_x + radius,
                    trail_y + radius,
                )
                canvas.itemconfigure(
                    trail_item,
                    state=("normal" if moving and trail_progress > 0.0 else "hidden"),
                )
            canvas.coords(
                items["halo"],
                packet_x - 12,
                packet_y - 12,
                packet_x + 12,
                packet_y + 12,
            )
            canvas.coords(
                items["packet"],
                packet_x - 8,
                packet_y - 8,
                packet_x + 8,
                packet_y + 8,
            )
            canvas.itemconfigure(items["halo"], state="normal" if moving else "hidden")
            canvas.itemconfigure(items["packet"], state="normal" if moving else "hidden")
        except (KeyError, tk.TclError):
            return

    def _apply_animation_state(self, state: _PipelineAnimationState) -> None:
        self._animation_state = state
        stage_style_changed = (
            state.active_stage != self._rendered_active_stage
            or self._paused != self._rendered_paused
        )
        if stage_style_changed:
            for index, card in enumerate(self._cards):
                active = index == state.active_stage
                card.configure(
                    fg_color=(
                        COLOR_PAIRS[self.STAGE_SOFT_COLORS[index]]
                        if active
                        else COLOR_PAIRS["surface"]
                    ),
                    border_color=(
                        COLOR_PAIRS[self.STAGE_STRONG_COLORS[index]]
                        if active
                        else COLOR_PAIRS["border"]
                    ),
                )
            active_stage = state.active_stage
            strong_color = COLOR_PAIRS[self.STAGE_STRONG_COLORS[active_stage]]
            self.stage_detail_shell.configure(border_color=strong_color)
            if self._paused:
                self.motion_status.configure(
                    text=f"Ⅱ  {self.STAGE_LABELS[active_stage]} PAUSED",
                    text_color=COLOR_PAIRS["muted_text"],
                )
            else:
                self.motion_status.configure(
                    text=f"●  {self.STAGE_LABELS[active_stage]} ACTIVE",
                    text_color=strong_color,
                )
            self._rendered_active_stage = state.active_stage
            self._rendered_paused = self._paused
        self._apply_stage_reveal(state.active_stage, state.stage_progress)
        for index, progress in enumerate(state.connector_progress):
            self._update_connector(index, progress)

    def _elapsed_ms(self, now: float | None = None) -> float:
        if self._paused:
            return self._paused_elapsed_ms
        now = time.monotonic() if now is None else now
        return max(0.0, (now - self._animation_started_at) * 1000.0)

    def _schedule_tick(self, delay_ms: int) -> None:
        if (
            self._destroying
            or self._stopped
            or self._paused
            or self._animation_after_id is not None
        ):
            return
        try:
            self._animation_after_id = self.after(
                max(1, int(delay_ms)),
                self._animation_tick,
            )
        except (tk.TclError, TypeError, ValueError):
            self._animation_after_id = None

    def start(self) -> None:
        """Start the decorative animation without creating duplicate timer jobs."""

        self._stopped = False
        self._schedule_tick(self.ANIMATION_INTERVAL_MS)

    def _animation_tick(self) -> None:
        self._animation_after_id = None
        if self._destroying or self._stopped or self._paused:
            return
        now = time.monotonic()
        try:
            visible = bool(self.winfo_exists() and self.winfo_viewable())
        except tk.TclError:
            return
        if not visible:
            if self._hidden_since is None:
                self._hidden_since = now
            self._schedule_tick(self.HIDDEN_RETRY_MS)
            return
        if self._hidden_since is not None:
            self._animation_started_at += now - self._hidden_since
            self._hidden_since = None
        self._apply_animation_state(
            _pipeline_animation_state(self._elapsed_ms(now))
        )
        self._schedule_tick(self.ANIMATION_INTERVAL_MS)

    def _toggle_animation(self) -> None:
        if self._destroying or self._stopped:
            return
        if self._paused:
            self._paused = False
            self._animation_started_at = (
                time.monotonic() - (self._paused_elapsed_ms / 1000.0)
            )
            self._sync_animation_control()
            self._apply_animation_state(self._animation_state)
            self._schedule_tick(self.ANIMATION_INTERVAL_MS)
            return
        self._paused_elapsed_ms = self._elapsed_ms()
        self._paused = True
        self._cancel_after_job("_animation_after_id")
        self._sync_animation_control()
        self._apply_animation_state(self._animation_state)

    def _toggle_animation_from_keyboard(self, _event=None) -> str:
        self._toggle_animation()
        return "break"

    def _show_animation_focus(self, _event=None) -> None:
        self.animation_button.configure(
            border_color=COLOR_PAIRS["primary"],
            border_width=2,
        )

    def _hide_animation_focus(self, _event=None) -> None:
        self.animation_button.configure(
            border_color=COLOR_PAIRS["border"],
            border_width=SHAPES.border_width,
        )

    def _on_theme_changed(self, _event=None) -> None:
        if self._destroying:
            return
        self._refresh_animation_icons()
        self._redraw_stage_icons()
        for index in range(len(self._connector_canvases)):
            self._redraw_connector(index)
        self._rendered_active_stage = None
        self._rendered_paused = None
        self._rendered_reveal = None
        self._apply_animation_state(self._animation_state)

    def _cancel_after_job(self, attribute: str) -> None:
        after_id = getattr(self, attribute, None)
        setattr(self, attribute, None)
        if after_id is None:
            return
        try:
            self.after_cancel(after_id)
        except tk.TclError:
            pass

    def stop(self) -> None:
        """Stop all pending animation/layout work before this page is replaced."""

        self._stopped = True
        self._cancel_after_job("_animation_after_id")
        self._cancel_after_job("_resize_after_id")

    def _unbind_theme_event(self) -> None:
        widget = self._theme_binding_widget
        binding_id = self._theme_binding_id
        self._theme_binding_widget = None
        self._theme_binding_id = None
        if widget is None or binding_id is None:
            return
        try:
            widget.unbind("<<AIDaSThemeChanged>>", binding_id)
        except (AttributeError, tk.TclError):
            pass

    def _on_destroy_event(self, event) -> None:
        if event.widget not in (self, getattr(self, "_canvas", None)):
            return
        self._destroying = True
        self.stop()
        self._unbind_theme_event()

    def destroy(self) -> None:
        if not self._destroying:
            self._destroying = True
            self.stop()
            self._unbind_theme_event()
        super().destroy()


class TutorialDialog(ctk.CTkToplevel):
    """Scrollable, keyboard-friendly guide to the complete AIDaS workflow."""

    PREFERRED_WIDTH = 1100
    PREFERRED_HEIGHT = 780
    MINIMUM_WIDTH = 820
    MINIMUM_HEIGHT = 560
    MAX_SCREEN_FRACTION = 0.92
    CONTENT_VIEWPORT_INSET = 270

    def __init__(
        self,
        owner: tk.Misc,
        *,
        initial_page: int = 0,
        on_step_selected: Callable[[int], None] | None = None,
        on_close: Callable[["TutorialDialog"], None] | None = None,
    ) -> None:
        super().__init__(owner)
        self.withdraw()
        self.title("AIDaS Workflow Tutorial")
        self.configure(fg_color=COLOR_PAIRS["application"])
        self.resizable(True, True)
        self.transient(owner.winfo_toplevel())
        apply_app_icon_to(self)

        bounds = work_area_bounds(self, parent=owner.winfo_toplevel())
        available_width = max(1, bounds[2] - bounds[0])
        available_height = max(1, bounds[3] - bounds[1])
        preferred_physical_width, preferred_physical_height = physical_window_size(
            self,
            self.PREFERRED_WIDTH,
            self.PREFERRED_HEIGHT,
        )
        physical_width = min(
            preferred_physical_width,
            round(available_width * self.MAX_SCREEN_FRACTION),
        )
        physical_height = min(
            preferred_physical_height,
            round(available_height * self.MAX_SCREEN_FRACTION),
        )
        dialog_width, dialog_height = logical_window_size(
            self,
            physical_width,
            physical_height,
        )
        self.minsize(
            min(self.MINIMUM_WIDTH, dialog_width),
            min(self.MINIMUM_HEIGHT, dialog_height),
        )
        self._content_wrap = max(260, min(650, dialog_width - 370))
        self._content_viewport_width = max(
            220,
            dialog_width - self.CONTENT_VIEWPORT_INSET,
        )

        self._owner = owner
        self._page_index = 0
        self._on_step_selected = on_step_selected
        self._on_close_callback = on_close
        self._closing = False
        self._modal_activation_after_id = None
        self._scrollbar_sync_after_id = None
        self._navigation_buttons: list[AppButton] = []
        self._wrapped_labels: list[tuple[ctk.CTkLabel, int]] = []
        self._overview_context_labels: list[ctk.CTkLabel] = []
        self._flow_visualizer: _WorkflowOverviewMap | _WorkflowPipeline | None = None

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.bind("<Escape>", lambda _event: self.close())
        self.bind("<Alt-Left>", lambda _event: self._previous_page())
        self.bind("<Alt-Right>", lambda _event: self._next_page())
        self.geometry(
            centered_logical_geometry(
                self,
                dialog_width,
                dialog_height,
                parent=owner.winfo_toplevel(),
            )
        )
        self.show_page(initial_page)
        self.deiconify()
        synchronize_window_chrome(
            self,
            background=COLOR_PAIRS["window_chrome"],
            foreground=COLOR_PAIRS["text"],
            border=COLOR_PAIRS["window_chrome"],
        )
        self._schedule_modal_activation()

    @property
    def current_page(self) -> TutorialPage:
        return TUTORIAL_PAGES[self._page_index]

    def _build_ui(self) -> None:
        shell = ctk.CTkFrame(
            self,
            fg_color=COLOR_PAIRS["surface"],
            corner_radius=SHAPES.corner_radius_lg,
            border_width=SHAPES.border_width,
            border_color=COLOR_PAIRS["border"],
        )
        shell.pack(fill="both", expand=True, padx=12, pady=12)
        shell.grid_columnconfigure(1, weight=1)
        shell.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(shell, fg_color="transparent", corner_radius=0)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=18, pady=(12, 8))
        header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            header,
            text="AIDaS workflow tutorial",
            anchor="w",
            text_color=COLOR_PAIRS["text"],
            font=ctk.CTkFont(
                family=TYPOGRAPHY.family,
                size=TYPOGRAPHY.title_size,
                weight=TYPOGRAPHY.bold_weight,
            ),
        ).grid(row=0, column=0, sticky="w")
        self.page_indicator = ctk.CTkLabel(
            header,
            text="",
            height=26,
            corner_radius=13,
            fg_color=COLOR_PAIRS["primary_soft"],
            text_color=COLOR_PAIRS["primary"],
            font=ctk.CTkFont(
                family=TYPOGRAPHY.family,
                size=TYPOGRAPHY.caption_size,
                weight=TYPOGRAPHY.semibold_weight,
            ),
        )
        self.page_indicator.grid(row=0, column=1, sticky="e", padx=(12, 0))
        ctk.CTkLabel(
            header,
            text="Select a topic, follow the process, or open the matching workspace step.",
            anchor="w",
            text_color=COLOR_PAIRS["muted_text"],
            font=ctk.CTkFont(family=TYPOGRAPHY.family, size=TYPOGRAPHY.caption_size),
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(2, 0))

        navigation = ctk.CTkFrame(
            shell,
            width=230,
            fg_color=COLOR_PAIRS["sidebar"],
            corner_radius=SHAPES.corner_radius_md,
            border_width=SHAPES.border_width,
            border_color=COLOR_PAIRS["border"],
        )
        navigation.grid(row=1, column=0, sticky="nsew", padx=(18, 10), pady=(0, 8))
        navigation.grid_propagate(False)
        ctk.CTkLabel(
            navigation,
            text="Tutorial topics",
            anchor="w",
            text_color=COLOR_PAIRS["muted_text"],
            font=ctk.CTkFont(
                family=TYPOGRAPHY.family,
                size=TYPOGRAPHY.caption_size,
                weight=TYPOGRAPHY.semibold_weight,
            ),
        ).pack(fill="x", padx=12, pady=(14, 8))

        for page_index, page in enumerate(TUTORIAL_PAGES):
            button = AppButton(
                navigation,
                text=page.navigation_label,
                variant="ghost",
                anchor="w",
                height=40,
                command=lambda index=page_index: self.show_page(index),
            )
            button.pack(fill="x", padx=8, pady=2)
            self._navigation_buttons.append(button)

        ctk.CTkLabel(
            navigation,
            text="Alt + Left / Right changes pages\nEsc closes the tutorial",
            anchor="w",
            justify="left",
            text_color=COLOR_PAIRS["muted_text"],
            font=ctk.CTkFont(family=TYPOGRAPHY.family, size=TYPOGRAPHY.caption_size),
        ).pack(side="bottom", fill="x", padx=12, pady=14)

        self.content_scroll = ctk.CTkScrollableFrame(
            shell,
            fg_color=COLOR_PAIRS["surface_subtle"],
            corner_radius=SHAPES.corner_radius_md,
            border_width=SHAPES.border_width,
            border_color=COLOR_PAIRS["border"],
            scrollbar_button_color=COLOR_PAIRS["border_strong"],
            scrollbar_button_hover_color=COLOR_PAIRS["primary"],
        )
        self.content_scroll.grid(row=1, column=1, sticky="nsew", padx=(0, 18), pady=(0, 8))
        self.content_scroll.grid_columnconfigure(0, weight=1)
        # Detailed pages fit at the normal dialog size, so start without a
        # scrollbar gutter. Overflow detection restores it for small windows.
        self.content_scroll._scrollbar.grid_remove()
        # CTkScrollableFrame's outer widget is wider than its real canvas
        # viewport.  Track the viewport itself so wrapped copy and the
        # responsive pipeline cannot request hidden horizontal overflow.
        self.content_scroll._parent_canvas.bind(
            "<Configure>",
            self._on_content_resize,
            add="+",
        )
        self.content_scroll._parent_canvas.bind(
            "<Configure>",
            self._schedule_scrollbar_sync,
            add="+",
        )
        self.page_host = ctk.CTkFrame(
            self.content_scroll,
            fg_color="transparent",
            corner_radius=0,
        )
        self.page_host.grid(row=0, column=0, sticky="ew", padx=12, pady=6)
        self.page_host.grid_columnconfigure(0, weight=1)

        footer = ctk.CTkFrame(shell, fg_color="transparent", corner_radius=0)
        footer.grid(row=2, column=0, columnspan=2, sticky="ew", padx=18, pady=(0, 12))
        footer.grid_columnconfigure(1, weight=1)

        self.back_button = AppButton(
            footer,
            text="Back",
            variant="secondary",
            width=104,
            command=self._previous_page,
        )
        self.back_button.grid(row=0, column=0, sticky="w")

        self.open_step_button = AppButton(
            footer,
            text="Open step",
            variant="success",
            width=136,
            command=self._open_current_step,
        )
        self.open_step_button.grid(row=0, column=1, sticky="e", padx=(10, 6))

        self.next_button = AppButton(
            footer,
            text="Next",
            variant="primary",
            width=104,
            command=self._next_page,
        )
        self.next_button.grid(row=0, column=2, sticky="e", padx=(0, 6))

        self.close_button = AppButton(
            footer,
            text="Close",
            variant="secondary",
            width=104,
            command=self.close,
        )
        self.close_button.grid(row=0, column=3, sticky="e")

    @staticmethod
    def _body_font(*, bold: bool = False) -> ctk.CTkFont:
        return ctk.CTkFont(
            family=TYPOGRAPHY.family,
            size=TYPOGRAPHY.body_size,
            weight=TYPOGRAPHY.semibold_weight if bold else TYPOGRAPHY.normal_weight,
        )

    def _wrapped_label(
        self,
        parent,
        *,
        wrap_inset: int = 0,
        **kwargs,
    ) -> ctk.CTkLabel:
        """Create a content label whose wrapping follows the live pane width."""

        label = ctk.CTkLabel(
            parent,
            wraplength=max(220, self._content_wrap - wrap_inset),
            **kwargs,
        )
        self._wrapped_labels.append((label, wrap_inset))
        return label

    def _on_content_resize(self, event) -> None:
        """Keep tutorial copy inside the content pane after window resizing."""

        try:
            logical_width, _ = logical_window_size(self, int(event.width), 1)
            if logical_width < 100:
                return
            content_wrap = max(220, min(650, logical_width - 50))
        except (AttributeError, TypeError, ValueError):
            return
        viewport_changed = logical_width != getattr(
            self,
            "_content_viewport_width",
            None,
        )
        self._content_viewport_width = logical_width
        visualizer = getattr(self, "_flow_visualizer", None)
        if viewport_changed and visualizer is not None:
            visualizer.set_available_width(max(220, logical_width - 24))
        if content_wrap == self._content_wrap:
            return
        self._content_wrap = content_wrap
        for label, wrap_inset in tuple(self._wrapped_labels):
            try:
                label.configure(
                    wraplength=max(220, self._content_wrap - wrap_inset)
                )
            except tk.TclError:
                pass
        overview_wrap = max(150, (self._content_wrap - 34) // 2)
        for label in tuple(getattr(self, "_overview_context_labels", ())):
            try:
                label.configure(wraplength=overview_wrap)
            except tk.TclError:
                pass

    def _schedule_scrollbar_sync(self, _event=None) -> None:
        """Recheck overflow after Tk has settled the current page geometry."""

        if getattr(self, "_closing", False):
            return
        if getattr(self, "_scrollbar_sync_after_id", None) is not None:
            return
        try:
            self._scrollbar_sync_after_id = self.after_idle(
                self._sync_content_scrollbar
            )
        except (AttributeError, tk.TclError):
            self._scrollbar_sync_after_id = None

    def _sync_content_scrollbar(self) -> None:
        """Hide the scrollbar when the whole tutorial page already fits."""

        self._scrollbar_sync_after_id = None
        if getattr(self, "_closing", False):
            return
        try:
            canvas = self.content_scroll._parent_canvas
            scrollbar = self.content_scroll._scrollbar
            bounds = canvas.bbox("all")
            content_height = 0 if bounds is None else max(0, bounds[3] - bounds[1])
            viewport_height = max(1, int(canvas.winfo_height()))
            has_overflow = content_height > viewport_height + 2
            is_visible = bool(scrollbar.winfo_manager())
            if has_overflow and not is_visible:
                scrollbar.grid()
            elif not has_overflow and is_visible:
                scrollbar.grid_remove()
                canvas.yview_moveto(0)
        except (AttributeError, IndexError, TypeError, ValueError, tk.TclError):
            pass

    def _section_title(self, parent, text: str, *, pady=(18, 8)) -> None:
        ctk.CTkLabel(
            parent,
            text=text,
            anchor="w",
            text_color=COLOR_PAIRS["text"],
            font=ctk.CTkFont(
                family=TYPOGRAPHY.family,
                size=TYPOGRAPHY.subtitle_size,
                weight=TYPOGRAPHY.bold_weight,
            ),
        ).pack(fill="x", pady=pady)

    def _render_overview_context(self, parent, page: TutorialPage) -> None:
        """Keep the overview's completion check and notes in one compact row."""

        context_panel = ctk.CTkFrame(
            parent,
            fg_color="transparent",
            corner_radius=0,
        )
        context_panel.pack(fill="x", pady=(10, 0))
        for column in range(2):
            context_panel.grid_columnconfigure(
                column,
                weight=1,
                uniform="overview_context",
            )

        context_wrap = max(150, (self._content_wrap - 34) // 2)
        cards = (
            (
                "READY WHEN",
                page.completion_check,
                "success",
                "success_soft",
            ),
            (
                "KEY NOTES",
                "\n".join(f"\u2022  {tip}" for tip in page.tips),
                "primary",
                "surface",
            ),
        )
        for column, (heading, copy, strong_color, background) in enumerate(cards):
            card = ctk.CTkFrame(
                context_panel,
                fg_color=COLOR_PAIRS[background],
                corner_radius=SHAPES.corner_radius_sm,
                border_width=SHAPES.border_width,
                border_color=COLOR_PAIRS[strong_color],
            )
            card.grid(
                row=0,
                column=column,
                sticky="nsew",
                padx=(0, 4) if column == 0 else (4, 0),
            )
            ctk.CTkLabel(
                card,
                text=heading,
                anchor="w",
                text_color=COLOR_PAIRS[strong_color],
                font=ctk.CTkFont(
                    family=TYPOGRAPHY.family,
                    size=TYPOGRAPHY.caption_size,
                    weight=TYPOGRAPHY.bold_weight,
                ),
            ).pack(fill="x", padx=8, pady=(6, 1))
            copy_label = ctk.CTkLabel(
                card,
                text=copy,
                anchor="nw",
                justify="left",
                wraplength=context_wrap,
                text_color=COLOR_PAIRS["text"],
                font=ctk.CTkFont(
                    family=TYPOGRAPHY.family,
                    size=TYPOGRAPHY.caption_size,
                ),
            )
            copy_label.pack(fill="both", expand=True, padx=8, pady=(0, 6))
            self._overview_context_labels.append(copy_label)

    def _render_flow(self, parent, page: TutorialPage) -> None:
        available_width = max(220, self._content_viewport_width - 24)
        if page.key == "overview":
            visualizer = _WorkflowOverviewMap(
                parent,
                available_width=available_width,
            )
            pack_options = {"fill": "x"}
        else:
            visualizer = _WorkflowPipeline(
                parent,
                (
                    page.input_summary,
                    page.function_summary,
                    page.output_summary,
                ),
                page.stage_points,
                page.completion_check,
                page.tips,
                available_width=available_width,
            )
            pack_options = {"fill": "x", "pady": (10, 0)}
        visualizer.pack(**pack_options)
        self._flow_visualizer = visualizer

    def _stop_flow_visualizer(self) -> None:
        visualizer = getattr(self, "_flow_visualizer", None)
        self._flow_visualizer = None
        if visualizer is not None:
            visualizer.stop()

    def _render_page(self, page: TutorialPage) -> None:
        visualizer = getattr(self, "_flow_visualizer", None)
        if page.key != "overview" and isinstance(visualizer, _WorkflowPipeline):
            reused = visualizer.replace_content(
                (
                    page.input_summary,
                    page.function_summary,
                    page.output_summary,
                ),
                page.stage_points,
                page.completion_check,
                page.tips,
            )
            if reused:
                self._page_title_label.configure(text=page.title)
                self._page_purpose_label.configure(text=page.purpose)
                return

        self._stop_flow_visualizer()
        for child in self.page_host.winfo_children():
            child.destroy()
        self._wrapped_labels.clear()
        self._overview_context_labels.clear()

        self._page_title_label = self._wrapped_label(
            self.page_host,
            text=page.title,
            anchor="w",
            justify="left",
            text_color=COLOR_PAIRS["text"],
            font=ctk.CTkFont(
                family=TYPOGRAPHY.family,
                size=TYPOGRAPHY.heading_size,
                weight=TYPOGRAPHY.bold_weight,
            ),
        )
        self._page_title_label.pack(fill="x", pady=(0, 3))
        self._page_purpose_label = self._wrapped_label(
            self.page_host,
            text=page.purpose,
            anchor="nw",
            justify="left",
            text_color=COLOR_PAIRS["muted_text"],
            font=self._body_font(),
        )
        self._page_purpose_label.pack(fill="x")

        if page.key == "overview":
            self._section_title(
                self.page_host,
                "Four-step workflow",
                pady=(12, 5),
            )
            self._render_flow(self.page_host, page)
            self._render_overview_context(self.page_host, page)
        else:
            self._render_flow(self.page_host, page)

    def show_page(self, page_index: int) -> None:
        """Display one valid page and synchronize all navigation controls."""

        try:
            page_index = int(page_index)
        except (TypeError, ValueError):
            page_index = 0
        self._page_index = max(0, min(len(TUTORIAL_PAGES) - 1, page_index))
        page = self.current_page
        self._render_page(page)

        self.page_indicator.configure(
            text=f"  {self._page_index + 1} of {len(TUTORIAL_PAGES)}  "
        )
        for index, button in enumerate(self._navigation_buttons):
            selected = index == self._page_index
            button.configure(
                fg_color=(
                    COLOR_PAIRS["primary_soft"]
                    if selected
                    else "transparent"
                ),
                hover_color=COLOR_PAIRS["primary_soft"],
                border_color=(
                    COLOR_PAIRS["primary"]
                    if selected
                    else COLOR_PAIRS["border"]
                ),
                text_color=(
                    COLOR_PAIRS["primary"]
                    if selected
                    else COLOR_PAIRS["text"]
                ),
            )

        self.back_button.configure(
            state="disabled" if self._page_index == 0 else "normal"
        )
        self.next_button.configure(
            state=(
                "disabled"
                if self._page_index == len(TUTORIAL_PAGES) - 1
                else "normal"
            )
        )
        if page.step_index is None:
            self.open_step_button.grid_remove()
        else:
            self.open_step_button.configure(text=f"Open Step {page.step_index + 1}")
            self.open_step_button.grid()

        try:
            self.content_scroll._parent_canvas.yview_moveto(0)
        except (AttributeError, tk.TclError):
            pass
        self._schedule_scrollbar_sync()

    def show_step(self, step_index: int) -> None:
        """Open the page matching a zero-based workflow step."""

        self.show_page(tutorial_page_index_for_step(step_index))

    def _previous_page(self) -> str:
        if self._page_index > 0:
            self.show_page(self._page_index - 1)
        return "break"

    def _next_page(self) -> str:
        if self._page_index < len(TUTORIAL_PAGES) - 1:
            self.show_page(self._page_index + 1)
        return "break"

    def _open_current_step(self) -> None:
        page = self.current_page
        callback = self._on_step_selected
        if page.step_index is None or callback is None:
            return
        step_index = page.step_index
        self.close()
        callback(step_index)

    def _schedule_modal_activation(self, *, delay_ms: int = 50) -> None:
        """Acquire the modal grab only after Windows has mapped the tutorial."""

        pending = getattr(self, "_modal_activation_after_id", None)
        if pending is not None:
            try:
                self.after_cancel(pending)
            except tk.TclError:
                pass
            self._modal_activation_after_id = None
        try:
            self.deiconify()
            self._modal_activation_after_id = self.after(
                max(1, int(delay_ms)),
                self._activate_modal_when_visible,
            )
        except (tk.TclError, TypeError, ValueError):
            self._modal_activation_after_id = None

    def _activate_modal_when_visible(self, attempt: int = 0) -> None:
        """Finish the modal handoff without leaving an invisible input grab."""

        self._modal_activation_after_id = None
        try:
            if not self.winfo_exists():
                return
            if not self.winfo_viewable():
                self.deiconify()
                if attempt < 20:
                    self._modal_activation_after_id = self.after(
                        25,
                        lambda: self._activate_modal_when_visible(attempt + 1),
                    )
                return
            self.lift()
            self.grab_set()
            self.focus_force()
        except tk.TclError:
            self._modal_activation_after_id = None

    def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        self._stop_flow_visualizer()
        callback = self._on_close_callback
        activation_id = getattr(self, "_modal_activation_after_id", None)
        self._modal_activation_after_id = None
        scrollbar_sync_id = getattr(self, "_scrollbar_sync_after_id", None)
        self._scrollbar_sync_after_id = None
        if activation_id is not None:
            try:
                self.after_cancel(activation_id)
            except tk.TclError:
                pass
        if scrollbar_sync_id is not None:
            try:
                self.after_cancel(scrollbar_sync_id)
            except tk.TclError:
                pass
        try:
            try:
                self.grab_release()
            except tk.TclError:
                pass
            self.destroy()
        finally:
            if callback is not None:
                callback(self)


__all__ = [
    "TUTORIAL_PAGES",
    "TutorialDialog",
    "TutorialPage",
    "tutorial_page_index_for_step",
]
