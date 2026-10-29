"""In-app tutorial for the five-stage AIDaS processing workflow."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
import tkinter as tk
from typing import Callable

import customtkinter as ctk

from aidas.core.display import work_area_bounds
from aidas.ui.components import AppButton
from aidas.ui.rendering import finish_deferred_rendering
from aidas.ui.theme import (
    COLOR_PAIRS,
    SHAPES,
    TYPOGRAPHY,
)
from aidas.ui.windowing import (
    centered_logical_geometry,
    logical_window_size,
    physical_window_size,
    synchronize_window_chrome,
)
from aidas.utils.ui_utils import apply_app_icon_to


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
            "AIDaS keeps selected subject folder(s) moving through five connected stages. "
            "Each handoff opens the exact folders completed by the previous step, "
            "while every stage can also scan a parent folder when work is resumed later."
        ),
        input_summary="Nested subject folders containing raw .sdb OCT scans",
        function_summary="Crop, segment, flatten, analyze, then compile measurements",
        output_summary="Analyze images, diagnostics, ROI stacks, and one compiled Excel workbook",
        completion_check=(
            "The workflow is complete when Step 5 has combined the selected Step 3 "
            "and Step 4 measurements into rrMCPAR_ELMRPE_ONL.xlsx."
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
                "Define the ROI by dragging on the image or entering X, Y, Width, and Height; the crop menu offers Auto OCT Region and Whole Image.",
            ),
            (
                "Choose Crop & Scale to create the Target view. AIDaS repeats the crop three times horizontally and once vertically; Undo returns to the editable Source view.",
                "Choose Save to write or replace the current result, mark the queue row complete, and open the next SDB image automatically.",
            ),
            (
                "Each processed folder receives a two-slice, 16-bit light.hdr and light.img Analyze pair.",
                "Completed queue rows identify the folders that are ready for segmentation.",
                "Go to Step 2 opens those completed folders in a preloaded review; confirm the selected rows and choose Run.",
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
        navigation_label="Step 2 - Segment",
        title="Step 2 - Segment retinal boundaries",
        purpose=(
            "Step 2 identifies the six retinal layer boundaries and foveal center "
            "required by flattening. AI generates the boundary traces for review. "
            "Boundary editing is disabled by default; the foveal center remains adjustable."
        ),
        input_summary="Step 1 Light Analyze pairs, loaded from selected folders",
        function_summary="Confirm the fovea, identify the image as temporal or nasal, and review six boundaries",
        output_summary="Nasal and temporal folders containing 16-bit Light and 8-bit Light_MARKED pairs",
        completion_check=(
            "All six boundaries and the foveal line have been reviewed, the image side "
            "is correct, and each nasal/temporal folder contains Light and Light_MARKED Analyze pairs."
        ),
        stage_points=(
            (
                "Use the Step 1 handoff or Select folders to segment, then review the discovered Light images and choose the ready rows.",
                "Folders that already contain Light_MARKED are identified and skipped.",
                "For each image, drag the foveal marker to the center, then choose Set Fovea.",
                "Next identify the image as Temporal or Nasal. Both prompts are required; Exit cancels the batch.",
            ),
            (
                "Use the boundary legend to match each layer to its color on the image. Boundary lines are locked by default; the foveal marker can still be adjusted.",
                "For ImageJ-style editing, turn on Enable manual segmentation (Step 2) in Settings > Step 2: Segmentation. Select an incomplete boundary to draw; double-click to finish. Select a completed boundary to edit its handles. The arrow buttons revert or regenerate lines.",
            ),
            (
                "Save or Save all commits the reviewed results; unsaved AI results and edits exist only in the open review tabs.",
                "AIDaS creates nasal and temporal child folders with normalized two-slice, 2133-pixel-wide 16-bit Light and 8-bit Light_MARKED pairs, mirrored according to the selected image side.",
                "Go to Step 3 preflights, saves, and hands the completed nasal and temporal folders to the R batch review.",
            ),
        ),
        tips=(
            "The lists track RNFL-Vitreous, GCL-RNFL, INL-IPL, ONL-OPL, ELM, and RPE.",
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
                "Select ready folders, set Batch size, and choose Parallel or Sequential execution. R scripts run without an automatic timeout, and the core limit is shared with active Step 2 CPU work.",
            ),
            (
                "Run selected folders and follow each folder’s progress and log. Stop terminates active R process trees and cancels queued folders; failed and cancelled folders remain identified.",
                "Review temporal and nasal vertex and tissue-border images together in the results grid; click any image for a zoomed view, or use Load R results to reopen a completed folder.",
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
            "Original and the default Optimized version retain all six boundaries. For Optimized deeper layers, select the matching main and output scripts in Settings. It preserves RPE, OLM, ONL/OPL, INL/IPL, and foveal calculations; the surface boundary remains an internal reference while GCL/RNFL processing is skipped.",
            "Deeper-layer outputs retain all 90 depth positions and eight thickness columns. Omitted superficial results are NA; zero-filled regions in normalized Analyze images are display placeholders, not measured brightness. Step 5's ELM-RPE and ONL columns remain available.",
        ),
        step_index=2,
    ),
    TutorialPage(
        key="step4",
        navigation_label="Step 4 - MCP/AR",
        title="Step 4 - MCP/AR profile measurements",
        purpose=(
            "Step 4 measures the inner-segment ellipsoid zone across 20 peripheral "
            "ROIs and one foveal ROI, then reproduces the lab's MATLAB/ImageJ-style outputs."
        ),
        input_summary="Completed Step 3 _flat_LIGHT Analyze pairs, using slice 0",
        function_summary="Automatically detect or manually revise profile bounds for 21 ROIs and calculate shape measurements",
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
                "Use Auto-detect ROIs 1-20 to propose Start positions in 70-90 and End positions in 90-110. ROI 21 has a different shape and always requires manual Start and End selection. Plot clicks snap within a fixed ±6-sample window by default. Check Disable automatic line positioning between the grid and bottom plot to keep the lines where you place them.",
            ),
            (
                "For ROIs 1-20, automatic detection smooths each profile with a degree-2 Savitzky-Golay filter, finds two minima around an internal maximum, validates peak prominence and quadratic fit, and checks agreement across those ROIs. High-confidence results are accepted; ROI 21 and other amber ! cells require review.",
                "Review the grid previews and Major, Minor, Angle, Circ., AR, Round, and Solidity measurements. Click an amber or incorrect grid preview, click the exact sample or drag the Start or End line in the zoomed editor, and use Apply changes to confirm it. Zoomed manual edits do not snap to a nearby minimum.",
            ),
            (
                "When all 21 ROIs are complete, Build stack becomes available and marks the folder tab Done after a successful build.",
                "The build shows a wait message while writing ROI_to_move_stck.tif, MAX_Stack.tif, and rr_MCPAR.xlsx, then advances to the next incomplete folder. A completed plot tab provides Open results folder, Restart this file, and Go to Step 5 actions.",
                "Go to Step 5 opens Create Excel File, carries forward the measurement root for all tabs, and starts compilation automatically.",
            ),
        ),
        tips=(
            "Build stack remains disabled until every one of the 21 ROI rows is complete.",
            "Automatic detection never processes ROI 21 or silently accepts a low-confidence curve; every manual or uncertain ROI remains amber until you confirm it.",
            "Revising a completed ROI updates its measurements without advancing, so the change can be verified immediately.",
            "The folder scanner also accepts legacy Results.xlsx or Results_org.xlsx as the measurement workbook when deciding that earlier work is complete.",
        ),
        step_index=3,
    ),
    TutorialPage(
        key="step5",
        navigation_label="Step 5 - Create Excel File",
        title="Step 5 - Create Excel File",
        purpose=(
            "Step 5 combines measurements produced throughout the workflow into "
            "one Excel file for downstream review and analysis."
        ),
        input_summary="A group / subject / nasal or temporal folder tree containing Step 3 text exports and Step 4 workbooks",
        function_summary="Combine rrMCP/AR, ELM-RPE, and ONL measurements into an Excel file",
        output_summary="rrMCPAR_ELMRPE_ONL.xlsx",
        completion_check=(
            "A completion message confirms the Excel file was saved, and the "
            "selected output path contains the compiled workbook (rrMCPAR_ELMRPE_ONL.xlsx by default)."
        ),
        stage_points=(
            (
                "Choose Go to Step 5 in Step 4 to transfer its measurement folders and start compilation automatically. Switching workspace tabs only opens the workspace. For manual compilation, browse to a folder containing group and subject folders at any depth.",
                "Review the Excel file path and choose a different .xlsx destination if needed.",
                "Choose whether to include the fovea RPEtoOLM row in the ELM-RPE sheet.",
                "Select Create Excel File to run a manual compilation or compile again after changing options.",
            ),
            (
                "AIDaS discovers current rr_MCPAR.xlsx and compatible legacy workbooks, then reads the Step 3 ELM-RPE and ONL exports. The nasal/temporal parent folder supplies the full subject ID. Its parent folder supplies the group name, including custom names or LE and RE. Folders without Excel results are ignored.",
                "Explore the Folder contents canvas to see recorded Excel results. Hide other files is checked by default; uncheck it to also see thickness tables, fovea measurements, and image stacks. Expand or collapse folders and use Refresh to read changes. Files outside nasal/temporal folders are visible with an explanation. Inputs stay locked while the Excel file is being created.",
            ),
            (
                "AIDaS creates the rrMCP-AR, ELM-RPE, and ONL worksheets.",
                "The Excel file is saved at the selected output path.",
                "A completion message confirms that the five-step workflow is finished. The saved workbook appears in the folder tree when it is inside the selected directory.",
            ),
        ),
        tips=(
            "Select a common parent to include every group. Each group gets ID, Nasal, and Temporal columns on each measurement sheet.",
            "AIDaS accepts both current rr_MCPAR.xlsx outputs and recognized legacy result names.",
        ),
        step_index=4,
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


class _TextMeasurements:
    """Cache Tk's complete text layout without measuring individual words."""

    MAX_HEIGHTS = 64

    def __init__(self, font: ctk.CTkFont, master: tk.Misc, *, scaling: float = 1.0) -> None:
        # An unmanaged native label computes requested geometry immediately.
        # It never paints or participates in the visible layout, and its parent
        # destroys it with the tutorial. Match CTk's actual scaled font so pixel
        # rounding cannot change line breaks on high-DPI displays.
        self._scaling = scaling
        self._label = tk.Label(
            master, font=font.create_scaled_tuple(scaling),
            borderwidth=0, highlightthickness=0, padx=0, pady=0,
        )
        self._heights: OrderedDict[tuple[tuple[str, ...], int], int] = OrderedDict()

    def copy_height(self, font: ctk.CTkFont, copies: tuple[str, ...], width: int) -> int:
        key = (copies, width)
        if key in self._heights:
            self._heights.move_to_end(key)
            return self._heights[key]
        self._label.configure(
            font=font.create_scaled_tuple(self._scaling),
            wraplength=width * self._scaling,
        )
        height = 0
        for copy in copies:
            self._label.configure(text=copy)
            height = max(height, self._label.winfo_reqheight())
        height = math.ceil(height / self._scaling) + 8
        self._heights[key] = height
        if len(self._heights) > self.MAX_HEIGHTS:
            self._heights.popitem(last=False)
        return height


class _WorkflowOverviewMap(ctk.CTkFrame):
    """Static responsive map of the five AIDaS workflow steps."""

    TWO_COLUMN_BREAKPOINT = 480
    STEP_CARDS = (
        (
            "Load & Crop",
            "Decode raw .sdb scans and create the light.hdr/light.img Analyze pair.",
            "primary",
            "primary_soft",
        ),
        (
            "Segment",
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
            "MCP/AR",
            "Measure 21 ROIs and build the TIFF stacks and Excel measurements.",
            "success",
            "success_soft",
        ),
        (
            "Create Excel File",
            "Combine rrMCP/AR, ELM-RPE, and ONL measurements into one workbook.",
            "primary",
            "primary_soft",
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
        """Lay out workflow step cards without introducing horizontal overflow."""

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
        """Match the guide lifecycle contract; no timer exists."""


class _WorkflowPipeline(ctk.CTkFrame):
    """Immediate, reader-controlled instructions with a fixed set of reusable rows."""

    SECTIONS = ("1  Prepare", "2  Run & review", "3  Save & continue", "Details")

    def __init__(
        self, master, summaries: tuple[str, str, str],
        stage_points: StagePointGroups, completion_check: str,
        tips: tuple[str, ...] = (), *, available_width: int = 650,
    ) -> None:
        super().__init__(master, fg_color="transparent", corner_radius=0)
        self._available_width = None
        self._section = 0
        self._summaries = summaries
        self._stage_points = stage_points
        self._completion_check = completion_check
        self._tips = tips
        self._body_labels = []
        self._text_measurements = {}
        self._section_values = self.SECTIONS
        self.sections = ctk.CTkSegmentedButton(
            self, values=self.SECTIONS, command=self._section_selected,
            height=38, font=ctk.CTkFont(family=TYPOGRAPHY.family, size=13),
            selected_color=COLOR_PAIRS["primary_soft"],
            selected_hover_color=COLOR_PAIRS["primary_soft"],
            unselected_color=COLOR_PAIRS["surface"],
            unselected_hover_color=COLOR_PAIRS["primary_soft"],
            fg_color=COLOR_PAIRS["surface"],
            text_color=COLOR_PAIRS["text"],
        )
        self.sections.pack(fill="x", pady=(0, 16))
        self.summary = ctk.CTkLabel(
            self, text="", anchor="w", justify="left",
            text_color=COLOR_PAIRS["text"],
            font=ctk.CTkFont(family=TYPOGRAPHY.family, size=15, weight="bold"),
        )
        self.summary.pack(fill="x", pady=(0, 10))
        self._body_labels.append((self.summary, 8))
        self.rows = []
        self._badges = []
        # Reuse these rows across sections and pages; avoid repeated widget creation.
        for index in range(max(4, len(tips), *(len(group) for group in stage_points))):
            row = ctk.CTkFrame(self, fg_color=COLOR_PAIRS["surface"], corner_radius=8)
            row.grid_columnconfigure(1, weight=1)
            row.grid_propagate(False)
            row.pack(fill="x", pady=3)
            badge = ctk.CTkLabel(
                row, text=str(index + 1), width=26, height=26, corner_radius=13,
                fg_color=COLOR_PAIRS["primary_soft"], text_color=COLOR_PAIRS["primary"],
                font=ctk.CTkFont(family=TYPOGRAPHY.family, size=13, weight="bold"),
            )
            badge.grid(row=0, column=0, sticky="nw", padx=(12, 10), pady=12)
            label = ctk.CTkLabel(
                row, text="", anchor="nw", justify="left",
                text_color=COLOR_PAIRS["text"],
                font=ctk.CTkFont(family=TYPOGRAPHY.family, size=14),
            )
            label.grid(row=0, column=1, sticky="ew", padx=(0, 14), pady=12)
            self._body_labels.append((label, 80))
            self.rows.append((row, label))
            self._badges.append(badge)
        self.ready = ctk.CTkFrame(
            self, fg_color=COLOR_PAIRS["success_soft"], corner_radius=8,
        )
        self.ready.pack(fill="x", pady=(14, 2))
        ctk.CTkLabel(
            self.ready, text="READY TO CONTINUE WHEN", anchor="w",
            text_color=COLOR_PAIRS["success"],
            font=ctk.CTkFont(family=TYPOGRAPHY.family, size=12, weight="bold"),
        ).pack(fill="x", padx=14, pady=(10, 2))
        self.completion = ctk.CTkLabel(
            self.ready, text=completion_check, anchor="w", justify="left",
            text_color=COLOR_PAIRS["text"],
            font=ctk.CTkFont(family=TYPOGRAPHY.family, size=13),
        )
        self.completion.pack(fill="x", padx=14, pady=(0, 12))
        self._body_labels.append((self.completion, 40))
        self.set_available_width(available_width)
        self.sections.set(self.sections.cget("values")[0])
        self._select_section(self.SECTIONS[0])

    def _select_section(self, value: str) -> None:
        self._section = self.SECTIONS.index(value)
        details = self._section == 3
        points = self._tips if details else self._stage_points[self._section]
        summary = "Tips & technical details" if details else self._summaries[self._section]
        if self.summary.cget("text") != summary:
            self.summary.configure(text=summary)
        for index, (row, label) in enumerate(self.rows):
            text = points[index] if index < len(points) else ""
            if label.cget("text") != text:
                label.configure(text=text)
            if index >= len(points):
                if row.winfo_manager():
                    row.pack_forget()
            elif not row.winfo_manager():
                row.pack(fill="x", pady=3, before=self.ready)
        self.winfo_toplevel().event_generate("<<TutorialSectionChanged>>")

    def _section_selected(self, value: str) -> None:
        index = self._section_values.index(value)
        if index != self._section:
            self._select_section(self.SECTIONS[index])

    def replace_content(self, summaries, stage_points, completion_check, tips) -> bool:
        required = max(len(tips), *(len(group) for group in stage_points))
        if required > len(self.rows):
            return False
        self._summaries = summaries
        self._stage_points = stage_points
        self._tips = tips
        self._completion_check = completion_check
        self.completion.configure(text=completion_check)
        self.sections.set(self.sections.cget("values")[0])
        self._select_section(self.SECTIONS[0])
        return True

    def set_available_width(self, logical_width: int) -> None:
        width = max(180, int(logical_width))
        if width == self._available_width:
            return
        self._available_width = width
        for label, inset in self._body_labels:
            label.configure(wraplength=max(100, width - inset))
        self._size_template(width)
        # Keep all section controls reachable on narrow/high-DPI displays.
        values = self.SECTIONS if width >= 540 else ("Prepare", "Run", "Save", "Details")
        if values != self._section_values:
            self._section_values = values
            self.sections.configure(values=values)
            self.sections.set(values[self._section])

    @staticmethod
    def _copy_height(label, copies, width: int, *, measurements=None) -> int:
        """Reserve space for all pages without repeating native word measurements."""

        font = label.cget("font")
        if measurements is None:
            measurements = label.__dict__.setdefault("_tutorial_text_measurements", {})
        # Equal font descriptions can share metrics, including the four body rows.
        # Re-read the description so font changes invalidate cached measurements.
        scaling = label._get_widget_scaling()
        signature = (font.cget("size"), tuple(sorted(font.actual().items())), scaling)
        if signature not in measurements:
            measurements[signature] = _TextMeasurements(font, label, scaling=scaling)
        return measurements[signature].copy_height(font, tuple(copies), width)

    def _size_template(self, width: int) -> None:
        pages = TUTORIAL_PAGES[1:]
        summary_height = self._copy_height(
            self.summary,
            [copy for page in pages for copy in (
                page.input_summary, page.function_summary, page.output_summary,
            )] + ["Tips & technical details"],
            max(100, width - 8),
            measurements=self._text_measurements,
        )
        if self.summary.cget("height") != summary_height:
            self.summary.configure(height=summary_height)
        for index, (row, label) in enumerate(self.rows):
            copies = [
                group[index]
                for page in pages
                for group in (*page.stage_points, page.tips)
                if index < len(group)
            ]
            text_height = self._copy_height(
                label, copies, max(100, width - 80),
                measurements=self._text_measurements,
            )
            # Keep the text widget's allocation stable as well as the card's.
            # Otherwise every section swaps in new canvas sizes and paints them.
            if label.cget("height") != text_height:
                label.configure(height=text_height)
            row_height = max(54, text_height + 24)
            if row.cget("height") != row_height:
                row.configure(height=row_height)
        completion_height = self._copy_height(
            self.completion, [page.completion_check for page in pages],
            max(100, width - 40),
            measurements=self._text_measurements,
        )
        if self.completion.cget("height") != completion_height:
            self.completion.configure(height=completion_height)

    @staticmethod
    def stop() -> None:
        """No animation timers or theme-event handlers to tear down."""


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
        # Scrollbars flush Tk's idle queue during their own draws. Keep those
        # flushes from painting the tutorial at intermediate construction sizes.
        self._aidas_defer_canvas_updates = True
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
        self._content_resize_after_id = None
        self._pending_content_width = None
        self._text_measurements = {}
        self._navigation_buttons: list[AppButton] = []
        self._wrapped_labels: list[tuple[ctk.CTkLabel, int]] = []
        self._overview_context_labels: list[ctk.CTkLabel] = []
        self._flow_visualizer: _WorkflowOverviewMap | _WorkflowPipeline | None = None

        # Install the native frame and caption while the window is still empty.
        # Installing it after mapping would lay out and paint the full guide twice.
        synchronize_window_chrome(
            self,
            background=COLOR_PAIRS["window_chrome"],
            foreground=COLOR_PAIRS["text"],
            border=COLOR_PAIRS["window_chrome"],
        )
        self._build_ui()
        self.bind("<<TutorialSectionChanged>>", self._on_section_changed)
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
        finish_deferred_rendering(self)
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
            text="AIDaS quick guide",
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
            text="Choose a step. Read at your pace. Open the workspace when ready.",
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
        self.content_scroll._parent_canvas.configure(
            yscrollcommand=self._update_content_scrollbar,
        )
        # Reserve the scrollbar's width even while it is hidden. Switching
        # sections must not change wrapping and trigger another full layout.
        self._scrollbar_gutter = ctk.CTkFrame(
            self.content_scroll._parent_frame,
            width=self.content_scroll._scrollbar.cget("width"),
            height=1, fg_color="transparent", corner_radius=0,
        )
        self._scrollbar_gutter.grid(
            row=1, column=1, sticky="ns",
            padx=(0, self.content_scroll._border_width + 1),
        )
        self._scrollbar_gutter.lower(self.content_scroll._scrollbar)
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
        self.page_host.bind("<Configure>", self._schedule_scrollbar_sync, add="+")

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
            variant="ghost",
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
        """Coalesce resize events so dragging a window cannot flood the UI thread."""

        if getattr(self, "_closing", False):
            return
        try:
            logical_width, _ = logical_window_size(self, int(event.width), 1)
            if logical_width < 100:
                return
        except (AttributeError, TypeError, ValueError):
            return
        if (
            logical_width == self._content_viewport_width
            and getattr(self, "_pending_content_width", None) is None
        ):
            return
        self._pending_content_width = logical_width
        if getattr(self, "_content_resize_after_id", None) is None:
            try:
                self._content_resize_after_id = self.after(40, self._apply_content_resize)
            except tk.TclError:
                self._pending_content_width = None

    def _apply_content_resize(self) -> None:
        """Apply only the most recent logical viewport size."""

        self._content_resize_after_id = None
        logical_width = self._pending_content_width
        self._pending_content_width = None
        if logical_width is None or getattr(self, "_closing", False):
            return
        content_wrap = max(220, min(650, logical_width - 50))
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
        self._size_page_headings()
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

    def _update_content_scrollbar(self, start, end) -> None:
        """Update the thumb without flushing partially updated section layouts."""

        scrollbar = self.content_scroll._scrollbar
        fractions = float(start), float(end)
        if scrollbar.get() == fractions:
            return
        already_deferred = self._aidas_defer_canvas_updates
        self._aidas_defer_canvas_updates = True
        try:
            scrollbar.set(*fractions)
        finally:
            if not already_deferred:
                finish_deferred_rendering(self)

    def _size_page_headings(self) -> None:
        for name, copies in (
            ("_page_title_label", [page.title for page in TUTORIAL_PAGES]),
            ("_page_purpose_label", [page.purpose for page in TUTORIAL_PAGES]),
        ):
            label = self.__dict__.get(name)
            if label is not None:
                height = _WorkflowPipeline._copy_height(
                    label, copies, self._content_wrap,
                    measurements=self._text_measurements,
                )
                if label.cget("height") != height:
                    label.configure(height=height)

    def _on_section_changed(self, _event=None) -> None:
        """Start each selected section at its first instruction."""

        canvas = self.content_scroll._parent_canvas
        if canvas.yview()[0] > 0:
            canvas.yview_moveto(0)
        self._schedule_scrollbar_sync()

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

        # Keep both layouts alive. Going through the overview must not destroy
        # and reconstruct the shared template for the five workflow steps.
        panels = self.__dict__.setdefault("_page_panels", {})
        kind = "overview" if page.key == "overview" else "step"
        for panel, _visualizer in panels.values():
            panel.pack_forget()

        if "_page_title_label" not in self.__dict__:
            self._page_title_label = self._wrapped_label(
                self.page_host, text="", anchor="w", justify="left",
                text_color=COLOR_PAIRS["text"],
                font=ctk.CTkFont(family=TYPOGRAPHY.family,
                                size=TYPOGRAPHY.heading_size, weight="bold"),
            )
            self._page_title_label.pack(fill="x", pady=(0, 3))
            self._page_purpose_label = self._wrapped_label(
                self.page_host, text="", anchor="nw", justify="left",
                text_color=COLOR_PAIRS["muted_text"], font=self._body_font(),
            )
            self._page_purpose_label.pack(fill="x")
            self._size_page_headings()
        self._page_title_label.configure(text=page.title)
        self._page_purpose_label.configure(text=page.purpose)

        if kind not in panels:
            panel = ctk.CTkFrame(self.page_host, fg_color="transparent")
            if kind == "overview":
                self._section_title(panel, "Five-step workflow", pady=(12, 5))
                self._render_flow(panel, page)
                self._render_overview_context(panel, page)
            else:
                self._render_flow(panel, page)
            panels[kind] = (panel, self._flow_visualizer)
        else:
            panel, visualizer = panels[kind]
            self._flow_visualizer = visualizer
            visualizer.set_available_width(max(220, self._content_viewport_width - 24))
            if kind == "step":
                visualizer.replace_content(
                    (page.input_summary, page.function_summary, page.output_summary),
                    page.stage_points, page.completion_check, page.tips,
                )
        panels[kind][0].pack(fill="x")

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
        resize_id = getattr(self, "_content_resize_after_id", None)
        self._content_resize_after_id = None
        self._pending_content_width = None
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
        if resize_id is not None:
            try:
                self.after_cancel(resize_id)
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
