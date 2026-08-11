"""Native Tk application menu used by AIDaS Classic interface mode.

The final pre-v3 AIDaS interface used the operating system's native menu bar.
This module restores that presentation without coupling it to workflow frames or
processing services.  The returned controller retains its Tk variables so the
checked Interface and Appearance entries can be updated without rebuilding the
menu. Appearance choices live in the Interface submenu and are disabled while
the Classic interface is active.
"""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable, Sequence


def canonical_menu_choice(value: object, choices: Sequence[object]) -> str:
    """Return the case-insensitive matching menu choice, or the first choice.

    Native menu radio groups require their variable value to exactly match one
    of the entry values.  Falling back to the first declared choice keeps a
    malformed or obsolete preference from leaving a group with no selection.
    """

    candidates = tuple(str(choice) for choice in choices)
    if not candidates:
        raise ValueError("A classic menu choice group cannot be empty")

    requested = str(value or "").strip().casefold()
    for candidate in candidates:
        if candidate.casefold() == requested:
            return candidate
    return candidates[0]


class ClassicApplicationMenu:
    """Controller for the native File/View/Help application menu.

    The controller is deliberately not a Tk widget subclass.  Keeping the
    native ``tk.Menu`` in :attr:`menubar` makes it easy for the application to
    attach the widget to its root while this object owns the radio variables and
    provides the same update surface as the modern menu bar.
    """

    def __init__(
        self,
        master: tk.Misc,
        *,
        interface_modes: Sequence[str],
        current_interface: str,
        set_interface_command: Callable[[str], None],
        appearance_modes: Sequence[str],
        current_appearance: str,
        set_appearance_command: Callable[[str], None],
        browse_sdb_command: Callable[[], None],
        settings_command: Callable[[], None],
        check_updates_command: Callable[[], None],
        about_command: Callable[[], None],
        exit_command: Callable[[], None] | None = None,
    ) -> None:
        self.master = master
        self.interface_modes = tuple(str(mode) for mode in interface_modes)
        self.appearance_modes = tuple(str(mode) for mode in appearance_modes)
        self._set_interface_command = set_interface_command
        self._set_appearance_command = set_appearance_command
        self._destroyed = False

        interface = canonical_menu_choice(current_interface, self.interface_modes)
        appearance = canonical_menu_choice(current_appearance, self.appearance_modes)
        self.interface_var = tk.StringVar(master=master, value=interface)
        self.appearance_var = tk.StringVar(master=master, value=appearance)

        self.menubar = tk.Menu(master, tearoff=0)
        self.file_menu = tk.Menu(self.menubar, tearoff=0)
        self.view_menu = tk.Menu(self.menubar, tearoff=0)
        self.interface_menu = tk.Menu(self.view_menu, tearoff=0)
        # Compatibility alias for callers that retained the old controller
        # attribute. Both groups now occupy one native submenu.
        self.appearance_menu = self.interface_menu
        self.help_menu = tk.Menu(self.menubar, tearoff=0)
        self._appearance_entry_indices: list[int] = []

        self.file_menu.add_command(
            label="Browse SDB Parent Directory",
            command=browse_sdb_command,
        )
        self.file_menu.add_command(label="Settings...", command=settings_command)
        self.file_menu.add_separator()
        self.file_menu.add_command(
            label="Exit",
            command=exit_command or master.winfo_toplevel().destroy,
            accelerator="Alt+F4",
        )
        self.menubar.add_cascade(label="File", menu=self.file_menu)

        for mode in self.interface_modes:
            self.interface_menu.add_radiobutton(
                label=mode,
                value=mode,
                variable=self.interface_var,
                command=self._select_interface,
            )

        self.interface_menu.add_separator()
        self.interface_menu.add_command(label="Appearance", state=tk.DISABLED)
        for mode in self.appearance_modes:
            self.interface_menu.add_radiobutton(
                label=mode,
                value=mode,
                variable=self.appearance_var,
                command=self._select_appearance,
            )
            entry_index = self.interface_menu.index("end")
            if entry_index is not None:
                self._appearance_entry_indices.append(int(entry_index))
        self._sync_appearance_entry_state()

        self.view_menu.add_cascade(label="Interface", menu=self.interface_menu)
        self.menubar.add_cascade(label="View", menu=self.view_menu)

        self.help_menu.add_command(
            label="Check for Updates...",
            command=check_updates_command,
        )
        self.help_menu.add_separator()
        self.help_menu.add_command(label="About", command=about_command)
        self.menubar.add_cascade(label="Help", menu=self.help_menu)

        master.configure(menu=self.menubar)

    @property
    def current_interface(self) -> str:
        """Return the interface mode currently checked in the native menu."""

        return canonical_menu_choice(self.interface_var.get(), self.interface_modes)

    @property
    def current_appearance(self) -> str:
        """Return the appearance mode currently checked in the native menu."""

        return canonical_menu_choice(self.appearance_var.get(), self.appearance_modes)

    @property
    def appearance_enabled(self) -> bool:
        """Return whether Modern appearance choices can currently be invoked."""

        return self.current_interface.casefold() == "modern"

    def set_interface(self, mode: object) -> str:
        """Check one Interface entry without invoking its application callback."""

        selected = canonical_menu_choice(mode, self.interface_modes)
        self.interface_var.set(selected)
        self._sync_appearance_entry_state()
        return selected

    def set_appearance(self, mode: object) -> str:
        """Check one Appearance entry without invoking its application callback."""

        selected = canonical_menu_choice(mode, self.appearance_modes)
        self.appearance_var.set(selected)
        return selected

    def _select_interface(self) -> None:
        selected = self.set_interface(self.interface_var.get())
        self._set_interface_command(selected)

    def _select_appearance(self) -> None:
        if not self.appearance_enabled:
            return
        selected = self.set_appearance(self.appearance_var.get())
        self._set_appearance_command(selected)

    def _sync_appearance_entry_state(self) -> None:
        """Enable Modern appearance entries only while Modern is selected."""

        state = tk.NORMAL if self.appearance_enabled else tk.DISABLED
        for entry_index in self._appearance_entry_indices:
            self.interface_menu.entryconfigure(entry_index, state=state)

    def destroy(self) -> None:
        """Detach and destroy the native menu once; repeated calls are safe."""

        if self._destroyed:
            return
        self._destroyed = True
        try:
            self.master.configure(menu="")
        except tk.TclError:
            pass
        try:
            self.menubar.destroy()
        except tk.TclError:
            pass


def build_classic_application_menu(
    master: tk.Misc,
    **options,
) -> ClassicApplicationMenu:
    """Build, attach, and return the native Classic application menu."""

    return ClassicApplicationMenu(master, **options)


__all__ = [
    "ClassicApplicationMenu",
    "build_classic_application_menu",
    "canonical_menu_choice",
]
