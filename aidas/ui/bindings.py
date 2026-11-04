"""Remove an owned native Tk binding without disturbing other callbacks."""

from types import SimpleNamespace


def bind_native_callback(widget, sequence, callback, *, widget_path=False, bindtag=None):
    """Bind a callback needing no event fields, or only the native widget path.

    Avoid Tkinter's full event conversion for handlers that ignore its fields.
    Return the owned command ID for the existing selective cleanup helper.
    """
    target = widget._w if bindtag is None else bindtag
    callback_id = widget.register(callback)
    fields = "%W" if widget_path else ""
    # Keep the space after the ID, including for zero-argument callbacks, so
    # older Tkinter cleanup recognizes the same prefix as native bind().
    script = f'if {{"[{callback_id} {fields}]" == "break"}} break\n'
    widget.tk.call("bind", target, sequence, "+" + script)
    return callback_id


def bind_widget_event(widget, sequence, callback, *, bindtag=None):
    """Retain an event.widget handler without converting unused Tk fields."""
    def dispatch(path):
        try:
            event_widget = widget.nametowidget(path)
        except KeyError:
            # Match Tkinter's handling of native widgets absent from its tree.
            event_widget = path
        return callback(SimpleNamespace(widget=event_widget))

    return bind_native_callback(widget, sequence, dispatch, widget_path=True, bindtag=bindtag)


def bind_all_widget_event(widget, sequence, callback):
    """Bind an owned widget-only event handler to Tk's global bind tag."""
    return bind_widget_event(widget, sequence, callback, bindtag="all")


def unbind_callback(widget, sequence, callback_id, *, bindtag=None):
    """Selectively unbind on older Python 3.11 releases as well as newer Tkinter.

    Use this with native ``tk.Misc.bind`` registrations. CTk widgets that
    redirect binding to an internal canvas need their own binding interface.
    ``bindtag='all'`` supports cleanup of an explicitly owned global handler.
    """
    if not callback_id:
        return
    target = widget._w if bindtag is None else bindtag
    unbind = getattr(widget, "_unbind", None)
    if unbind is not None:
        unbind(("bind", target, sequence), callback_id)
        return

    # Older Tkinter's unbind(sequence, funcid) clears the whole sequence.
    # Retain every Tcl script line except the registered Python callback.
    script = widget.tk.call("bind", target, sequence)
    prefix = f'if {{"[{callback_id} '
    retained = "\n".join(line for line in script.split("\n") if not line.startswith(prefix))
    widget.tk.call("bind", target, sequence, retained if retained.strip() else "")
    widget.deletecommand(callback_id)
