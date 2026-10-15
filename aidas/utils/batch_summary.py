"""Consistent, compact folder-selection statistics for batch steps."""


def format_batch_summary(*, scanned, ready, processed, skipped, unavailable, selected=0):
    """Format scan totals and the current number of runnable selections."""
    counts = (
        ("Scanned", scanned),
        ("Ready", ready),
        ("Processed", processed),
        ("Skipped", skipped),
        ("Unavailable", unavailable),
        ("Selected", selected),
    )
    return " · ".join(f"{label}: {int(count)}" for label, count in counts)
