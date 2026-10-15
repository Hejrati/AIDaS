"""Processing steps, imported individually when requested."""

from importlib import import_module

__all__ = ["Step1Frame", "Step2Frame", "Step3Frame", "Step5Frame"]
_FRAME_MODULES = {
    "Step1Frame": "step1_resize_raw",
    "Step2Frame": "step2_annotate",
    "Step3Frame": "step3_flatten",
    "Step5Frame": "step5_compile",
}


def __getattr__(name):
    module_name = _FRAME_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    frame = getattr(import_module(f".{module_name}", __name__), name)
    globals()[name] = frame
    return frame
