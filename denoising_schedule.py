"""Explicit LiveAct denoising schedules for quality/speed comparisons."""


def denoising_schedule(steps: int) -> tuple[tuple[float, ...], tuple[bool, ...]]:
    """Return timestep boundaries and per-forward audio-skip flags."""
    if steps == 3:
        return (1000.0, 937.5, 833.33333333, 0.0), (True, False, False)
    if steps == 2:
        return (1000.0, 833.33333333, 0.0), (True, False)
    raise ValueError("LiveAct currently supports 2 or 3 denoising steps")
