"""Explicit LiveAct denoising schedules for quality/speed comparisons."""


def denoising_schedule(steps: int, audio_first_step: bool = False) -> tuple[tuple[float, ...], tuple[bool, ...]]:
    """Return timestep boundaries and per-forward audio-skip flags."""
    if audio_first_step and steps != 2:
        raise ValueError("audio_first_step is only supported with 2 denoising steps")
    if steps == 3:
        return (1000.0, 937.5, 833.33333333, 0.0), (True, False, False)
    if steps == 2:
        return (1000.0, 833.33333333, 0.0), (not audio_first_step, False)
    raise ValueError("LiveAct currently supports 2 or 3 denoising steps")
