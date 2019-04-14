import math

from constants import MAX_FREQ_NOTE, MIN_FREQ_NOTE


def note_to_freq(note_idx: int) -> float:
    """
    Converts MIDI note index into its corresponding tone frequency.
    f_n = 2^(n/12)*440 | Reference: https://newt.phys.unsw.edu.au/jw/notes.html
    :param note_idx: MIDI note index.
    :return: Note frequency in Hz
    """
    return math.pow(2, (note_idx - 69) / 12) * 440


def freq_to_note(note_freq: float) -> int:
    """
    Converts note frequency into its corresponding MIDI tone index.
    f_n = 2^(n/12)*440 | Reference: https://newt.phys.unsw.edu.au/jw/notes.html
    :param note_freq: Note frequency in Hz.
    :return: MIDI note index.
    """
    if note_freq > 0:
        note_freq = max(MIN_FREQ_NOTE, min(note_freq, MAX_FREQ_NOTE))
        return int(12 * math.log2(note_freq / 440)) + 69
    else:
        return 0
