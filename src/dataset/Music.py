from typing import List, Tuple


class NoteData(object):
    def __init__(self, data: Tuple[int, int, int, int]):
        """
        Creates Note Data object which has Note timings, Encoding and Velocity.
        :param data: (Note_start, Note_end, Note_code, Velocity)
        """
        self.note_start, self.note_end, self.note, self.velocity = data

    @property
    def norm_vel(self) -> float:
        return self.velocity / 127.0

    @property
    def enc_note(self) -> str:
        raise NotImplementedError

    def is_playing(self, time: int) -> bool:
        return self.note_start <= time <= self.note_end


class Track:
    def __init__(self, data: List[NoteData]):
        self.notes_data = data

    def get_max_time(self):
        return self.notes_data[-1].note_end

    def get_note_data(self, idx: int) -> NoteData:
        return self.notes_data[idx]

class Song:
    def __init__(self, data: List[Track]):
        self.tracks = data
        self.number_tracks = len(data)

    @property
    def max_time(self):
        return max(map(lambda t: t.get_max_time(), self.tracks))

    def get_track(self, idx: int) -> Track:
        return self.tracks[idx]