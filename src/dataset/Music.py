from typing import List


class NoteData(object):
    def __init__(self, start: int, end: int, note: int, velocity: int):
        self.note_start = start
        self.note_end = end
        self.note = note
        self.velocity = velocity

    def __str__(self):
        return f'NoteData({self.note_start}, {self.note_end}, {self.note}, {self.velocity})'

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

    @property
    def len_track(self):
        return len(self.notes_data)


class Song:
    def __init__(self, data: List[Track]):
        self.tracks = data
        self.number_tracks = len(data)

    @property
    def max_time(self):
        return max(map(lambda t: t.get_max_time(), self.tracks))

    def get_track(self, idx: int) -> Track:
        return self.tracks[idx]
