from pydub import AudioSegment
from pydub.playback import play
from midiutil import MIDIFile
import numpy as np

def generate_track(bars, bpm, sounds_dir = "sounds"):
    beat_duration = 60 / bpm

    # Load the audio samples
    bass = AudioSegment.from_wav(f"{sounds_dir}/Bass.wav")
    snare = AudioSegment.from_wav(f"{sounds_dir}/Snare.wav")
    open_hihat = AudioSegment.from_wav(f"{sounds_dir}/Open Hi-Hat.wav")
    closed_hihat = AudioSegment.from_wav(f"{sounds_dir}/Closed Hi-Hat.wav")

    silence = AudioSegment.silent(duration=beat_duration * 1000)
    track = AudioSegment.silent(duration=0)

    for beat in bars:
        beat_segment = silence
        if beat[0]:
            beat_segment = beat_segment.overlay(bass)
        if beat[1]:
            beat_segment = beat_segment.overlay(closed_hihat)
        if beat[2]:
            beat_segment = beat_segment.overlay(snare)
        if beat[3]:
            beat_segment = beat_segment.overlay(open_hihat)
        track += beat_segment

    track.export("track.wav", format="wav")