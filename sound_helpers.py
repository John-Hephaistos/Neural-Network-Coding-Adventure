from pydub import AudioSegment


def generate_track(bars, bpm, sounds_dir = 'sounds', filename='track'):
    '''
    Generate a .wav track from a list of bars for the 4 instruments
    Instruments are in the sound directory
    '''

    beat_duration = 60 / bpm

    # Load the audio samples
    bass = AudioSegment.from_wav(f"{sounds_dir}/Bass.wav")
    snare = AudioSegment.from_wav(f"{sounds_dir}/Snare.wav")
    open_hihat = AudioSegment.from_wav(f"{sounds_dir}/Open Hi-Hat.wav")
    closed_hihat = AudioSegment.from_wav(f"{sounds_dir}/Closed Hi-Hat.wav")

    # Initialise track and silent overlay
    silence = AudioSegment.silent(duration=int(beat_duration * 1000))
    track = AudioSegment.silent(duration=0)

    for beat in bars:
        # Initialize beat segment
        beat_segment = AudioSegment.silent(duration=0)

        # Overlay instrument sounds
        if beat[0]:
            beat_segment += bass
        if beat[1]:
            beat_segment += closed_hihat
        if beat[2]:
            beat_segment += snare
        if beat[3]:
            beat_segment += open_hihat

        # Add segment to the track
        track += beat_segment + silence

    # Export track with proper format
    track.export(filename + '.wav', format="wav")
