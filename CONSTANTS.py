import pandas as pd

data = {
    'Pitch': [36, 38, 40, 37, 48, 50, 45, 47, 43, 58, 46, 26, 42, 22, 44, 49, 55, 57, 52, 51, 59, 53],
    'Roland Mapping': ['Kick', 'Snare (Head)', 'Snare (Rim)', 'Snare X-Stick', 'Tom 1', 'Tom 1 (Rim)', 'Tom 2', 'Tom 2 (Rim)', 'Tom 3 (Head)', 'Tom 3 (Rim)', 'HH Open (Bow)', 'HH Open (Edge)', 'HH Closed (Bow)', 'HH Closed (Edge)', 'HH Pedal', 'Crash 1 (Bow)', 'Crash 1 (Edge)', 'Crash 2 (Bow)', 'Crash 2 (Edge)', 'Ride (Bow)', 'Ride (Edge)', 'Ride (Bell)'],
    'GM Mapping': ['Bass Drum 1', 'Acoustic Snare', 'Electric Snare', 'Side Stick', 'Hi-Mid Tom', 'High Tom', 'Low Tom', 'Low-Mid Tom', 'High Floor Tom', 'Vibraslap', 'Open Hi-Hat', 'N/A', 'Closed Hi-Hat', 'N/A', 'Pedal Hi-Hat', 'Crash Cymbal 1', 'Splash Cymbal', 'Crash Cymbal 2', 'Chinese Cymbal', 'Ride Cymbal 1', 'Ride Cymbal 2', 'Ride Bell'],
    'Paper Mapping': ['Bass (36)', 'Snare (38)', 'Snare (38)', 'Snare (38)', 'High Tom (50)', 'High Tom (50)', 'Low-Mid Tom (47)', 'Low-Mid Tom (47)', 'High Floor Tom (43)', 'High Floor Tom (43)', 'Open Hi-Hat (46)', 'Open Hi-Hat (46)', 'Closed Hi-Hat (42)', 'Closed Hi-Hat (42)', 'Closed Hi-Hat (42)', 'Crash Cymbal (49)', 'Crash Cymbal (49)', 'Crash Cymbal (49)', 'Crash Cymbal (49)', 'Ride Cymbal (51)', 'Ride Cymbal (51)', 'Ride Cymbal (51)'],
    'Frequency': [88067, 102787, 22262, 9696, 13145, 1561, 3935, 1322, 11260, 1003, 3905, 10243, 31691, 34764, 52343, 720, 5567, 1832, 1046, 43847, 2220, 5567]
}

# Create DataFrame
pitch_name_map = pd.DataFrame(data)

COLUMNS = ['Bass (36)', 'Closed Hi-Hat (42)', 'Snare (38)', 'Open Hi-Hat (46)']

DATASET_PATH = './groove'