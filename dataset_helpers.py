import os
import pretty_midi as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import CONSTANTS as c


def process_files(genre):
    dfs = []

    drummers = [d for d in os.listdir(c.DATASET_PATH) if 'drummer' in d]
    for drummer in drummers:
        drummer_path = os.path.join(c.DATASET_PATH, drummer)
        sessions = [s for s in os.listdir(drummer_path) if 'session' in s and 'eval' not in s]
        for session in sessions:
            session_path = os.path.join(drummer_path, session)
            midi_files = [f for f in os.listdir(session_path) if 'beat' in f and '4-4' in f]
            for file_name in midi_files:
                file_info = parse_file_info(file_name)
                file_path = os.path.join(session_path, file_name)

                if (
                    genre in file_info['genre'] and
                    file_info['notation'] == '4-4'
                ):
                    dfs.append(
                        process_session(file_path, file_info)
                    )

    return dfs


def parse_file_info(file_name):
    file_info = file_name.split("_")

    return {
        'genre': file_info[1],
        'tempo': int(file_info[2]),
        'notation': file_info[-1].split('.')[0] # Remove extension from notation
    }


def process_session(file_path, file_info):
    midi_stream = pm.PrettyMIDI(file_path)

    pitches = []
    start_times = []
    for instrument in midi_stream.instruments:
        for note in instrument.notes:
            pitches.append(note.pitch)
            start_times.append(note.start)

    df = pd.DataFrame(
        {
            'pitch': pitches,
            'time': start_times
        }
    )
    df['tempo'] = file_info['tempo']

    df = df.join(c.pitch_name_map.set_index('Pitch')['Paper Mapping'].rename('name'), on='pitch')

    df = preprocess_df(df)
    df = convert_to_beat_dataframe(df)
    df = standardize_columns(df)

    return df


def preprocess_df(df_original):
    df = df_original.copy()

    sequence_start_time = df['time'].iloc[0]
    if sequence_start_time != 0:
        df['time'] = df['time'] - sequence_start_time

    return df


def convert_to_beat_dataframe(df):
    # Determine the total duration of the piece
    total_duration = df['time'].max()
    spb = 60 / df['tempo'].iloc[0]  # Seconds per beat

    # Create a list of beat numbers corresponding to each 1/8th note
    num_beats = int(total_duration / spb)
    beat_numbers = list(range(num_beats + 1))

    # Initialize dictionary to store instrument play times
    instrument_play_times = {instrument: [0] * len(beat_numbers) for instrument in df['name'].unique()}

    # Iterate through each instrument entry and mark 1 for the corresponding beats
    for _, row in df.iterrows():
        beat_number = int(row['time'] / spb)
        instrument_play_times[row['name']][beat_number] = 1

    # Create a new DataFrame
    beat_df = pd.DataFrame(instrument_play_times, index=beat_numbers)

    return beat_df


def standardize_columns(df_original):
    df = df_original.copy()
    for column in c.COLUMNS:
        if column not in df.columns:
            df[column] = 0
    return df[c.COLUMNS]


def pad_dataframe(df, vector_len):
    # Pad DataFrame with NaN values to ensure its length is divisible by 4
    remainder = len(df) % vector_len
    if remainder != 0:
        pad_length = vector_len - remainder
        padding = pd.DataFrame(0, index=range(pad_length), columns=df.columns)
        df = pd.concat([df, padding])
    
    return df


def generate_io_pairs_from_beat_df(df, vector_len):
    io_pairs = []
    slices = [
        df.iloc[i:i+vector_len].values 
        for i in range(0, len(df), vector_len)
    ]

    for i in range(len(slices)-1):
        io_pairs.append(
            [
                np.array(slices[i]),
                np.array(slices[i+1])
            ]
        )

    return np.array(io_pairs)


def save_dataset_to_file(dfs, vector_len):
    io_pairs_list = []
    for df in dfs:
        # Skip dataframe if there is not enough data to make an i/o pair
        if len(df) >= 2 * vector_len:
            padded_df = pad_dataframe(df, vector_len)
            io_pairs_list.append(
                generate_io_pairs_from_beat_df(padded_df, vector_len)
            )

    # Concatenate the list of io_pairs along the first axis, and save to file
    concatenated_io_pairs = np.concatenate(io_pairs_list, axis=0)
    np.save('dataset.npy', concatenated_io_pairs)


def plot_instrument_playtimes(beat_df):
    fig, ax = plt.subplots(figsize=(10,6))

    # Plot drum playtimes
    for drum in c.COLUMNS:
        ax.scatter(
            beat_df.index,
            [drum] * len(beat_df),
            c=beat_df[drum],
            cmap='Blues',
            marker='o',
            label=drum
        )
    
    ax.set_yticks(range(len(beat_df.columns)))
    ax.set_yticklabels(beat_df.columns)
    ax.set_xlabel('Beat Number')
    ax.set_ylabel('Drum')
    ax.set_title('Drum Beat Playtimes')
    ax.grid(True)
    ax.legend()

    fig.savefig('beats_plot.png')