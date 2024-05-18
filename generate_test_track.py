from sound_helpers import generate_track
import numpy as np

def main():
    # Load in random 10 bars
    data = np.load('dataset.npy')
    bars_list = np.array(
        [data[i][0] for i in range(50)]
    )

    generate_track(
        bars_list=bars_list,
        bpm=140,
        sounds_dir='sounds'
    )

if __name__ == "__main__":
    main()