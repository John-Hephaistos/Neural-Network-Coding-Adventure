from sound_helpers import generate_track
import numpy as np

def main():
    # Load in random 10 bars
    data = np.load('test_pred.npy')
    print(data)

    generate_track(
        bars=data,
        bpm=140,
        sounds_dir='sounds'
    )

if __name__ == "__main__":
    main()
