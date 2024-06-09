import dataset_helpers as dh
import random

def user_input():
    '''
    Handle user input for vector lengths and music genre
    '''

    genre = input('Pick a genre: ').lower()
    if not isinstance(genre, str):
        raise ValueError('Genre has to be a string!')

    vector_len = int(input('Pick the number of notes per sample: '))
    if not isinstance(vector_len, int):
        raise ValueError('Vector length has to be an integer!')
    if vector_len <= 0:
        raise ValueError('Vector length has to be a positive integer!')

    return genre, vector_len


def main():
    '''
    Prepare dataset based on user input choices
    '''

    genre, vector_len = user_input()

    dfs = dh.process_files(genre=genre)

    # Plot an example beat dataframe
    dh.plot_instrument_playtimes(dfs[0])

    # Split the data for lockbox
    train_dataset_length = int(len(dfs) * 9/10)

    # Shuffle the list
    random.shuffle(dfs)

    # Save train dataset
    dh.save_dataset_to_file(dfs[train_dataset_length:], vector_len , 'train_dataset')

    # Save train dataset
    dh.save_dataset_to_file(dfs[:train_dataset_length], vector_len , 'val_dataset')


if __name__ == '__main__':
    main()
