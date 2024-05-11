import dataset_helpers as dh

def user_input():
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
    genre, vector_len = user_input()

    dfs = dh.process_files(genre=genre)

    # Plot an example beat dataframe
    dh.plot_instrument_playtimes(dfs[0])

    dh.save_dataset_to_file(dfs, vector_len)


if __name__ == '__main__':
    main()