# Neural-Network-Coding-Adventure
Project for Neural Network Course

## Setup
- Create a venv, and then activate it (do not put this in the git repo)
- Run:
```bash
pip install -r requirements.txt
```

## Generate dataset
- Run:
```bash
python3 prepare_dataset.py
```
- Pick genre, should be "funk" for our dataset
- Pick the length of the sample (numbre of notes in a sample), this probably should be 4 or 8 to start with
- You can see the plot of one of the MIDI files in beats_plot.png, and dataset is saved as dataset.npy (to read it use numpy.load)
