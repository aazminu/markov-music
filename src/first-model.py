# Notes from: https://medium.com/data-science/markov-chain-for-music-generation-932ea8a88305

# Same as natural languages we may think about music as a sequence of notes. 
# But because I play guitar I’ll be operating with chords. If we take chords 
# sequence and learn its patternt we may find that certain chords may follow 
# particular chords more often, and other chords rarely follow that chords. 
# We will build our model to find and understand this pattern.

# General plan:
# 1. Take corpus of chords
# 2. Calculate probability distribution for chords to follow a particular chord
# 3. Define the first chord or make a random choice
# 4. Make a random choice of the next chord taking into an account probability 
# distribution
# 5. Repeat steps 4 for the generated chord
# 6. …
# 7. Stochastic music is awesome!

# 1. First make the chords into bigrams... why? Making chords into bigrams for 
# Markov music generation is a way to model the statistical likelihood of one 
# chord following another, creating more musically plausible progressions than 
# random selection

# 2. Choose a chord randomly, so need to calc prob of other chords to follow it

# 3. So, calc the freq of each unique bigram to appear in a sequence with that
# chosen first chord 

# 3. Normalize to get prob distribution of these unique bigrams 

# 4. Move to next state choose randomly from all the options of chords, but all
# the chords have a more or less likely chance to get chosen (more likely to 
# choose C cord over G7)
# Use np.random.choice(options, p=probabilities) for next choice of chord

import numpy as np
import csv
from collections import Counter
from music21 import *

# -----------------------------
# Load chords from CSV
# -----------------------------

def load_chords_from_csv(filepath):
    chords = []
    with open(filepath, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row:
                chords.append(row[0])
    return chords

chords = load_chords_from_csv("/Users/nehirozden/Desktop/markov-music/data/Liverpool_band_chord_sequence.csv")

# -----------------------------
# Convert chords to bigrams
# -----------------------------

def make_bigrams(chords):
    return [f"{chords[i]} {chords[i+1]}" for i in range(len(chords)-1)]

bigrams = make_bigrams(chords)

# -----------------------------
# Predict next chord
# -----------------------------

def predict_next_state(chord:str, data:list=bigrams):
    # list of bigrams starting with chord
    bigrams_with_current_chord = [bg for bg in data if bg.split(" ")[0] == chord]

    if not bigrams_with_current_chord:
        # fallback: random chord
        return np.random.choice([bg.split(" ")[1] for bg in data])

    count_appearance = dict(Counter(bigrams_with_current_chord))

    # convert to probabilities
    total = sum(count_appearance.values())
    for key in count_appearance:
        count_appearance[key] /= total

    options = [key.split(" ")[1] for key in count_appearance.keys()]
    probabilities = list(count_appearance.values())

    return np.random.choice(options, p=probabilities)

# -----------------------------
# Generate sequence
# -----------------------------

def generate_sequence(chord:str=None, data:list=bigrams, length:int=30):
    if chord is None:
        # choose a starting chord randomly
        chord = bigrams[0].split(" ")[0]

    sequence = [chord]

    for _ in range(length):
        next_chord = predict_next_state(chord, data)
        sequence.append(next_chord)
        chord = next_chord

    return sequence

# def print_sequence_numbered(seq):
#     for chord in enumerate(seq, start=1):
#         print(f"{chord}")

# Example:
# def print_seq(seq)
# print_sequence_numbered(generate_sequence(length=20))
def print_sequence(seq):
    print(" ".join(str(chord) for chord in seq))

seq = generate_sequence(length=20)
print_sequence(seq)

def chords_to_midi(chord_sequence, filename="markov_output.mid"):
    stream_score = stream.Stream()

    for ch in chord_sequence:
        try:
            # Try to parse chord symbol like "A7", "Dm", "C7"
            c = harmony.ChordSymbol(ch)
            c_chord = c.toChord()
        except Exception as e:
            print(f"Skipping chord {ch}: {e}")
            continue

        stream_score.append(c_chord)

    stream_score.write('midi', fp=filename)
    print(f"Saved MIDI to {filename}")


chords_to_midi(seq)