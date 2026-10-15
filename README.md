# Markov Chain Music Generator

## Data sets used:

-   https://github.com/jukedeck/nottingham-dataset
-   https://ifdo.ca/~seymour/nottingham/nottingham.html
-   https://www.kaggle.com/datasets/pranjalsriv/bach-chorales-2
-   https://github.com/music-x-lab/POP909-Dataset

## How to Run the Model

### Method 1: Command Line (Easiest)

```bash
# Basic usage - train and generate
python train_and_evaluate.py --dataset nottingham --order 1 --max_files 100

# With rhythm included
python train_and_evaluate.py --dataset nottingham --order 1 --include_rhythm --max_files 100

# Second-order model
python train_and_evaluate.py --dataset nottingham --order 2 --max_files 200

# Generate more samples
python train_and_evaluate.py --dataset nottingham --order 1 --generate_samples 10

# Use POP909 dataset
python train_and_evaluate.py --dataset pop909 --order 1 --max_files 50
```

**What this does:**

1. Loads MIDI files from the dataset
2. Splits into training (80%) and validation (20%)
3. Trains the Markov chain model
4. Evaluates on validation set (prints NLL)
5. Generates sample music pieces
6. Saves MIDI files to `output/` directory
7. Saves the trained model to `output/model_order1.pkl`

**Output files:**

-   `output/nottingham_order1_sample1.mid`
-   `output/nottingham_order1_sample2.mid`
-   ... (one file per sample)
-   `output/model_order1.pkl` (saved model)

---

## What's in the Generated MIDI Files?

### It Depends on `include_rhythm`!

The content of the generated MIDI files depends on whether you trained with
rhythm or not:

---

### Case 1: `include_rhythm=True` (Pitch + Rhythm)

**When you use this:**

```python
sequences = load_dataset(..., include_rhythm=True)
```

**What the states look like:**

```python
[(60, 1.0), (62, 0.5), (64, 1.0), (65, 0.5), ...]
  ↑    ↑      ↑    ↑
pitch dur   pitch dur
```

**What's in the MIDI file:**

-   **Notes with their actual durations**
-   Example: C (quarter note), D (eighth note), E (quarter note), F (eighth
    note)
-   The rhythm information from training is preserved

**MIDI file contains:**

-   Note pitches: C, D, E, F, ...
-   Note durations: quarter note, eighth note, quarter note, eighth note, ...
-   **Both pitch AND rhythm are in the file**

---

### Case 2: `include_rhythm=False` (Pitch Only)

**When you use this:**

```python
sequences = load_dataset(..., include_rhythm=False)
```

**What the states look like:**

```python
[60, 62, 64, 65, ...]
 ↑   ↑   ↑   ↑
pitch only (no duration)
```

**What's in the MIDI file:**

-   **Notes with DEFAULT duration (quarter note = 1.0 beats)**
-   Example: C (quarter note), D (quarter note), E (quarter note), F (quarter
    note)
-   All notes have the same duration because no rhythm was learned

**MIDI file contains:**

-   Note pitches: C, D, E, F, ...
-   Note durations: **All quarter notes (default)**
-   **Only pitch is in the file, rhythm is uniform**

---

## Visual Comparison

### With Rhythm (`include_rhythm=True`):

**Generated sequence:**

```python
[(60, 1.0), (62, 0.5), (64, 1.0), (65, 0.5)]
```

**MIDI file:**

```
C (quarter note) → D (eighth note) → E (quarter note) → F (eighth note)
```

**Musical notation:**

```
C    D E    F
♩    ♪ ♩    ♪
```

---

### Without Rhythm (`include_rhythm=False`):

**Generated sequence:**

```python
[60, 62, 64, 65]
```

**MIDI file:**

```
C (quarter note) → D (quarter note) → E (quarter note) → F (quarter note)
```

**Musical notation:**

```
C    D    E    F
♩    ♩    ♩    ♩
```

---

## How to Check What's in Your Generated MIDI

### Method 1: Listen to It

Just open the `.mid` file in any music player or DAW (GarageBand, Logic, etc.)

### Method 2: Inspect the Code

Look at `midi_generator.py` lines 63-84:

```python
for state in sequence:
    if isinstance(state, tuple):
        # State is (pitch, duration)
        pitch, duration = state
    else:
        # State is just a pitch (int)
        pitch = state
        duration = 1.0  # Default to quarter note
```

**This shows:**

-   If state is a tuple `(pitch, duration)` → uses that duration
-   If state is just an int `pitch` → uses default duration 1.0 (quarter note)

---

## Summary Table

| Setting                | States Look Like              | MIDI Contains                    | Rhythm Learned? |
| ---------------------- | ----------------------------- | -------------------------------- | --------------- |
| `include_rhythm=True`  | `[(60, 1.0), (62, 0.5), ...]` | Pitches + Durations              | ✅ Yes          |
| `include_rhythm=False` | `[60, 62, 64, ...]`           | Pitches only (all quarter notes) | ❌ No           |

---

## Quick Test

To see the difference, run:

```bash
# With rhythm
python train_and_evaluate.py --dataset nottingham --order 1 --include_rhythm --max_files 50 --generate_samples 1

# Without rhythm
python train_and_evaluate.py --dataset nottingham --order 1 --max_files 50 --generate_samples 1
```

Then listen to both MIDI files - you'll hear the difference!

---

## What About the Model Itself?

The model stores:

-   **Transition matrix**: What states follow what contexts
-   **State counts**: How often each context appears
-   **All states**: Set of all unique states seen

The generated sequence is just a list of states (either pitches or
pitch-duration pairs), which then gets converted to MIDI format.

---

## Example: Full Workflow

```python
# 1. Load with rhythm
sequences = load_dataset("data/nottingham_github/MIDI",
                        include_rhythm=True,
                        max_files=100)

# 2. Train
model = MarkovChain(order=1)
model.train(sequences[:80])  # First 80 for training

# 3. Generate (will be pitch+duration pairs)
generated = model.generate(length=50)
# generated = [(60, 1.0), (62, 0.5), (64, 1.0), ...]

# 4. Convert to MIDI (preserves durations)
sequence_to_midi(generated, "output.mid")
# MIDI file has: C (quarter), D (eighth), E (quarter), ...
```

---

**Bottom line**:

-   If you use `include_rhythm=True`, MIDI files contain **both pitches and
    rhythms**
-   If you use `include_rhythm=False`, MIDI files contain **pitches only** (all
    quarter notes)

The choice is yours when you load the dataset! Have fun.
