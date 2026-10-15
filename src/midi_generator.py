"""
MIDI Generator Module

This module converts generated state sequences back into MIDI files that can be
played or analyzed. It handles both pitch-only and pitch+rhythm sequences.
"""

from music21 import stream, note, tempo, meter
from typing import List, Tuple, Union


def sequence_to_midi(sequence: List[Union[int, Tuple]], 
                     output_path: str = "generated_music.mid",
                     tempo_bpm: int = 120,
                     time_signature: str = "4/4"):
    """
    Convert a generated sequence of states into a MIDI file.
    
    Parameters:
    -----------
    sequence : List[Union[int, Tuple]]
        Generated sequence. Can be:
        - List of pitches (int): [60, 62, 64, ...]
        - List of (pitch, duration) tuples: [(60, 1.0), (62, 0.5), ...]
    output_path : str
        Path where the MIDI file will be saved
    tempo_bpm : int
        Tempo in beats per minute
    time_signature : str
        Time signature (e.g., "4/4", "3/4")
    
    Returns:
    --------
        Saves a MIDI file to output_path
    """
    score = stream.Stream()
    
    # Set tempo
    score.append(tempo.MetronomeMark(number=tempo_bpm))
    
    # Set time signature
    score.append(meter.TimeSignature(time_signature))
    
    current_time = 0.0
    
    # Process each state in the sequence
    for state in sequence:
        if isinstance(state, tuple):
            pitch, duration = state
        else:
            pitch = state
            duration = 1.0  # Default to quarter note
        
        # Create a note
        # MIDI pitch 60 = C4 (middle C)
        if isinstance(pitch, (int, float)):
            pitch = int(pitch)
            if 0 <= pitch <= 127:
                n = note.Note(pitch)
                n.duration.quarterLength = duration
                n.offset = current_time
                score.append(n)
                
                current_time += duration
        else:
            # Skip invalid pitches
            continue
    
    # Write to MIDI file
    score.write('midi', fp=output_path)
    print(f"Generated MIDI saved to {output_path}")

