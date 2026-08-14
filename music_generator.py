import os
import glob
import numpy as np
from music21 import converter, instrument, note, chord, stream
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

DATA_DIR = "data"
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_notes():
    notes = []

    midi_files = glob.glob(os.path.join(DATA_DIR, "*.mid"))
    midi_files += glob.glob(os.path.join(DATA_DIR, "*.midi"))

    print(f"Found {len(midi_files)} MIDI files.")

    for file in midi_files:
        try:
            midi = converter.parse(file)

            print("Processing:", file)

            parts = instrument.partitionByInstrument(midi)

            if parts:
                notes_to_parse = parts.parts[0].recurse()
            else:
                notes_to_parse = midi.flat.notes

            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))

                elif isinstance(element, chord.Chord):
                    notes.append(
                        ".".join(str(n) for n in element.normalOrder)
                    )

        except Exception as e:
            print("Error:", file, e)

    return notes


def create_sequences(notes, sequence_length=100):
    pitchnames = sorted(set(notes))
    note_to_int = {note: number for number, note in enumerate(pitchnames)}

    network_input = []
    network_output = []

    for i in range(len(notes) - sequence_length):
        sequence = notes[i:i + sequence_length]
        target = notes[i + sequence_length]

        network_input.append(
            [note_to_int[n] for n in sequence]
        )
        network_output.append(note_to_int[target])

    n_patterns = len(network_input)

    if n_patterns == 0:
        raise ValueError(
            "Not enough notes. Add more MIDI files to the data folder."
        )

    network_input = np.reshape(
        network_input,
        (n_patterns, sequence_length, 1)
    )

    network_input = network_input / float(len(pitchnames))
    network_output = to_categorical(
        network_output,
        num_classes=len(pitchnames)
    )

    return network_input, network_output, pitchnames, note_to_int


def build_model(network_input, n_vocab):
    model = Sequential()

    model.add(
        LSTM(
            512,
            input_shape=(
                network_input.shape[1],
                network_input.shape[2]
            ),
            return_sequences=True
        )
    )

    model.add(Dropout(0.3))

    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))

    model.add(LSTM(512))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.3))

    model.add(Dense(n_vocab, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam"
    )

    return model


def generate_music(model, network_input, pitchnames, note_to_int):
    start = np.random.randint(
        0,
        len(network_input) - 1
    )

    pattern = network_input[start]

    int_to_note = {
        number: note_name
        for note_name, number in note_to_int.items()
    }

    prediction_output = []

    for _ in range(100):
        prediction_input = np.reshape(
            pattern,
            (1, len(pattern), 1)
        )

        prediction = model.predict(
            prediction_input,
            verbose=0
        )

        index = np.argmax(prediction)

        result = int_to_note[index]
        prediction_output.append(result)

        pattern = np.append(
            pattern[1:],
            index / float(len(pitchnames))
        )

    return prediction_output


def create_midi(prediction_output):
    offset = 0
    output_notes = []

    for pattern in prediction_output:

        if "." in pattern:
            notes_in_chord = pattern.split(".")

            chord_notes = []

            for current_note in notes_in_chord:
                new_note = note.Note(
                    int(current_note)
                )
                new_note.storedInstrument = (
                    instrument.Piano()
                )
                chord_notes.append(new_note)

            new_chord = chord.Chord(chord_notes)
            new_chord.offset = offset

            output_notes.append(new_chord)

        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()

            output_notes.append(new_note)

        offset += 0.5

    midi_stream = stream.Stream(output_notes)

    output_file = os.path.join(
        OUTPUT_DIR,
        "generated_music.mid"
    )

    midi_stream.write(
        "midi",
        fp=output_file
    )

    print("Generated MIDI saved to:", output_file)


def main():
    print("AI Music Generation using LSTM")

    notes = get_notes()

    print("Total notes:", len(notes))

    network_input, network_output, pitchnames, note_to_int = \
        create_sequences(notes)

    print("Training sequences:", len(network_input))
    print("Unique notes/chords:", len(pitchnames))

    model = build_model(
        network_input,
        len(pitchnames)
    )

    model.summary()

    model.fit(
        network_input,
        network_output,
        epochs=30,
        batch_size=64
    )

    model.save(
        os.path.join(OUTPUT_DIR, "music_model.keras")
    )

    prediction_output = generate_music(
        model,
        network_input,
        pitchnames,
        note_to_int
    )

    create_midi(prediction_output)


if __name__ == "__main__":
    main()
