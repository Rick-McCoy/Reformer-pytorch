import warnings
import pretty_midi as pm
import numpy as np
from tqdm import tqdm

def midi_to_roll(path, output_length, augment=False) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        song = pm.PrettyMIDI(str(path))
    event_list = []
    for inst in song.instruments:
        for note in inst.notes:
            event_list.append((
                int(note.start * 2048),
                (128 if inst.is_drum else inst.program),
                note.pitch,
                note.velocity,
                int(note.end * 2048)
            ))
    event_list.sort()
    input_list = [[129, 128, 128, 128, 128, 128, 128]]
    current_time = 0
    pitch_augment = np.random.randint(-6, 6) if augment else 0
    velocity_augment = np.random.randint(-10, 11) if augment else 0
    time_augment = np.random.rand() + 0.5 if augment else 1
    for event in event_list:
        delta = min(int((event[0] - current_time) * time_augment), 16383)
        dur = min(int((event[4] - event[0]) * time_augment), 16383)
        instrument = event[1]
        pitch = min(max(event[2] + pitch_augment, 0), 127)
        velocity = min(max(event[3] + velocity_augment, 0), 127)
        input_list.append([
            instrument, pitch, velocity,
            dur // 128, dur % 128, delta // 128, delta % 128
        ])
        current_time = event[0]
    input_list.append([130, 129, 129, 129, 129, 129, 129])
    if len(input_list) < output_length:
        input_list.extend([[131, 130, 130, 130, 130, 130, 130]] * (output_length - len(input_list)))
    num = int(np.random.randint(0, len(input_list) - output_length + 1))
    output = np.array(input_list[num : num + output_length], dtype=np.int64)
    return output

def roll_to_midi(roll: np.array) -> pm.PrettyMIDI:
    midi = pm.PrettyMIDI(resolution=960)
    instruments = [pm.Instrument(i) for i in range(128)] \
                + [pm.Instrument(0, is_drum=True)]
    current_time = 0
    for event in roll:
        if event[0] == 130 or 129 in event[1:]:
            break
        if event[0] == 129 or 128 in event[1:]:
            continue
        if event[0] == 131 or 130 in event[1:]:
            continue
        instrument = event[0]
        pitch = event[1]
        velocity = event[2]
        dur = event[3] * 128 + event[4]
        delta = event[5] * 128 + event[6]
        instruments[instrument].notes.append(
            pm.Note(
                velocity=velocity,
                pitch=pitch,
                start=(current_time + delta) / 2048,
                end=(current_time + delta + dur) / 2048
            )
        )
        current_time += delta
    for inst in instruments:
        if inst.notes:
            midi.instruments.append(inst)
    return midi
