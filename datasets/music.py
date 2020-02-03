import warnings
import pretty_midi as pm
import numpy as np

def midi_to_roll(path, output_length) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        song = pm.PrettyMIDI(str(path))
    event_list = []
    for inst in song.instruments:
        for note in inst.notes:
            event_list.append((
                int(note.start * 2048),
                note.pitch + 128 if inst.is_drum else inst.program,
                note.pitch + 256,
                note.velocity + 384,
                int(note.end * 2048)
            ))
    event_list.sort()
    input_list = [1024] * 7
    current_time = 0
    for event in event_list:
        delta = min(event[0] - current_time, 16383)
        dur = min(event[4] - event[0], 16383)
        input_list.extend([
            event[1], event[2], event[3],
            dur % 128 + 512, dur // 128 + 640,
            delta % 128 + 768, delta // 128 + 896
        ])
        current_time = event[0]
    input_list += [1024] * 7
    if len(input_list) < output_length:
        input_list.extend([1025] * (output_length - len(input_list)))
    num = int(np.random.randint(0, len(input_list) - output_length + 1))
    output = np.array(input_list[num : num + output_length], dtype=np.int64)
    return output

def roll_to_midi(roll) -> pm.PrettyMIDI:
    midi = pm.PrettyMIDI(resolution=960)
    instruments = [pm.Instrument(i) for i in range(128)] \
                + [pm.Instrument(0, is_drum=True)]
    current_time = 0
    while roll[0] > 255:
        roll = roll[1:]
    roll = [roll[i * 7 : (i + 1) * 7] for i in range(0, len(roll) // 7)]
    for step, event in enumerate(roll):
        if event[0] == 1025:
            break
        if event[0] == 1024:
            continue
        try:
            assert event[0] < 256,\
                "Failed at step {}, event 0: {} is not between 0 and 256".format(step, event[i])
            for i in range(1, 7):
                assert (i + 1) * 128 <= event[i] and event[i] < (i + 2) * 128,\
                    "Failed at step {}, event {}: {} is not between {} and {}".format(
                        step, i, event[i], (i + 1) * 128, (i + 2) * 128
                    )
        except AssertionError:
            if step > 10:
                print('Generation stopped at step {}'.format(step))
                break
            raise
        instrument = min(128, event[0])
        dur = (event[3] - 512) + (event[4] - 640) * 128
        delta = (event[5] - 768) + (event[6] - 896) * 128
        instruments[instrument].notes.append(
            pm.Note(
                velocity=event[2] - 384,
                pitch=event[1] - 256,
                start=(current_time + delta) / 2048,
                end=(current_time + delta + dur) / 2048
            )
        )
        current_time += delta
    for inst in instruments:
        if inst.notes:
            midi.instruments.append(inst)
    return midi
