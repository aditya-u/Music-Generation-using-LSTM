import numpy as np
import pickle
from music21 import *
from keras.models import Sequential
from keras.layers import *
def Gen():
    with open('musicgen/Data/notes', 'rb') as path:
        notes = pickle.load(path)
    pitch = sorted(set(item for item in notes))
    v = len(set(notes))
    ninput, normalized = seq(notes, pitch, v)
    model = nnetwork(normalized, v)
    op = genn(model, ninput, pitch, v)
    create_midi(op)
def seq(notes,pitch,v):
    inote = dict((note, number) for number, note in enumerate(pitch))
    leng = 100
    ninput = []
    op = []
    for i in range(0, len(notes) - leng, 1):
        seq_in = notes[i:i + leng]
        seq_out = notes[i + leng]
        ninput.append([inote[char] for char in seq_in])
        op.append(inote[seq_out])
    pattern=len(ninput)
    normalized=np.reshape(ninput, (pattern, leng, 1))
    normalized=normalized/float(v)
    return(ninput,normalized)
def nnetwork(ninput,v):
    model = Sequential()
    model.add(LSTM(256,input_shape=(ninput.shape[1], ninput.shape[2]),return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(256))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(v))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    model.load_weights('musicgen/weights.hdf5')
    return model
def genn(model,ninput,pitch,v):
    start=np.random.randint(0,len(ninput)-1)
    inote=dict((n1,n2)for n1,n2 in enumerate (pitch))
    pattern=ninput[start]
    op=[]
    for note_index in range(500):
        ip = np.reshape(pattern, (1, len(pattern), 1))
        ip = ip / float(v)

        prediction = model.predict(ip, verbose=0)

        index = np.argmax(prediction)
        result = inote[index]
        op.append(result)

        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return op
def create_midi(op):
    offset = 0
    output_notes = []
    for pattern in op:
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        offset += 0.5

    midi_stream = stream.Stream(output_notes)

    midi_stream.write('midi', fp='output.mid')

if __name__ == '__main__':
    Gen()
