from music21 import *
import glob2
import pickle
import numpy as np
from keras.layers import *
from keras.models import Sequential
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
def get():
    notes=[]
    for file in glob2.glob("musicgen/MIDI/*.mid"):
        midi=converter.parse(file)
        print("Parsing %s" % file)
        notes_to_parse=None
        parts=instrument.partitionByInstrument(midi)
        if parts: 
            notes_to_parse = parts.parts[0].recurse()
        else: 
            notes_to_parse = midi.flat.notes    
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
    with open('musicgen/Data/notes','wb+') as path:
        pickle.dump(notes, path)
    return notes
def sequence(notes,v):
    leng=100;
    v=float(v)
    name=sorted(set(item for item in notes))
    inote=dict((note, number) for number, note in enumerate(name))
    ninput=[]
    noutput=[]
    for i in range(0, len(notes) - leng, 1):
        seq_in = notes[i:i + leng]
        seq_out = notes[i + leng]
        ninput.append([inote[char] for char in seq_in])
        noutput.append(inote[seq_out])
    patterns=len(ninput)
    ninput=np.reshape(ninput,[patterns,leng,1])
    ninput=ninput/v
    noutput= np_utils.to_categorical(noutput)
    return(ninput,noutput)
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
    return model
def train(model,ninput,noutput):
    path="musicgen/weights.hdf5"
    cp=ModelCheckpoint(path,monitor='loss',verbose=0,save_best_only=True,mode='min')
    clist=[cp]
    model.fit(ninput,noutput,epochs=10,batch_size=128,callbacks=clist)
def networktrain():
    notes=get()
    v=len(set(notes))
    ninput,noutput=sequence(notes,v)
    model=nnetwork(ninput,v)
    train(model,ninput,noutput)
if __name__ == '__main__':
    networktrain()
