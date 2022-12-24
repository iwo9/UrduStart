import DataProcessing

# Creating CommonVoice Class

import os
import csv
from pathlib import Path
from typing import List, Dict, Tuple, Union
from pydub import AudioSegment


import torchaudio
from torch import Tensor
from torch.utils.data import Dataset
import soundfile


#basic path will only be path to drive, folder_audio needs to be complete path to audio folder inside that
def load_commonvoice_item(line: List[str], header: List[str], folder_audio: str, ext_audio: str) ->Tuple[Tensor, int, Dict[str, str]]:

  #Need to change this for validation set - doesn't have a header. - made change, I think this will work
  #if header[0] != path or not header[0].endswith(ext_audio):
    #htemp = header[0]
    #raise ValueError(f"expect `header[0]` to be 'path', but got {htemp}")
  fileid = line[0]


  #path and folder_audio are just the complete path to the audio folder - fileid is the individual files 
  #within it that we get from csv file
  #right now changed the folder_audio to have the complete path to audio folder so don't need path
  filename = os.path.join(folder_audio, fileid)
  if not filename.endswith(ext_audio):
        filename += ext_audio
  #waveform, sample_rate = torchaudio.load(filename)
  audio_test = r"C:\Users\marya\Music\iTunes\The Lumineers - Cleopatra (Deluxe) (2016) - WEB FLAC\01 - Sleep On The Floor.flac"
  np_audio, sample_rate =  soundfile.read(audio_test)

  dic = dict(zip(header, line))

  # waveform, sample_rate, 
  return np_audio, sample_rate, dic


#right now need to focus on changing all paths - second order or business getting pretrained model imported
class COMMONVOICE(Dataset):
  _ext_txt = ".txt"
  _ext_audio = ".mp3"
  
  _folder_audio = r"C:Users\marya\Documents\New folder\clips"
  _train = r"C:/Users/marya/Documents/cv-corpus-11.0-2022-09-21/ur/TVSets/training_set.csv"
  _validate = r"C:/Users/marya/Documents/cv-corpus-11.0-2022-09-21/ur/TVSets/validation_set.csv"
  
  _ndownaudio_path = "https://mozilla-common-voice-datasets.s3.dualstack.us-west-2.amazonaws.com/cv-corpus-11.0-2022-09-21/cv-corpus-11.0-2022-09-21-ur.tar.gz"

  #just need to call COMMONVOICE("train" or "validate")
  #e.g. COMMONVOICE("train") or COMMONVOICE("validate")
  def __init__(self, fcsv: str = "train") -> None:

    #csv should be the path to the sheets after the drive - will be two, one for training, one for validation
    if fcsv == "train":
      self._csv = self._train
    elif fcsv == "validate":
      self._csv = self._validate

    with open(self._csv, "r", encoding="utf-8") as csv_:
            walker = csv.reader(csv_)
            #saving two things - just the header in _header, and list of entire sheet in _walker
            self._header = next(walker)
            self._walker = list(walker)

  #getitem returns waveform, samplerate and dictionary - path and transcript
  def __getitem__(self, n: int) -> Tuple[Tensor, int, Dict[str, str]]:
    
      #getting nth row from list
      line = self._walker[n]
      
      #row of excel sheet, header of sheet, path to folder with audio, name of audio folder inside, extension of every audio file
      #folder audio needs to be more specific than just "clips" - it's inside "/content/drive/Othercomputers/My PC/clips"
      return load_commonvoice_item(line, self._header, self._folder_audio, self._ext_audio)

  def __len__(self):
        return len(self._walker)



tobj = COMMONVOICE("train")
print("Length of tobj: " + str(len(tobj)))
test_arr, sr, dump = tobj.__getitem__(4)
line_4 = dump["sentence"]
print(line_4)
print(test_arr)

print(soundfile.__libsndfile_version__)