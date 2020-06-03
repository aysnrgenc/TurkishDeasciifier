# TurkishDeasciifier
In Turkish Natural Language Processing normalized text is crucial for well-designed models and high accuracy results. One of the main step in text normalization is diacritic resolution which converts non-normalized, asciified text to normalized and deasciified text. To resolve diacritic resolution task, recurrent neural network based model is prepared.

## Dependencies
To run this project, necessarcy libraries are:
* Dynet
* NTLK
* Numpy
* Codecs

## Setup
Please clone or download project into your local

## Data
Under the /data directory, you can find train and test data. 
*	data_input.txt and data_output.txt are for training
*	data_input_test.txt and data_output_test.txt are for testing
* Each train and test data consist sentences per line
*	If you want to use your own train or test file please upload your files under this directory with the specified file name

Under the /models directory, you can find pretrained RNN Diacritic Resolution model

char2int and int2char are prebuilted index arrays, please load these variables while using running project.

Load function is predefined for these variables. Function loads char2int.p and int2char.p files.

## How to Create Train Dataset
You can easily create your own dataset by obtaining large Turkish well-written text from web or other medias. 
Obtained text should be asciified by replacing Turkish diacritic letters with ascii form.
For example: 
- Replace all "ş" with "s"
- Replace all "ü" with "u"
- ...

You have two different text files:
  - Ascii text: the one you can use as input file 
  - Original text: the one you can use as output file

## Train 
```
python deasciifierRNN.py train
```

## Test
```
python deasciifierRNN.py test
```

## Article
You can review article: diacritic-restoration-recurrent.pdf
