# speech-synthesis
Speech Synthesis algorithm using Unit-Selection approach. Designed to work with PMA biosignals.
Project includes:

~UNIT_SELECTION: Algorithm for speech synthesis. Includes the creation of a database in the training stage as well as MFCC parameters prediction from PMA biosignals. Also performs k-fold validation for robustness.

~LINEAR_REGRESSION: Base method for speech synthesis that uses a simple Linear Regression approach for predicting MFCC parameters from PMA signals.

~WORLD: VoCoder WORLD. Its main purpose is to obtain Mel-Cepstral coefficients from audio signals and synthesise audio from those Mel-Cepstral coefficients.

~MEDIDAS: Script used to perform measurements from the results obtained from predicted MFCC parameters as well as synthesized voice from WORLD.

