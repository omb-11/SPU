### SIGNAL PROCESSING UNIT USING SOME BASICS OF PYTORCH AND SOME PY. LIBS.
# HENCE THE SPU.
A Signal Processing Unit (SPU) or Digital Signal Processor (DSP) is a specialized microprocessor designed to measure, filter, and mathematically manipulate real-world continuous signals (like audio, radio waves, or video) at extremely high speeds.


# project flow:
- signal input
- filtering 
- FFT
- Feature Extraction
- ML Model
- Prediction
- Visualization
 
# FFT CONCEPT.
X(f) = integrate x(t)*e^(-j*2pi*ft).dt from (- ∞ to ∞)
# Let the model sturcture be:
SPU_DSP/
│
├── data/
├── notebooks/
├── models/
├── outputs/
├── app/
|-- documentation.md
├── README.md
└── requirements.txt

# TECH STACK 
numpy --maths/arrays
scipy -- DSP algorithm 
matplotlib -- visualization
librosa -- audio DSP 
sklearn -- ML
pandas -- datasets 
torch -- deep learning
sounddevice -- real-timeaudio

# requirements.
pip install numpy scipy matplotlib librosa scikit-learn pandas sounddevice jupyter




# Research and Dev for future things.. 
- creating a ui with drag and drop for the model 
- adding a audio mic to directly getting a spu/dpu
- adding a .wav file converter for future references. 