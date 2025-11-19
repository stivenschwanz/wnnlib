# Weightless Neural Networks library (wnnlib)

## Setup

Clone the repository:
```bash
git clone https://github.com/stivenschwanz/wnnlib
cd wnnlib
```

Create the virtual environment:
```bash
pip3 install virtualenv
python3 -m venv .venv
```

Activate the virtual environment:
```bash
.venv\Scripts\activate.bat # (on Windows)
source .venv/bin/activate # (on Linux)
```

Install required packages:
```bash
pip3 install -r requirements.txt
```

## Run experiments

Activate the virtual environment:
```bash
.venv\Scripts\activate.bat # (on Windows)
source .venv/bin/activate # (on Linux)
```

Toy anomaly detection problems:
```bash
python3 -m unittest wnnlib/algos/NPCLAD.py
```

## Run unit tests

Test VGRAM node:
```bash
python3 -m unittest wnnlib/vgram/VGRAMNode.py
```

Test VGRAM array:
```bash
python3 -m unittest wnnlib/vgram/VGRAMArray.py
```

Test fixed scalar codec:
```bash
python3 -m unittest wnnlib/codecs/FixedScalarCodec.py
```

Test adaptive scalar codec:
```bash
python3 -m unittest wnnlib/codecs/AdaptiveScalarCodec.py
```

Test flex scalar codec:
```bash
python3 -m unittest wnnlib/codecs/FlexScalarCodec.py
```

Test KD-tree vector codec:
```bash
python3 -m unittest wnnlib/codecs/KDTree.py
```

Test binary utilities:
```bash
python3 -m unittest wnnlib/utils/BitUtils.py
```