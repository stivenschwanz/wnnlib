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

Add the source folder permanently to the Python path:
```bash
echo "$(pwd)/src" > `echo $VIRTUAL_ENV/lib/python*/site-packages/`src.pth
```

Activate the virtual environment:
```bash
source .venv/bin/activate
```

Install required packages:
```bash
pip3 install -r requirements.txt
```

## Run experiments

Activate the virtual environment:
```bash
source .venv/bin/activate
```

Toy anomaly detection problems:
```bash
python3 -m unittest src/wnnlib/algos/NPCLAD.py
```

## Run unit tests

Test VGRAM node:
```bash
python3 -m unittest src/wnnlib/vgram/VGRAMNode.py
```

Test VGRAM array:
```bash
python3 -m unittest src/wnnlib/vgram/VGRAMArray.py
```

Test fixed scalar codec:
```bash
python3 -m unittest src/wnnlib/codecs/FixedScalarCodec.py
```

Test adaptive scalar codec:
```bash
python3 -m unittest src/wnnlib/codecs/AdaptiveScalarCodec.py
```

Test flex scalar codec:
```bash
python3 -m unittest src/wnnlib/codecs/FlexScalarCodec.py
```

Test KD-tree vector codec:
```bash
python3 -m unittest src/wnnlib/codecs/KDTree.py
```

Test binary utilities:
```bash
python3 -m unittest src/wnnlib/utils/BitUtils.py
```

## Generate distribution archives

Make sure you have the latest version of PyPA’s build installed:
```bash
python3 -m pip install --upgrade build
```

Build the distribution archives:
```bash
python3 -m build
```

Install twine to upload the distribution packages:
```bash
python3 -m pip install --upgrade twine
```

Upload the distribution packages:
```bash
python3 -m twine upload --repository testpypi dist/*
```

