# Weightless Neural Networks library (wnnlib)

## Setup

Clone the repository:
```bash
git clone https://github.com/stivenschwanz/wnnlib
cd wnnlib
```

Install required system packages:
```bash
sudo apt install python3-pip python3.12-venv
```

Create the virtual environment:
```bash
python3 -m venv .venv
```

Activate the virtual environment:
```bash
source .venv/bin/activate
```


Add the source folder permanently to the Python path:
```bash
echo "$(pwd)/src" > `echo $VIRTUAL_ENV/lib/python*/site-packages/`src.pth
```

Install required Python packages:
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
python3 -m unittest test/test_npclad.py
```

## Run all unit tests using PyTest

Run all tests under the ./tests folder:
```bash
pytest
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

