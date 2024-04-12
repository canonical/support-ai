VENV = venv
PIP = pip
PYTHON = python3
RMDIR = rm -rf
FIND = find
VENV_BIN = $(VENV)/bin
VENV_PYTHON = $(VENV_BIN)/python3
COLLECTION_META = collection_metadata
VECTORDB_DIR = vectordb

all: prepare

prepare: requirements.txt
	$(PYTHON) -m venv $(VENV)
	. $(VENV_BIN)/activate && $(PYTHON) -m $(PIP) install -r requirements.txt

clean:
	$(RMDIR) $(VENV) $(COLLECTION_META) $(VECTORDB_DIR)
	$(FIND) . -depth -type d -name __pycache__ -exec $(RMDIR) {} \;
