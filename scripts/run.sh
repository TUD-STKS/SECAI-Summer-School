#!/bin/sh

# "SECAI Summer School"
#
# Copyright (C) 2023 Peter Steiner
# License: BSD 3-Clause

python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m pip install --editable .
# TODO add the correct programm call
python3 src/main.py --fit_basic_esn

deactivate
