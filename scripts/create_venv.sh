#!/bin/sh

# "SECAI Summer School"
#
# Copyright (C) 2023 Peter Steiner
# License: BSD 3-Clause

python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade .[notebook]
python3 -m medmnist download
deactivate
