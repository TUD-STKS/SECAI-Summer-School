# "SECAI Summer School"
#
# Copyright (C) 2023 Peter Steiner
# License: BSD 3-Clause

python.exe -m venv venv

.\venv\Scripts\activate.ps1
python.exe -m pip install -r requirements.txt
python.exe -m pip install --editable .
# TODO add the correct programm call
python.exe .\src\main.py --fit_basic_esn

deactivate
