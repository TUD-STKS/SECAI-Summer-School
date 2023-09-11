# "SECAI Summer School"
#
# Copyright (C) 2023 Peter Steiner
# License: BSD 3-Clause

python.exe -m venv venv

.\venv\Scripts\activate.ps1

python.exe -m pip install --upgrade .[notebook]

jupyter-lab.exe

deactivate
