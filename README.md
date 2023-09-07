# Introduction to Machine Learning
## Metadata
- [Peter Steiner](mailto:peter.steiner@tu-dresden.de)
- AI Applications for Medicine, International Summer School in Dresden,
Technische Universität Dresden, Dresden, Saxony, Germany
- Weblink:
[https://www.secai-ceti-summerschool.de](https://www.secai-ceti-summerschool.de)

## Summary and Contents
This repository contains code accompanying the workshop entitled "Introduction to 
Machine Learning". The Jupyter Notebooks are prepared to redo all steps introduced 
during the workshop.

## File list
- The following scripts are provided in this repository
    - `scripts/run_jupyter-lab.sh`: UNIX Bash script to start the Jupyter Notebook for 
   the workshop.
    - `scripts/run_jupyter-lab.bat`: Windows batch script to start the Jupyter Notebook 
  for the workshop.
- The following Python code is provided in `src`
    - `src/data/dataset_without_pytorch.py`: Utility functions for data handling.
- `requirements.txt`: Text file containing all required Python modules to be installed. 
- `README.md`: The README displayed here.
- `LICENSE`: Textfile containing the license for this source code. You can find 
- `results/`
    - (Pre)-trained modelss.
- `.gitignore`: Command file for Github to ignore files with specific extensions.

## Usage
The easiest way to get started is to either use Binder or Colab. Links to open the 
Jupyter Notebook there are given below.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TUD-STKS/SECAI-Summer-School/blob/main/IntroductionMachineLearning.ipynb)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/TUD-STKS/SECAI-Summer-School/HEAD)

To run the scripts or to start the Jupyter Notebook locally, at first, please ensure 
that you have a valid Python distribution installed on your system. Here, at least 
Python 3.9 is required.

You can then call `run_jupyter-lab.ps1` or `run_jupyter-lab.sh`. This will install a new 
[Python venv](https://docs.python.org/3/library/venv.html), which is our recommended way 
of getting started.

## Acknowledgements
This research was supported by
```
Nobody
```

## License and Referencing
This program is licensed under the BSD 3-Clause License.

More information about licensing can be found in 
[Wikipedia](https://en.wikipedia.org/wiki/License).


## Appendix
For any questions, do not hesitate to open an issue or to drop a line to [Peter Steiner](mailto:peter.steiner@tu-dresden.de)
