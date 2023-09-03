# HAR_with_CSI_data-stream
## Prerequisite
Tensorflow >= 1.0 \
numpy\
pandas\
matplotlib\
scikit-learn


## How to run
- Download dataset from [here](https://drive.google.com/file/d/19uH0_z1MBLtmMLh8L4BlNA0w-XAFKipM/view)
- "git clone" this repository.

- Run the feature extract.py\
This script extracts DWT fetaures of all csv files(input features & label) of each activity.　　

- run any model of choice to test the results.


## Dataset
The files with "input_" prefix are WiFi Channel State Information data.
- 1st column shows timestamp.
- 2nd - 91st column shows (30 subcarrier * 3 antenna) amplitude.
- 92nd - 181st column shows (30 subcarrier * 3 antenna) phase.

The files with "annotation_" prefix are annotation data.
- [PCA_STFT_visualize](https://github.com/shreyu258/HAR_with_CSI_data-stream/blob/main/PCA_DWT_visualize_plots.ipynb) shows the plot of all features.
