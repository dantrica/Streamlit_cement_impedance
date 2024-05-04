# Software KDssZ version 1.0

This software tool works to analyze impedance measurements through a lumped circuital model.

The user can upload the impedance measurements in format .txt

## The left sidebar contains:

This object contains the elements that will be presented in the left sidebar, such as i) the uploading file option, ii) the lumped circuit model, iii) Add and Remove buttons to move data in the Optimization zone, iv) Run and Reset buttons to start and clear the optimization.



## The repository contains:

- The Python module called [KDssZ.py](https://github.com/dantrica/Streamlit_cement_impedance/blob/c79735cefb679f8cb21de758db89e5011d7d15d4/KDssZ.py) contains the algorithms to optimize the electrical impedance measurements.
- All [data](https://github.com/dantrica/Streamlit_cement_impedance/tree/c79735cefb679f8cb21de758db89e5011d7d15d4/data) acquired during the experimental campaign is located in this repository Streamlit_cement_impedance/data/.
- The [Figures](https://github.com/dantrica/Streamlit_cement_impedance/tree/c79735cefb679f8cb21de758db89e5011d7d15d4/figures) folder, where the new dataframes (Excel files), figures (.png files), or new series (.txt files) are saved.
- Some python [scripts](scripts) that can be called by [analytics_GO.ipynb](analytics_GO.ipynb) to improve the mechanical or electrical characterization.
- The [Simulink](/Simulink) folder conotains the Simulink scripts (files with extension .slx) and the parameter estimator files for each rGO-cement composites frabricated.

## Project Description: 

$\texttt{KDssZ}$ has been developed to optimize experimental electrical impedance data to a lumped circuit model. This approach allows users to correlate the interactions among the components in the composite-electrodes, cement, nanoparticles, and pores with the measured impedance, which is analyzed across three frequency ranges: high, medium, and low. Although the software has been validated with impedance measurements from gold nanoparticles/cement-based composites, it can also be used to study the electrical properties of other cement-based composites. KDssZ is deployed on the Streamlit platform, and in line with the principles of open science, it allows researchers working on similar studies to use this tool to explore new research questions related to cement-based composites.

## Usage instructions:

The module must be [analytics_GO.ipynb](analytics_GO.ipynb) uploaded to Google Drive and opened with Google Collaboratory. Then, the folders [data](/data), [scripts](/scripts), and [outputs](/outputs) must be uploaded to the Google Colab folder. 

## License information:

The module [analytics_GO.ipynb](analytics_GO.ipynb) is open source and published under the GPL option. Nevertheless, the Matlab files require the user get has a LICENCE agreement with Matlab.

## Troubleshooting:

The users must be take into account that the pandas' library requires the openpyxl module. Google collaboratory uses to update the version of this library periodically. Then, the users must modify the version of openpyxl on the section <b>0.3. Importing the libraries: lmfit, pro_data, ... from the Colab directory in Google Drive.</b> into the module [analytics_GO.ipynb](analytics_GO.ipynb)

### Matlab:
The authors have tested the Matlab (Simulink model) scripts with Matlab 9.12 (R2022a), but it should be compatible also with older versions.


## Contact information:

For support, comments or sugestions feel free to contact dantrica@saber.uis.edu.co
