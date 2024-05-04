# Software KDssZ version 1.0

This software tool works to analyze impedance measurements through a lumped circuital model.

## The repository contains:

- The Python module called [KDssZ.py](https://github.com/dantrica/Streamlit_cement_impedance/blob/c79735cefb679f8cb21de758db89e5011d7d15d4/KDssZ.py) contains the algorithms to optimize the electrical impedance measurements.
- The Python module called [appZ.py](https://github.com/dantrica/Streamlit_cement_impedance/blob/b421c1652a15b18d4e0220819822e362abc5bec6/appZ.py) contains the code to deploy the application on the Streamlit platform.
- All [data](https://github.com/dantrica/Streamlit_cement_impedance/tree/c79735cefb679f8cb21de758db89e5011d7d15d4/data) acquired during the experimental campaign is located in this repository Streamlit_cement_impedance/data/.
- The [Figures](https://github.com/dantrica/Streamlit_cement_impedance/tree/c79735cefb679f8cb21de758db89e5011d7d15d4/figures) folder contains an image describing the lumped circuit model.

## The left sidebar functionalities:

The user can upload the impedance measurements in format .txt. Moreover, this object contains the elements that will be presented in the left sidebar, such as i) the uploading file option, ii) the lumped circuit model, iii) Add and Remove buttons to move data in the Optimization zone, iv) Run and Reset buttons to start and clear the optimization.

## Project Description: 

$\texttt{KDssZ}$ has been developed to optimize experimental electrical impedance data to a lumped circuit model. This approach allows users to correlate the interactions among the components in the composite-electrodes, cement, nanoparticles, and pores with the measured impedance, which is analyzed across three frequency ranges: high, medium, and low. Although the software has been validated with impedance measurements from gold nanoparticles/cement-based composites, it can also be used to study the electrical properties of other cement-based composites. KDssZ is deployed on the Streamlit platform, and in line with the principles of open science, it allows researchers working on similar studies to use this tool to explore new research questions related to cement-based composites.

## Usage instructions:

The main functionalities of the software KDssZ are displayed in Figure~\ref{}, and enumerated according to the following list:

1. Push button \texttt{Browse Files} to upload \texttt{.txt} files with electrical impedance spectroscopy data.
2. Help menu explaining the format files to be received; help hints boxes to report whether a file was added previously or it is already in the optimization zone; and help tips on buttons.
3. Choosing the file data since Select tag.
4. Buttons $\texttt{Add data}$ and $\texttt{Remove}$, to add or remove data to the optimization zone.
5. Buttons $\texttt{Run}$ and $\texttt{Reset}$, to start the optimization, create the plots and return the DataFrame with the parameters.
6. Figures: wide mode and saving options.
7. Table: wide mode, searching information into the DataFrame, and save data as $\texttt{.csv}$ file.

## License information:

The application [KDssZ](https://kdssz-impedance.streamlit.app/) is open source and published under the GPL option on the [Streamlit](https://streamlit.io/) platform.
## Troubleshooting:

The users must take into account that the pandas' library requires the openpyxl module. Other requirements can be found in [requirements.txt](https://github.com/dantrica/Streamlit_cement_impedance/blob/b421c1652a15b18d4e0220819822e362abc5bec6/requirements.txt)

## Contact information:

For support, comments or sugestions feel free to contact daniel.triana.camacho@gmail.com
