# Proeven_verzameling plugin

QGIS plugin to automatically Query Geotechnical Parameters from the BIS database. Continuation of [proeven_verzameling](https://github.com/KRS-dev/proeven_verzameling).


## Getting Started

These instructions will get you a copy of the plugin up and running on your local machine. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

* #### Oracle BIS database running in the background

* #### QGIS 3.x.x
  with Python 3 and an Oracle database driver 

* #### Python Modules Installed

  * cx_Oracle
  * numpy
  * matplotlib
  * pandas
  * xlsxwriter

### Installing

A step by step series of examples that tell you how to get [proeven_verzameling_plugin](https://github.com/KRS-dev/proeven_verzameling_plugin) running in your QGIS build.



Step 1: Download the .zip file of the repository 

Step 2: Install it using __Install from ZIP__ in QGIS under __Plugins__ -> __Manage and Install Plugins...__

![install from ZIP](/images/install_from_zip.PNG)


And you are finished installing the plugin!
A small icon, ![icon](/icon.png), should now show up in your QGIS toolbar. If you click on it a form will show up where your inputs are asked to query the BIS database.

![Plugin Window](/images/plugin_window.PNG)


## Deployment

Plug-in creates a database connection through the login information in the Parameter layer in QGIS. 
Therefore to make the plug-in work:

Step 1: Import the Parameter layer (or make your own). Opening the QGIS project with the Parameter layer works as well.

Step 2: Select points in the Parameter layer that you want fetched.

Step 3: Start the Plug-in and select the Parameter layer in the top combobox in the plug-in window.

![Plugin Window](/images/plugin_window.PNG)

## Built With

* [Qgis-Plugin-Builder](https://github.com/g-sherman/Qgis-Plugin-Builder) - Qgis Plugin Template
* [cx_Oracle](https://oracle.github.io/python-cx_Oracle/) - Oracle Database Driver


## Authors

* **Kevin Schuurman** 

## License

Needs to be filled

## Acknowledgments


