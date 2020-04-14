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
  * xlwt

### Installing

A step by step series of examples that tell you how to get [proeven_verzameling_plugin](https://github.com/KRS-dev/proeven_verzameling_plugin) running in your QGIS build.


Step 1: Connecting to the Oracle database in QGIS.

![new connection](/images/new_connection.png)
![GitHub Logo](/images/connection_window.PNG)

Step 2: Download the .zip file of the repository and install it using __Install from ZIP__ in QGIS under __Plugins__ -> __Manage and Install Plugins...__

![install from ZIP](/images/install_from_zip.PNG)


And you are finished installing the plugin!
A small icon, ![icon](/icon.png), should now show up in your QGIS toolbar. If you click on it a form will show up where your inputs are asked to query the BIS database.

![Plugin Window](/images/plugin_window.PNG)


## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Qgis-Plugin-Builder](https://github.com/g-sherman/Qgis-Plugin-Builder) - Qgis Plugin Template
* [cx_Oracle](https://oracle.github.io/python-cx_Oracle/) - Oracle Database Driver


## Authors

* **Kevin Schuurman** 

## License

Needs to be filled

## Acknowledgments


