

# Proeven_verzameling plugin - English

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



Step 1: Download the .zip file from the repository 

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

Step 3: Start the Plug-in and select the Parameter layer in the top combobox in the plug-in dialog.

Step 4: Fill in the rest of the dialog window and click ok.

![Plugin Window](/images/plugin_window.PNG)

## Built With

* [Qgis-Plugin-Builder](https://github.com/g-sherman/Qgis-Plugin-Builder) - Qgis Plugin Template
* [cx_Oracle](https://oracle.github.io/python-cx_Oracle/) - Oracle Database Driver


## Authors

* **Kevin Schuurman** 

## License

Needs to be filled





# Proeven_verzameling plugin - Nederlands

QGIS plug-in om automatisch geotechnische data op te vragen uit de BIS database. Continuation of [proeven_verzameling](https://github.com/KRS-dev/proeven_verzameling).


## Opstarten

Deze instructies zorgen dat je QGIS plug-in geinstalleerd is en je hem kan gebruiken op je QGIS versie op het Gemeente Rotterdam Netwerk. Voor het gebruik van de Plug-in verwijs ik door naar deployment.

### Vereiste

* #### Oracle BIS database in de achtergrond

* #### QGIS 3.x.x
  with Python 3 and an Oracle database driver 

* #### Python Modules geinstalleerd in QGIS

  * cx_Oracle
  * numpy
  * matplotlib
  * pandas
  * xlsxwriter

### Installeren

Een stap bij stap serie hoe je de [proeven_verzameling_plugin](https://github.com/KRS-dev/proeven_verzameling_plugin) kan installeren in QGIS.

Stap 1: Download de .zip file van de github repository 

Stap 2: Installeer de .zip in QGIS via __Install from ZIP__ in QGIS odner __Plugins__ -> __Manage and Install Plugins...__

![install from ZIP](/images/install_from_zip.PNG)


En je bent klaar met de installatie!
De plug-in, ![icon](/icon.png), zou nu moeten verschijnen in de toolbar van QGIS. Als je erop klikt krijg je de dialoog waarmee je de BIS database bevraagd.

![Plugin Window](/images/plugin_window.PNG)


## Gebruik

Deze Plug-in maakt een database connectie via de inloggegevens die in de Parameter laag staan in QGIS, daarom kan de plug-in alleen gebruikt worden met de Parameter laag of een vergelijkbare laag.
Om de plug-in te gebruiken moet je dus:

Stap 1: Voeg de Meetpunten laag (of maak je eigen laag). Het bijgevoegde QGIS project openen met Meetpunten laag werkt ook.

Stap 2: Selecteer de punten in de Meetpunten laag die je wil opvragen.

Stap 3: Start de Plug-in en selecteer de Meetpunten laag in de bovenste combobox van het plug-in formulier.

Stap 4: Vul naar wensen de rest van het formulier in en druk ok.

![Plugin Window](/images/plugin_window.PNG)

## Gemaakt met

* [Qgis-Plugin-Builder](https://github.com/g-sherman/Qgis-Plugin-Builder) - Qgis Plugin Template
* [cx_Oracle](https://oracle.github.io/python-cx_Oracle/) - Oracle Database Driver


## Auteurs 

* **Kevin Schuurman** 

## Licentie

Needs to be filled




