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
  * xlwt
  * ...

```
Give examples
```

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

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc

