"""
/***************************************************************************
 ProevenVerzameling
                                 A QGIS plugin
 Queries chosen features from a database

 Generated by Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/
                              -------------------
        begin                : 2019-10-20
        git sha              : $Format:%H$
        copyright            : (C) 2019 by Kevin Schuurman
        email                : k.schuurman1@rotterdam.nl
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""

import os
import pandas as pd
import numpy as np
import cx_Oracle
from typing import Union, List, Dict

# Import all necessary classes from QGIS
from qgis.core import QgsDataSourceUri, QgsCredentials, Qgis, QgsTask, QgsApplication, QgsVectorLayer
from qgis.PyQt.QtCore import QSettings, QTranslator, QCoreApplication, QRegExp
from qgis.PyQt.QtGui import QIcon, QRegExpValidator
from qgis.PyQt.QtWidgets import QAction, QDialogButtonBox, QProgressDialog

# Initialize Qt resources from file resources.py
from .resources import *

# Import the code for the dialog
from .proeven_verzameling_dialog import ProevenVerzamelingDialog

# Import the qgis_backend module
from . import qgis_backend


class ProevenVerzameling:
    """
    The class behind the Proeven_verzameling_dialog.ui.

    This class forms the logic behind the plugin's UI. Inputs and outputs from the UI
    are directly handled by the ProevenVerzameling class.

    Attributes
    ----------
    iface : QgsInterface object
        The current session Qgis interface instance that will be
        passed to this class which provides the hook by which you
        can manipulate the QGIS application at run time.
    plugin_dir : str
        Plugin directory location.
    translator : QTranslator object
        Translation object from Qt5.
    actions : list of QAction
        QAction list that Qgis performs when starting up the Plugin.
    menu : QString
        Translated menu name.
    first_start : bool
        Check if plugin was started the first time in current QGIS session.
    dlg : ProevenVerzamelingDialog object
        Reference to the ProevenVerzamelingDialog UI.

    Methods
    ----------
    __init__
        initializer.
    tr(message)
        Get the translation for a string using Qt translation API.
    add_action(
        icon_path,
        text,
        callback,
        enabled_flag=True,
        add_to_menu=True,
        add_to_toolbar=True,
        status_tip=None,
        whats_this=None,
        parent=None)
        Add a toolbar icon to the toolbar
    initGui()
        Create the menu entries and toolbar icons inside the QGIS GUI
    unload()
        Removes the plugin menu item and icon from QGIS GUI
    run()
        Run method that performs all the real work
    read_form()
        Extracts all inputs from the dialog form and provides run_task when succesful
    run_task(args)
        Sets up a QgsTask to do the heavy lifting in a background process of QGIS
    reset_ui()
        Reset all inputs to default values in the dialog
    get_credentials(host, port, database, username=None, password=None, message=None)
        Runs a QgsCredentials instance to input database credentials
    """

    def __init__(self, iface):
        """Initializer.

        Parameters
        ----------
        iface : QgsInterface object
            An interface instance that will be passed to this class
            which provides the hook by which you can manipulate the QGIS
            application at run time.
        """
        # Save reference to the QGIS interface
        self.iface = iface
        # initialize plugin directory
        self.plugin_dir = os.path.dirname(__file__)
        # initialize locale
        locale = QSettings().value('locale/userLocale')[0:2]
        locale_path = os.path.join(
            self.plugin_dir,
            'i18n',
            'ProevenVerzameling_{}.qm'.format(locale))

        if os.path.exists(locale_path):
            self.translator = QTranslator()
            self.translator.load(locale_path)
            QCoreApplication.installTranslator(self.translator)

        # Declare instance attributes
        self.actions = []
        self.menu = self.tr(u'&Proeven Verzameling')

        # Check if plugin was started the first time in current QGIS session
        # Must be set in initGui() to survive plugin reloads
        self.first_start = None

    def tr(self, message):
        """Get the translation for a string using Qt translation API.

        We implement this ourselves since we do not inherit QObject.

        Parameters
        ----------
        message : str, QString
            String for translation.
        
        Returns
        ----------
        QString
            Translated version of message.
        """
        # noinspection PyTypeChecker,PyArgumentList,PyCallByClass
        return QCoreApplication.translate('ProevenVerzameling', message)

    def add_action(
            self,
            icon_path,
            text,
            callback,
            enabled_flag=True,
            add_to_menu=True,
            add_to_toolbar=True,
            status_tip=None,
            whats_this=None,
            parent=None):
        """Add a toolbar icon to the toolbar.

        Parameters
        ----------
        icon_path : str
            Path to the icon for this action. Can be a resource path
            (e.g. ':/plugins/foo/bar.png') or a normal file system path
        text : str
            Text that should be shown in menu items for this action
        callback : function
            Function to be called when the action is triggered
        enabled_flag : bool, optional
            A flag indicating if the action should be enabled by default 
            (defaults to True)
        add_to_menu : bool, optional
            Flag indicating whether the action should also be added to the menu 
            (defaults to True)
        add_to_toolbar : bool, optional
            Flag indicating whether the action should also be added to the toolbar
            (defaults to True)
        status_tip : str or None, optional
            Text to show in a popup when mouse pointer hovers over the action 
            (defaults to None)
        whats_this : str or None, optional
            Text to show in the status bar when the mouse pointer hovers over
            the action (defaults to None)
        parent : QWidget or None, optional
            Parent widget for the new action (defaults to None)
        
        Returns
        ----------
        QAction
            The action that was created. Note that the action is also added to
            self.actions list.
        """

        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)

        if status_tip is not None:
            action.setStatusTip(status_tip)

        if whats_this is not None:
            action.setWhatsThis(whats_this)

        if add_to_toolbar:
            # Adds plugin icon to Plugins toolbar
            self.iface.addToolBarIcon(action)

        if add_to_menu:
            self.iface.addPluginToMenu(
                self.menu,
                action)

        self.actions.append(action)

        return action

    def initGui(self):
        """Create the menu entries and toolbar icons inside the QGIS GUI."""

        icon_path = ':/plugins/proeven_verzameling/icon.png'
        self.add_action(
            icon_path,
            text=self.tr(u'Proeven Verzameling'),
            callback=self.run,
            parent=self.iface.mainWindow())

        # will be set False in run()
        self.first_start = True

    def unload(self):
        """Removes the plugin menu item and icon from QGIS GUI."""
        for action in self.actions:
            self.iface.removePluginMenu(
                self.tr(u'&Proeven Verzameling'),
                action)
            self.iface.removeToolBarIcon(action)

    def run(self):
        """Run method that starts all the real work"""
        # Create the dialog with elements (after translation) and keep reference
        # Only create GUI ONCE in callback, so that it will only load when the plugin is started
        if self.first_start:
            self.first_start = False
            self.dlg = ProevenVerzamelingDialog()
            self.reset_ui()
            # Initialize QGIS filewidget to select a directory
            self.dlg.fileWidget.setStorageMode(1)
            # Signalling the Open button. Here the actual logic behind the plugin starts
            self.dlg.buttonBox.button(
                QDialogButtonBox.Ok).clicked.connect(self.read_form)
            # Signalling the reset button.
            self.dlg.buttonBox.button(
                QDialogButtonBox.RestoreDefaults).clicked.connect(self.reset_ui)
            rx1 = QRegExp(r"^\[\d{1,2}(\.\d{1})?(?:,\d{1,2}(\.\d{1})?)+\]$")
            vg_validator = QRegExpValidator(rx1)
            self.dlg.le_vg_sdp.setValidator(vg_validator)
            self.dlg.le_vg_trx.setValidator(vg_validator)

            rx2 = QRegExp(r"^[\w\-. ]+$")
            filename_validator = QRegExpValidator(rx2)
            self.dlg.le_outputName.setValidator(filename_validator)

        # show the dialog
        self.dlg.show()

    def read_form(self):
        """Extracts all inputs from the dialog form and provides run_task when succesful.
        
        The selected_layer object is used to find the connection info to the Oracle
        database. If the database connection is there but either the username or
        password is missing get_credentials() will be called until valid credentials
        are entered or the task is canceled.
        """
        filter_on_height = self.dlg.cb_filterOnHeight.isChecked()
        filter_on_volumetric_weight = self.dlg.cb_filterOnVolumetricWeight.isChecked()
        selected_layer = self.dlg.cmb_layers.currentLayer()
        trx_bool = self.dlg.cb_TriaxiaalProeven.isChecked()
        sdp_bool = self.dlg.cb_SamendrukkingProeven.isChecked()
        output_location = self.dlg.fileWidget.filePath()
        output_name = self.dlg.le_outputName.text()
        args = {'selected_layer': selected_layer,
                'output_location': output_location, 'output_name': output_name,
                'trx_bool': trx_bool, 'sdp_bool': sdp_bool
                }
        
        # General Asserts
        assert isinstance(selected_layer, QgsVectorLayer), 'De geselecteerde laag \'{}\' is geen vector laag.'.format(selected_layer.name())
        assert output_name, 'Het veld \'uitvoernaam\' mag niet leeg zijn.'
        assert output_location, 'Het veld \'uitvoermap\' mag niet leeg zijn.' 

        if trx_bool:
            # TRX Asserts
            assert any([self.dlg.cb_CU.isChecked(), self.dlg.cb_CD.isChecked(), self.dlg.cb_UU.isChecked()]), 'Een van de drie Proeftypes moet aangekruisd worden.'
            proef_types = []
            if self.dlg.cb_CU.isChecked():
                proef_types.append('CU')
            if self.dlg.cb_CD.isChecked():
                proef_types.append('CD')
            if self.dlg.cb_UU.isChecked():
                proef_types.append('UU') 
            args['proef_types'] = proef_types
            args['ea'] = self.dlg.sb_strain.value()
            args['save_plot'] = self.dlg.cb_savePlot.isChecked()

            if self.dlg.le_vg_trx.text():
                volG_trx = self.dlg.le_vg_trx.text().strip('[').strip(']').split(',')
                volG_trx = [float(x) for x in volG_trx]
                volG_trx.sort()
                if len(volG_trx) < 2:
                    self.iface.messageBar().pushMessage("Warning", 'Maar 1 volumegewicht interval voor triaxiaalproeven is gegeven, het interval wordt automatisch gegenereerd.', level=1, duration=5)
                    volG_trx = None
            else:
                volG_trx = None
            args['volG_trx'] = volG_trx

        if sdp_bool:
            if self.dlg.le_vg_sdp.text():
                volG_sdp = self.dlg.le_vg_sdp.text().strip('[').strip(']').split(',')
                volG_sdp = [float(x) for x in volG_sdp]
                volG_sdp.sort()
                if len(volG_sdp) < 2:
                    self.iface.messageBar().pushMessage("Warning", 'Maar 1 volumegewicht interval voor samendrukkingsproeven is gegeven, het interval wordt automatisch gegenereerd.', level=1, duration=5)
                    volG_sdp = None
            else:
                volG_sdp = None
            args['volG_sdp'] = volG_sdp

        if filter_on_height:
            args['maxH'] = self.dlg.sb_maxHeight.value()
            args['minH'] = self.dlg.sb_minHeight.value()
            assert args['maxH'] > args['minH'], 'Maximum hoogte moet hoger zijn dan minimum hoogte.'
        if filter_on_volumetric_weight:
            args['maxVg'] = self.dlg.sb_maxVolumetricWeight.value()
            args['minVg'] = self.dlg.sb_minVolumetricWeight.value()
            assert args['maxVg'] > args['minVg'], 'Maximum volumegewicht moet hoger zijn dan het minimum volumegewicht.'

        source = selected_layer.source()
        uri = QgsDataSourceUri(source)

        savedUsername = uri.hasParam('username')
        savedPassword = uri.hasParam('password')

        host = uri.host()
        port = uri.port()
        database = uri.database()
        username = uri.username()
        password = uri.password()

        errorMessage = None
        if savedUsername is True and savedPassword is True:
            try:
                qb = qgis_backend.QgisBackend(
                    host=host, port=port, database=database, username=username, password=password)
                qb.check_connection()
                args['qb'] = qb
                self.run_task(args)
            except cx_Oracle.DatabaseError as e:
                errorObj, = e.args
                errorMessage = errorObj.message
                suc = 'false'
                while suc == 'false':
                    suc, qb, errorMessage = self.get_credentials(
                        host, port, database, username=username, password=password, message=errorMessage)
                if suc == 'exit':
                    pass
                elif suc == 'true':
                    args['qb'] = qb
                    self.run_task(args)
        else:
            suc, qb, errorMessage = self.get_credentials(
                host, port, database, username=username, password=password)
            while suc == 'false':
                suc, qb, message = self.get_credentials(
                    host, port, database, message=errorMessage)
            if suc == 'exit':
                pass
            elif suc == 'true':
                args['qb'] = qb
                self.run_task(args)

    def run_task(self, args):
        """Sets up a QgsTask to do the heavy lifting in a background process of QGIS.
        
        Sets up the ProevenVerzamelingTask, which inherits from QgsTask, to handle the
        processor heavy queries and calculations in a background process, QGIS will stay
        stay responsive in the meantime. The ProevenVerzamelingTask is added to the QGIS
        taskmanager to handle it.

        Parameters
        ----------
        args : arguments dict
            Dictionary of arguments that will be passed to a QgsTask class.
        """
        progressDialog = QProgressDialog(
            'Initializing Task: BIS Bevraging...', 'Cancel', 0, 100)
        progressDialog.show()
        task = ProevenVerzamelingTask(
            'Proeven Verzameling Bevraging', self, **args)
        task.progressChanged.connect(
            lambda: progressDialog.setValue(task.progress()))
        progressDialog.canceled.connect(task.cancel)
        task.begun.connect(lambda: progressDialog.setLabelText(
            'Task Running: BIS Bevraging...'))
        QgsApplication.taskManager().addTask(task)

    def reset_ui(self):
        """Reset all inputs to default values in the dialog."""
        self.dlg.cb_filterOnHeight.setChecked(False)
        self.dlg.sb_maxHeight.setValue(100)
        self.dlg.sb_minHeight.setValue(-100)
        self.dlg.cb_filterOnVolumetricWeight.setChecked(False)
        self.dlg.sb_maxVolumetricWeight.setValue(22)
        self.dlg.sb_minVolumetricWeight.setValue(8)
        self.dlg.cb_TriaxiaalProeven.setChecked(False)
        self.dlg.cb_SamendrukkingProeven.setChecked(False)
        self.dlg.cb_CU.setChecked(True)
        self.dlg.cb_CD.setChecked(False)
        self.dlg.cb_UU.setChecked(False)
        self.dlg.sb_strain.setValue(5)
        self.dlg.cb_savePlot.setChecked(False)
        self.dlg.fileWidget.setFilePath(self.dlg.fileWidget.defaultRoot())
        self.dlg.le_outputName.setText('BIS_Geo_Proeven')

    def get_credentials(self, host, port, database, username=None, password=None, message=None):
        """
        Runs a QgsCredentials instance to ask for database credentials.

        The credentials are tested in a QgisBackend instance using
        test_connection(). Returns \'true\' or \'false\' depending on if the connection
        works. cx_Oracle errors will be caught and the error message returned.
        If the QgsCredentials dialog is canceled/escaped return \'exit\'.

        Parameters
        ----------
        host : str
            Oracle database host name
        port : str, int
            Oracle database port number
        database : str
            Oracle database service name
        username : str or None, optional
            Oracle database user (defaults to None)
        password : str or None, optional
            Oracle database password (defaults to None)
        message : str or None, optional
            Message to be shown in the credentials dialog, usually the errormessage 
            from a previous connection attempt
        
        Returns
        ----------
        str
            \'true\' if connection valid
            \'false\' if connection invalid
            \'exit\' if dialog is canceled
        qb : QgisBackend object
            QgisBackend with the database properties initialized
        errorMessage : str
            Oracle Database connection error message
        """
        
        uri = QgsDataSourceUri()
        # assign this information before you query the QgsCredentials data store
        uri.setConnection(host, port, database, username, password)
        connInfo = uri.connectionInfo()

        (success, user, passwd) = QgsCredentials.instance().get(
            connInfo, username, password, message)
        qb = None
        errorMessage = None
        if success:
            try:
                qb = qgis_backend.QgisBackend(
                    host=host, port=port, database=database, username=user, password=passwd)
                qb.check_connection()
                return 'true', qb, errorMessage
            except cx_Oracle.DatabaseError as e:
                errorObj, = e.args
                errorMessage = errorObj.message
                return 'false', qb, errorMessage
        else:
            return 'exit', qb, errorMessage


class ProevenVerzamelingTask(QgsTask):
    """Creating a task to run all the heavy processes in the background on a different
    thread.
    
    This class contains the methods for creating an excel file full with the
    ProevenVerzameling data.
    
    Attributes
    ----------
    iface : QgsInterface object
        The current session Qgis interface instance that will be
        passed to this class which provides the hook by which you
        can manipulate the QGIS application at run time
    exception : Exception or None
        Caught exception when one of the processing scripts fails
    qb : QgisBackend object
        QgisBackend object with initialized database parameters
    selected_layer : QgsVectorLayer
        Reference to the selected layer in QGIS
    output_location : str
    output_name : str
    maxH : float or int
        Maximal GeoMonster height in mNAP (default to 1000)
    minH : float or int
        Minimal GeoMonster sample height in mNAP (default to -1000)
    maxVg : float or int
        Maximal GeoMonster sample volumetric weight in kN/m3 (default to 40)
    minVg : float or int
        Minimal GeoMonster sample volumetric weight in kN/m3 (default to 0)
    trx_bool : bool
        Boolean to decide if Triaxiaal proeven will be queried
    proef_types : list of str
        Selection of TRX proefsoorten which will be queried (possibilities: 
        ['CU', 'CD', 'UU'], default to ['CU'])
    ea : list of int
        Selection of rek/strain percentages on which TRX statistics 
        will be calculated (default to [2])
    volG_trx : list of float or int, optional
        Intervals on which TRX statistics are calculated (default to None)
    save_plot : bool
        Save TRX statistics plots
    sdp_bool : bool
        Boolean to decide if Samendrukkingsproeven will be queried
    volG_sdp : list of float or int, optional
        Intervals on which SDP statistics are calculated (default to None)

    Methods
    ----------
    run()
        Is called when the QgsTask is ran in the background
    finished(result)
        This function is automatically called when the task has
        completed (successfully or not)
    cancel()
        Runs when QgsTask is canceled
    get_data()
        Control method for all the heavy work
    trx(gtm_ids)
        Queries TRX proeven and calculates the statistics with the 
        QgisBackend module
    sdp(gtm_ids)
        Queries SDP proeven with QgisBackend and calculates the statistics

    Notes
    ----------
    ProevenVerzamelingTask class's methods cannot be incorporated in the main
    plugin class (ProevenVerzameling) because the heavy work needs to be done in
    a secondary thread to keep the QGIS application responsive.
    """

    def __init__(self, description: str, ProevenVerzameling, **kwargs):
        super().__init__(description, QgsTask.CanCancel)
        self.iface = ProevenVerzameling.iface
        self.exception = None

        self.qb = kwargs.get('qb')
        self.selected_layer = kwargs.get('selected_layer')
        self.output_location = kwargs.get('output_location')
        self.output_name = kwargs.get('output_name')
        self.maxH = kwargs.get('maxH', 1000)
        self.minH = kwargs.get('minH', -1000)
        self.maxVg = kwargs.get('maxVg', 40)
        self.minVg = kwargs.get('minVg', 0)
        
        self.trx_bool = kwargs.get('trx_bool')
        if self.trx_bool:
            self.proef_types = kwargs.get('proef_types', ['CU'])
            self.ea = kwargs.get('ea', [2])
            self.volG_trx = kwargs.get('volG_trx', None)
            self.save_plot = kwargs.get('save_plot', False)

        self.sdp_bool = kwargs.get('sdp_bool')
        if self.sdp_bool:
            self.volG_sdp = kwargs.get('volG_sdp')

    def run(self) -> bool:
        """
        Is called when the QgsTask is ran in the background.

        Returns
        ----------
        bool
            When the task is completed without exceptions return True

        Notes
        ----------
        No exception can be raised inside a QgsTask therefore we catch
        them only raise them when we are in finished(). 
        Finished() is called from the main thread and can therefore raise exceptions.
        """
        try:
            result = self.get_data()
            if result:
                return True
            else:
                return False
        except Exception as e:
            self.exception = e
            return False

    def finished(self, result: bool):
        """
        This function is automatically called when the task has
        completed (successfully or not).

        You implement finished() to do whatever follow-up stuff
        should happen after the task is complete.
        finished is always called from the main thread, so it's safe
        to do GUI operations and raise Python exceptions here.

        returns
        ----------
        result : bool
            Result is the return value from self.run
        """
        if result:
            self.iface.messageBar().pushMessage(
                'Task: "{name}" completed in {duration} seconds.'.format(
                    name=self.description(),
                    duration=round(self.elapsedTime()/1000, 2)),
                Qgis.Info,
                duration=3)
        else:
            if self.exception is None:
                self.iface.messageBar().pushMessage(
                    'Task: "{name}" not successful but without '
                    'exception (probably the task was manually '
                    'canceled by the user)'.format(
                        name=self.description()),
                    Qgis.Warning,
                    duration=3)
            else:
                self.iface.messageBar().pushMessage(
                    'Task: "{name}" threw an Exception: {exception}'.format(
                        name=self.description(),
                        exception=self.exception),
                    Qgis.Critical,
                    duration=10)
                raise self.exception

    def cancel(self):
        """Runs when QgsTask is canceled."""

        self.iface.messageBar().pushMessage(
            'Task "{name}" was canceled.'.format(
                task=self.description()),
            Qgis.Info, duration=3)
        super().cancel()

    def get_data(self):
        """Control method for all the heavy work.
        
        In here QgisBackend will be used to
         - Get the selected Meetpunten from the QgsVectorLayer
         - Query the Meetpunten table, GeoDossier table and GeotechMonster table 
         from the BIS database
         - Query the TRX and SDP proeven and calculate the statistics
         - Export all data to an Excel file
        """
        self.setProgress(0)

        output_file = self.output_name + '.xlsx'

        # Check if the directory still has to be made.
        if os.path.isdir(self.output_location) is False:
            os.mkdir(self.output_location)

        # Extract the loc ids from the selected points in the selected layer
        loc_ids = self.qb.get_loc_ids(self.selected_layer)
        loc_ids = [int(x) for x in loc_ids]
        if self.isCanceled():
            return False
        self.setProgress(10)

        # Get all meetpunten related to these loc_ids
        df_meetp = self.qb.get_meetpunten(loc_ids)
        df_geod = self.qb.get_geo_dossiers(df_meetp.GDS_ID)

        if self.isCanceled():
            return False
        self.setProgress(20)

        df_gm = self.qb.get_geotech_monsters(loc_ids)
        if df_gm is not None:
            df_gm_filt_on_z = self.qb.select_on_z_coord(df_gm, self.maxH, self.minH)
            if df_gm_filt_on_z.empty:
                raise ValueError(
                    "There are no Geotechnische monsters in the depth range  {} to {} mNAP.".format(self.minH, self.maxH))

        # Add the df_meetp, df_geod and df_gm_filt_on_z to a dataframe dictionary
        df_dict = {'BIS_Meetpunten': df_meetp, 'BIS_GEO_Dossiers': df_geod,
                'BIS_Geotechnische_Monsters': df_gm_filt_on_z}

        if self.isCanceled():
            return False
        self.setProgress(30)

        if self.trx_bool:
            dict_trx,  fig_list = \
                self.trx(df_gm_filt_on_z.GTM_ID)
            df_dict.update(dict_trx)

        if self.isCanceled():
            return False
        self.setProgress(60)

        if self.sdp_bool:
            dict_sdp = self.sdp(df_gm_filt_on_z.GTM_ID)
            df_dict.update(dict_sdp)
        
        if self.isCanceled():
            return False
        self.setProgress(80)

        output_file_dir = os.path.join(self.output_location, output_file)
        if os.path.exists(output_file_dir):
            name, ext = output_file.split('.')
            i = 1
            while os.path.exists(os.path.join(self.output_location, name + '{}.'.format(i) + ext)):
                i += 1
            output_file_dir = os.path.join(self.output_location, name + '{}.'.format(i) + ext)
        
        # At the end of the 'with' function it closes the excelwriter automatically, even if there was an error
        # left out: writer in append mode so that the NEN tables are kept
        with pd.ExcelWriter(output_file_dir, engine='xlsxwriter', mode='w') as writer:
            for key in df_dict:
                if isinstance(df_dict[key], list):
                    row = 0
                    for df in df_dict[key]:
                        df.to_excel(writer, sheet_name=key, startrow=row)
                        row = row + len(df.index) + 2
                else:
                    # Writing every dataframe in the dictionary to a different sheet
                    df_dict[key].to_excel(writer, sheet_name=key, freeze_panes=(1, 1))
                
                if isinstance(df_dict[key], list):
                    columnnames = df_dict[key][0].columns
                else:
                    columnnames = df_dict[key].columns

                if not isinstance(columnnames, pd.MultiIndex):
                    sheet = writer.sheets[key]
                    # Sets the width of each column
                    sheet.set_column(0, 0, 10)
                    for i, colname in enumerate(columnnames):
                        n = i + 1 
                        sheet.set_column(n, n, len(str(colname)) * 1.25)

            self.setProgress(90)
            
            if self.trx_bool:  
                if self.save_plot:
                    i = 1
                    for fig in fig_list:
                        fig.savefig(os.path.join(self.output_location, 'fig_{}.pdf'.format(i)))
                        i = i + 1

        if self.isCanceled():
            return False
        os.startfile(output_file_dir)
        self.setProgress(100)
        return True

    def trx(self, gtm_ids: List[int]):
        """Queries TRX proeven and calculates the statistics with the 
        QgisBackend module.
        
        Parameters
        ----------
        gtm_ids: list[int]
            Selected Geotechnische monster ids
        
        Returns
        ----------
        df_dict: dict[pandas.DataFrame or list[pandas.DataFrame]]
            Returns a dictionary of pd.DataFrames. The key
            for each DataFrame is the (sheet)name. Multiple pd.DataFrames which need to
            be exported in a single sheet (such as statistics) are managed by putting
            them in a list inside the dictionary.
        fig_list: list[numpy.Figure or None]
            Statistic plot Figures. If save_plot is False returns list of None.
        """
        rek_selectie = [self.ea]

        df_trx = self.qb.get_trx(gtm_ids, proef_type=self.proef_types)

        df_trx = self.qb.select_on_vg(df_trx, self.maxVg, self.minVg)
        if df_trx.empty:
            raise pd.errors.EmptyDataError('Tussen {minVg} en {maxVg} kn/m3 volumegewicht zijn er geen Triaxiaalproeven.'.format(minVg=self.minVg, maxVg=self.maxVg))
        # Get all TRX results, TRX deelproeven and TRX deelproef results
        df_trx_results = self.qb.get_trx_result(df_trx.GTM_ID)
        df_trx_dlp = self.qb.get_trx_dlp(df_trx.GTM_ID)
        df_trx_dlp_result = self.qb.get_trx_dlp_result(df_trx.GTM_ID)

        self.setProgress(40)

        df_dict = {
            'BIS_TRX_Proeven': df_trx, 
            'BIS_TRX_Results': df_trx_results,
            'BIS_TRX_DLP': df_trx_dlp, 
            'BIS_TRX_DLP_Results': df_trx_dlp_result
        }

        lstsqrs_list = []
        fig_list = []
        # Doing statistics on the select TRX proeven
        if len(df_trx.index) > 1:

            if self.volG_trx is None:
                # Create a linear space between de maximal volumetric weight and the minimal volumetric weight
                minvg, maxvg = min(df_trx.VOLUMEGEWICHT_NAT), max(
                    df_trx.VOLUMEGEWICHT_NAT)
                N = round(len(df_trx.index) / 5) + 1
                cutoff = 1  # The interval cant be lower than 1 kn/m3
                if (maxvg - minvg)/N > cutoff:
                    Vg_linspace = np.linspace(minvg, maxvg, N)
                else:
                    N = round((maxvg-minvg)/cutoff)
                    if N < 2:
                        Vg_linspace = np.linspace(minvg, maxvg, 2)
                    else:
                        Vg_linspace = np.linspace(minvg, maxvg, N)

                Vgmax = Vg_linspace[1:]
                Vgmin = Vg_linspace[0:-1]
            else:
                Vgmax = self.volG_trx[1:]
                Vgmin = self.volG_trx[0:-1]
            
            self.setProgress(45)

            for ea in rek_selectie:
                df_list = []
                for vg_max, vg_min in zip(Vgmax, Vgmin):
                    # Make a selection for this volumetric weight interval
                    gtm_ids = self.qb.select_on_vg(
                        df_trx, Vg_max=vg_max, Vg_min=vg_min, soort='nat')['GTM_ID']
                    if len(gtm_ids) > 0:
                        # Create a tag for this particular volumetric weight interval
                        key = 'Vg: ' + str(round(vg_min, 1)) + \
                            '-' + str(round(vg_max, 1)) + ' kN/m3'
                        # Get the related TRX results...
                        df = df_trx_dlp_result.query('GTM_ID in [{}]'.format(
                            ','.join([str(x) for x in gtm_ids])))

                        fi, coh, E, E_per_n, eps, N, fig = self.qb.trx_least_squares(                            
                            df,
                            ea=ea,
                            plot_name='Least Squares Analysis, ea: ' +
                            str(ea) + '\n' + key,
                            save_plot=self.save_plot
                        )
                        fig_list.append(fig)
                        temp = pd.DataFrame(index=[key], data=[[round(vg_min, 1), round(vg_max, 1), fi, coh, E, E_per_n, eps, N]],
                                                columns=['MIN(VG)', 'MAX(VG)', 'FI', 'COH', 'ABS. SQ. ERR.', 'ABS. SQ. ERR./N', 'MEAN REL. ERR. %', 'N'])
                        df_list.append(temp)
                if len(df_list) > 0:
                    df = pd.concat(df_list)
                    df.index.name = 'ea: ' + str(ea) + '%'                    
                    lstsqrs_list.append(df)
            
            df_dict.update({
                'Least Squares Vg Stat.': lstsqrs_list
            })

        return df_dict, fig_list

    def sdp(self, gtm_ids: List[int]) -> Dict[str, Union[pd.DataFrame, List[pd.DataFrame]]]:
        """
        Queries SDP proeven with QgisBackend and calculates the statistics.
        
        ...

        Parameters
        ----------
        gtm_ids: list[int]
            Selected Geotechnische monster ids
        
        Returns
        ----------
        df_dict: dict[pandas.DataFrame or list[pandas.DataFrame]]
            Returns a dictionary of pd.DataFrame. The key
            for each DataFrame is the (sheet)name. Multiple pd.DataFrames which need to
            be exported in a single sheet (such as statistics) are managed by putting
            them in a list inside the dictionary.
        """

        df_dict = {}
        df_sdp = self.qb.get_sdp(gtm_ids)
        if df_sdp.empty:
            raise pd.errors.EmptyDataError('De geselecteerde meetpunten bevatten geen samendrukkingsproeven.')
        df_sdp = self.qb.select_on_vg(df_sdp, self.maxVg, self.minVg)
        if df_sdp.empty:
            raise pd.errors.EmptyDataError('Tussen {minVg} en {maxVg} kn/m3 volumegewicht zijn er geen Samendrukkingsproeven.'.format(minVg=self.minVg, maxVg=self.maxVg))
        else:
            df_sdp_result = self.qb.get_sdp_result(df_sdp.GTM_ID)
            df_dict.update({'BIS_SDP_Proeven': df_sdp,
                    'BIS_SDP_Resultaten': df_sdp_result})
            
            if self.volG_sdp is None:
                # Create a linear space between de maximal volumetric weight and the minimal volumetric weight
                minvg, maxvg = min(df_sdp.VOLUMEGEWICHT_NAT), max(
                    df_sdp.VOLUMEGEWICHT_NAT)
                N = round(len(df_sdp.index) / 5) + 1
                cutoff = 1  # The interval cant be lower than 1 kn/m3
                if (maxvg-minvg)/N > cutoff:
                    Vg_linspace = np.linspace(minvg, maxvg, N)
                else:
                    N = round((maxvg-minvg)/cutoff)
                    if N < 2:
                        Vg_linspace = np.linspace(minvg, maxvg, 2)
                    else:
                        Vg_linspace = np.linspace(minvg, maxvg, N)

                Vgmax = Vg_linspace[1:]
                Vgmin = Vg_linspace[0:-1]
            else:
                Vgmax = self.volG_sdp[1:]
                Vgmin = self.volG_sdp[0:-1]

            sdp_stat_list = []
            sdp_stat_data_list = []
            sdp_stat_invalid_list = []
            for vgmin, vgmax in zip(Vgmin, Vgmax):
                sdp = df_sdp[(df_sdp['VOLUMEGEWICHT_NAT'] >= vgmin) & (df_sdp['VOLUMEGEWICHT_NAT'] < vgmax)]
                sdp_slice = sdp[['GTM_ID', 'KOPPEJAN_PG', 'BJERRUM_PG']]

                rows = []
                for i, sdprow in sdp_slice.iterrows():
                    gtm_id = sdprow['GTM_ID']
                    grensspanning = np.max(sdprow[['KOPPEJAN_PG', 'BJERRUM_PG']])
                    df = df_sdp_result[(df_sdp_result['GTM_ID'] == gtm_id)].sort_values('STEP')
                    oldrow = None
                    for i, row in df.iterrows():
                        if oldrow is not None:
                            if (row['LOAD'] > grensspanning) & (oldrow['LOAD'] > grensspanning):
                                rows.append(row)
                                break
                        oldrow = row
                
                vg_str = 'Vg: {} - {} KN/m^3'.format(round(vgmin, 2), round(vgmax, 2))

                df_out = pd.DataFrame(columns=df_sdp_result.columns)
                df_out = df_out.append(rows)
                df_out.index.name = vg_str
                

                sdp_stat_data_list.append(df_out)

                df_out_val = df_out.iloc[:, 3:]
                sdp_stat = df_out_val.agg(['mean', 'std', 'count'])
                sdp_stat.index = pd.MultiIndex.from_tuples([(vg_str, 'Mean'), (vg_str, 'Std'), (vg_str, 'Count')])
                sdp_stat = sdp_stat.T
                sdp_stat[(vg_str, 'Count')] = sdp_stat[(vg_str, 'Count')].astype('int64')
                sdp_stat_list.append(sdp_stat)
            
            sdp_stat = pd.concat(sdp_stat_list, 1)
            df_dict.update({
                'SDP_RAW': sdp_stat_data_list,
                'SDP_INVALID': sdp_stat_invalid_list,
                'SDP_STAT': sdp_stat
            })

        return df_dict

