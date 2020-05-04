'''
Script: qgis_backend.py
Repository: https://github.com/KRS-dev/proeven_verzameling_plugin
Author: Kevin Schuurman
E-mail: kevinschuurman98@gmail.com
Summary: Base functions for qgis_frontend.py, Querying the data from the proeven verzameling database.
'''
import pandas as pd
import numpy as np
#import psycopg2 as psy
import cx_Oracle as cora
# or
#import pyodbc as cora
import matplotlib.pyplot as plt
import matplotlib.offsetbox as offsetbox


class qgis_backend:

    def __init__(self, host, port, database, username, password):
        self.username = username
        self.password = password
        self.bis_dsn = cora.makedsn(host, port, service_name=database)

    def check_connection(self):
        with cora.connect(
            user=self.username,
            password=self.password,
            dsn=self.bis_dsn
        ) as dbcon:
            pass

    # fetch Connection
    def fetch(self, query, data):
        '''
        ## Using a PostgreSQL database
        with psy.connect(
            host = self.host,
            database = self.database,
            user = self.username,
            password = self.password
            ) as dbcon:
        '''
        # Using an Oracle database:
        with cora.connect(
            user=self.username,
            password=self.password,
            dsn=self.bis_dsn
        ) as dbcon:
            # *Can be a:
            # 1. Oracle Easy Connect string
            # 2. Oracle Net Connect Descriptor string
            # 3. Net Service Name mapping to connect description

            cur = dbcon.cursor()
            cur.execute(query, data)
            fetched = cur.fetchall()
            description = cur.description
            return fetched, description

    # Getting the loc_id's from the Qgislayer
    def get_loc_ids(self, QgisLayer):
        loc_ids = []
        features = QgisLayer.selectedFeatures()

        if len(features) > 0:
            for f in features:
                try:
                    loc_ids.append(f.attribute('loc_id'))
                except KeyError:
                    raise KeyError(
                        'This layer does not contain an attribute called loc_id')
                except:
                    raise IOError(
                        'Something went wrong in selecting the attribute \'loc_id\'')
            return loc_ids
        else:
            raise KeyError('No features were selected in the layer')

    # Querying meetpunten
    def get_meetpunten(self, loc_ids):
        if isinstance(loc_ids, (list, tuple, pd.Series)):
            if len(loc_ids) > 0:
                if(all(isinstance(x, int) for x in loc_ids)):
                    values = list(loc_ids)
                    chunks = [values[x:x + 1000]
                              for x in range(0, len(values), 1000)]
                    df_list = []
                    for chunk in chunks:
                        values = chunk
                        bindValues = [':' + str(i+1)
                                      for i in range(len(values))]
                        query = 'SELECT LGA_ID, LOC_ID, GRAF_PRIM_SOORT, SUB_TYPE, AANGR_PUNT_X, AANGR_PUNT_Y, T, MPT_ID, GBO_KODE, PJM_ID, TDK_KODE, KODE_FIN_PROJECT, STATUS, MPB_SUB_TYPE, MPO_SUB_TYPE, BOR_SUB_TYPE, GBO_SUB_TYPE, MBO_SUB_TYPE, DATUM, GDS_ID, REFERENTIEVLAK, REFERENTIEVLAK_NIVEAU, NIVEAU_TOV_REFVLAK, NIVEAU_TOV_NAP, FOTO ' \
                            + 'FROM bis_graf_loc_aanduidingen '\
                            + 'INNER JOIN bis_meetpunten ON bis_meetpunten.mpt_id = bis_graf_loc_aanduidingen.loc_id '\
                            + 'WHERE bis_graf_loc_aanduidingen.loc_id IN ({})'.format(','.join(bindValues))
                        fetched, description = self.fetch(query, values)
                        if (0 < len(fetched)):
                            meetp_df = pd.DataFrame(fetched)
                            colnames = [desc[0] for desc in description]
                            meetp_df.columns = colnames
                            meetp_df = meetp_df.set_index('MPT_ID')
                            meetp_df.GDS_ID = meetp_df.GDS_ID.fillna(0)
                            meetp_df.GDS_ID = pd.to_numeric(
                                meetp_df.GDS_ID, downcast='integer')
                            df_list.append(meetp_df)
                    meetp_df_all = pd.concat(df_list, ignore_index=True)
                    if meetp_df_all.empty is False:
                        return meetp_df_all
                    else:
                        raise ValueError(
                            'These selected geometry points do not contain valid loc_ids: ' + str(values))
                else:
                    raise TypeError('not all inputs are integers')
            else:
                raise ValueError('No bor_ids were supplied.')
        else:
            raise TypeError('Input is not a list or tuple')

    def get_geo_dossiers(self, gds_ids):
        """Querying geodossiers"""
        if isinstance(gds_ids, (list, tuple, pd.Series)):
            if len(gds_ids) > 0:
                if(all(isinstance(x, int) for x in gds_ids)):
                    values = list(gds_ids)
                    chunks = [values[x:x+1000]
                              for x in range(0, len(values), 1000)]
                    df_list = []
                    for chunk in chunks:
                        values = chunk
                        bindValues = [':' + str(i+1)
                                      for i in range(len(values))]
                        query = 'SELECT * FROM bis_geo_dossiers WHERE gds_id IN ({})'.format(
                            ','.join(bindValues))
                        fetched, description = self.fetch(query, values)
                        if (0 < len(fetched)):
                            geod_df = pd.DataFrame(fetched)
                            colnames = [desc[0] for desc in description]
                            geod_df.columns = colnames
                            df_list.append(geod_df)
                    geod_df_all = pd.concat(df_list, ignore_index=True)
                    if geod_df_all.empty is False:
                        return geod_df_all
                    '''else:
                        raise ValueError('The selected gds_ids: ' + str(values) + \
                        ' do not contain any geodossiers.')'''
                else:
                    raise TypeError('not all inputs are integers')
            else:
                raise ValueError('No bor_ids were supplied.')
        else:
            raise TypeError('Input is not a list or tuple')

    def get_geotech_monsters(self, bor_ids):
        """Querying geotechnische monsters"""
        if isinstance(bor_ids, (list, tuple, pd.Series)):
            if len(bor_ids) > 0:
                if(all(isinstance(x, (int)) for x in bor_ids)):
                    values = list(bor_ids)
                    chunks = [values[x:x+1000]
                              for x in range(0, len(values), 1000)]
                    df_list = []
                    for chunk in chunks:
                        values = chunk
                        bindValues = [':' + str(i+1)
                                      for i in range(len(values))]
                        query = 'SELECT * FROM bis_geotech_monsters WHERE bor_id IN ({})'.format(
                            ','.join(bindValues))
                        fetched, description = self.fetch(query, values)
                        if(len(fetched) > 0):
                            g_mon_df = pd.DataFrame(fetched)
                            colnames = [desc[0] for desc in description]
                            g_mon_df.columns = colnames
                            g_mon_df['Z_COORDINAAT_LAAG'] = pd.to_numeric(
                                g_mon_df['Z_COORDINAAT_LAAG'])
                            df_list.append(g_mon_df)
                    g_mon_df_all = pd.concat(df_list)
                    if g_mon_df_all.empty is False:
                        return g_mon_df_all
                    '''else:
                        print('These selected boring(en): ' + str(values) + \
                        ' do not contain any GEO_Monsters.')'''
                else:
                    raise TypeError('not all inputs are integers')
            else:
                raise ValueError('No bor_ids were supplied.')
        else:
            raise TypeError('Input is not a list or tuple')

    def select_on_z_coord(self, g_mon_df, zmax, zmin):
        """Filter on height of the Geotechnical monsters"""
        if isinstance(g_mon_df, pd.DataFrame):
            new_g_mon_df = g_mon_df[(zmax > g_mon_df.Z_COORDINAAT_LAAG) & (
                g_mon_df.Z_COORDINAAT_LAAG > zmin)]
            if new_g_mon_df is not None:
                return new_g_mon_df
            '''else:
                print('Between ' + zmax + ' and ' + zmin + 'm there are no GEO_Monsters found.')'''
        else:
            raise TypeError('No pandas dataframe was supplied')

    def get_trx(self, gtm_ids, proef_type=('CD')):
        """Querying TRX_proeven"""
        if isinstance(gtm_ids, (list, tuple, pd.Series)):
            if all(any(x == i for i in ('CU', 'CD', 'UU')) for x in proef_type):
                if len(gtm_ids) > 0:
                    if all(isinstance(x, (int)) for x in gtm_ids):
                        values = list(gtm_ids)
                        chunks = [values[x:x+990]
                                  for x in range(0, len(values), 990)]
                        df_list = []
                        for chunk in chunks:
                            values = chunk
                            bindValues = [':' + str(i+1)
                                          for i in range(len(values))]
                            proef_type = list(proef_type)
                            bindProef = [':p' + str(i + 1)
                                         for i in range(len(proef_type))]
                            bindAll = bindValues + bindProef
                            values = values + proef_type
                            bindDict = dict(zip(bindAll, values))
                            query = 'SELECT * FROM bis_trx_proeven WHERE proef_type IN ({}) AND gtm_id IN ({})'.format(
                                ','.join(bindProef), ','.join(bindValues))
                            fetched, description = self.fetch(query, bindDict)
                            if(len(fetched) > 0):
                                trx_df = pd.DataFrame(fetched)
                                colnames = [desc[0] for desc in description]
                                trx_df.columns = colnames
                                trx_df[['VOLUMEGEWICHT_DROOG', 'VOLUMEGEWICHT_NAT', 'WATERGEHALTE', 'TEREINSPANNING', 'BEZWIJKSNELHEID']] = \
                                    trx_df[['VOLUMEGEWICHT_DROOG', 'VOLUMEGEWICHT_NAT', 'WATERGEHALTE',
                                            'TEREINSPANNING', 'BEZWIJKSNELHEID']].apply(pd.to_numeric)
                                trx_df.VOLUMEGEWICHT_NAT = trx_df.VOLUMEGEWICHT_NAT.astype(
                                    float)
                                df_list.append(trx_df)
                        trx_df_all = pd.concat(df_list, ignore_index=True)
                        if trx_df_all.empty is False:
                            return trx_df_all
                        '''else:
                            raise ValueError('These selected boring(en): ' + str(values) + \
                                ' do not contain any triaxiaal proeven with proef_type: ' + str(proef_type))'''
                    else:
                        raise TypeError('not all inputs are integers')
                else:
                    raise ValueError('No gtm_ids were supplied.')
            else:
                raise TypeError(
                    'Only CU, CD and UU or a combination of the types mentioned are allowed as proef_type')
        else:
            raise TypeError('Input is not a list or tuple')

    def select_on_vg(self, trx_df, Vg_max=20, Vg_min=17, soort='nat'):
        """Filter on Volumetric weight"""
        # Volume gewicht y in kN/m3
        if isinstance(trx_df, pd.DataFrame):
            if soort == 'nat':
                new_trx_df = trx_df[(Vg_max >= trx_df.VOLUMEGEWICHT_NAT) & (
                    trx_df.VOLUMEGEWICHT_NAT >= Vg_min)]
            elif soort == 'droog':
                new_trx_df = trx_df[(Vg_max >= trx_df.VOLUMEGEWICHT_DROOG) & (
                    trx_df.VOLUMEGEWICHT_DROOG >= Vg_min)]
            else:
                raise TypeError('\'' + soort + '\' is not allowed as argument for soort,\
                    only \'nat\' and \'droog\' are allowed.')
            if new_trx_df is not None:
                return new_trx_df
            '''else:
                print('Between: \'' + Vg_max + '\' and \'' + Vg_min + '\' kg/m3 there are no GEO_Monsters found')'''
        else:
            raise TypeError('No pandas dataframe was supplied')

    def get_trx_result(self, gtm_ids):
        """Querying TRX_results"""
        if isinstance(gtm_ids, (list, tuple, pd.Series)):
            if len(gtm_ids) > 0:
                if all(isinstance(x, (int)) for x in gtm_ids):
                    values = list(gtm_ids)
                    chunks = [values[x:x+1000]
                              for x in range(0, len(values), 1000)]
                    df_list = []
                    for chunk in chunks:
                        values = chunk
                        bindValues = [':' + str(i+1)
                                      for i in range(len(values))]
                        query = 'SELECT * FROM bis_trx_proef_result WHERE gtm_id IN ({})'.format(
                            ','.join(bindValues))
                        fetched, description = self.fetch(query, values)
                        if(len(fetched) > 0):
                            trx_result_df = pd.DataFrame(fetched)
                            colnames = [desc[0] for desc in description]
                            trx_result_df.columns = colnames
                            trx_result_df[['EA', 'COH', 'FI']] = trx_result_df[[
                                'EA', 'COH', 'FI']].apply(pd.to_numeric)
                            df_list.append(trx_result_df)
                    trx_result_df_all = pd.concat(df_list, ignore_index=True)
                    if trx_result_df_all.empty is False:
                        return trx_result_df_all
                    '''else:
                        print('These selected boring(en): ' + str(values) + \
                            ' do not contain any trx_results.')'''
                else:
                    raise TypeError('Not all inputs are integers')
            else:
                raise ValueError('No gtm_ids were supplied.')
        else:
            raise TypeError('Input is not a list or tuple')

    def get_trx_dlp(self, gtm_ids):
        """Querying TRX_deelproeven"""
        if isinstance(gtm_ids, (list, tuple, pd.Series)):
            if len(gtm_ids) > 0:
                if all(isinstance(x, (int)) for x in gtm_ids):
                    values = list(gtm_ids)
                    chunks = [values[x:x+1000]
                              for x in range(0, len(values), 1000)]
                    df_list = []
                    for chunk in chunks:
                        values = chunk
                        bindValues = [':' + str(i + 1)
                                      for i in range(len(values))]
                        query = 'SELECT * FROM bis_trx_dlp WHERE gtm_id IN ({})'.format(
                            ','.join(bindValues))
                        fetched, description = self.fetch(query, values)
                        if(len(fetched) > 0):
                            trx_dlp = pd.DataFrame(fetched)
                            colnames = [desc[0] for desc in description]
                            trx_dlp.columns = colnames
                            trx_dlp.loc[:, 'EPS50':] = trx_dlp.loc[:,
                                                                   'EPS50':].apply(pd.to_numeric)
                            df_list.append(trx_dlp)
                    trx_dlp_all = pd.concat(df_list, ignore_index=True)
                    if trx_dlp_all.empty is False:
                        return trx_dlp_all
                    '''else:
                        print('These selected geomonsters: ' + str(values) + \
                            ' do not contain any trx_dlp.')'''
                else:
                    raise TypeError('Not all inputs are integers')
            else:
                raise ValueError('No gtm_ids were supplied.')
        else:
            raise TypeError('Input is not a list or tuple')

    def get_trx_dlp_result(self, gtm_ids):
        """Querying TRX_dlp_results"""
        if isinstance(gtm_ids, (list, tuple, pd.Series)):
            if len(gtm_ids) > 0:
                if all(isinstance(x, (int)) for x in gtm_ids):
                    values = list(gtm_ids)
                    chunks = [values[x:x+1000]
                              for x in range(0, len(values), 1000)]
                    df_list = []
                    for chunk in chunks:
                        values = chunk
                        bindValues = [':' + str(i + 1)
                                      for i in range(len(values))]
                        query = 'SELECT * FROM bis_trx_dlp_result WHERE gtm_id IN ({})'.format(
                            ','.join(bindValues))
                        fetched, description = self.fetch(query, values)
                        if(len(fetched) > 0):
                            trx_dlp_result = pd.DataFrame(fetched)
                            colnames = [desc[0] for desc in description]
                            trx_dlp_result.columns = colnames
                            trx_dlp_result.rename(
                                columns={'TPR_EA': 'EA'}, inplace=True)
                            trx_dlp_result.loc[:, 'EA':] = trx_dlp_result.loc[:, 'EA':].apply(
                                pd.to_numeric)
                            df_list.append(trx_dlp_result)
                    trx_dlp_result_all = pd.concat(df_list, ignore_index=True)
                    if trx_dlp_result_all.empty is False:
                        return trx_dlp_result_all
                    '''else:
                        print('These selected boring(en): ' + str(values) + \
                            ' do not contain any trx_results.')'''
                else:
                    raise TypeError('Not all inputs are integers')
            else:
                raise ValueError('No gtm_ids were supplied.')
        else:
            raise TypeError('Input is not a list or tuple')

    @staticmethod
    def select_on_ea(trx_result, ea=2):
        """Filter on ea/strain"""
        if isinstance(trx_result, pd.DataFrame):
            new_trx_result_ea = trx_result[ea == trx_result.EA]
            return new_trx_result_ea
        else:
            raise TypeError('No pandas dataframe was supplied')

    def get_average_per_ea(self, df_trx_result, ea=5):
        """Calculating averages and standard deviations on TRX_results"""
        if isinstance(df_trx_result, pd.DataFrame):
            df_trx_temp = self.select_on_ea(df_trx_result, ea)
            mean_coh = round(np.mean(df_trx_temp['COH']), 1)
            mean_fi = round(np.mean(df_trx_temp['FI']), 1)
            std_coh = round(np.std(df_trx_temp['COH']), 1)
            std_fi = round(np.std(df_trx_temp['FI']), 1)
            N = int(len(df_trx_temp.index))
            return mean_fi, std_fi, mean_coh, std_coh, N
        else:
            raise TypeError('No pandas dataframe was supplied.')

    def get_least_squares(
        self,
        df_trx_dlp_result,
        plot_name='Lst_Sqrs_name',
        ea=2,
        save_plot=True
    ):
        """Creating least square fits on TRX_dlp_results"""

        df = self.select_on_ea(df_trx_dlp_result, ea)
        data_full = (df.P, df.Q)
        # Begin Least Squares fitting van een 'linear regression'
        x, y = data_full
        N = len(x)
        x_m = np.mean(x)
        x_quadm = np.sum(x*x)/N
        y_m = np.mean(y)
        yx_quadm = np.sum(x*y)/N

        a = (yx_quadm - y_m*x_m)/(x_quadm - x_m**2)  # Hellings Coefficient
        b = y_m - a*x_m   # Start Coefficent/cohesie
        alpha = np.arctan(a)
        fi = np.arcsin(a)
        coh = b/np.cos(fi)

        def func(a, b, x):
            return a*x + b

        y_res = y - func(a, b, x)

        E = np.sum(y_res**2)  # Abs. Squared Error
        E_per_n = E/N  # Mean Squared Error
        # Relative Squared Error average for all points
        eps = np.mean(y_res**2)/np.mean(y**2)
        # Einde Least Squares fitting

        if save_plot:
            dlp1, dlp2, dlp3 = df[(df.TDP_DEELPROEF_NUMMER == 1)], df[(
                df.TDP_DEELPROEF_NUMMER == 2)], df[(df.TDP_DEELPROEF_NUMMER == 3)]
            data_colors = ((dlp1.P, dlp1.Q, dlp1.GTM_ID), (dlp2.P,
                                                           dlp2.Q, dlp2.GTM_ID), (dlp3.P, dlp3.Q, dlp3.GTM_ID))
            colors = ('red', 'green', 'blue')
            dlp_label = ('dlp 1', 'dlp 2', 'dlp 3')

            fig = plt.figure(figsize=(14, 7))
            gs = fig.add_gridspec(2, 3, width_ratios=[2, 1, 2])
            ax = fig.add_subplot(gs[0:, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[1, 1])
            ax4 = fig.add_subplot(gs[0:, 2], sharex=ax)
            # Plotten verschillende deelproeven
            txt_annot = False
            for data, color, lab in zip(data_colors, colors, dlp_label):
                x2, y2, gtm_ids = data
                y_res = y2 - func(a, b, x2)
                ax.scatter(x2, y2, alpha=0.8, c=color,
                           edgecolors='none', s=30, label=lab)
                ax4.scatter(x2, y_res, alpha=0.8, c=color,
                            edgecolors='none', s=30, label=lab)
                for i in x2.index:
                    if y_res[i]**2 > 3*E_per_n:
                        txt_annot = True
                        ax4.annotate(gtm_ids[i], xy=(
                            x2[i], y_res[i]), xycoords='data', weight='bold')

            ax.plot([min(x), max(x)], [func(a, b, min(x)), func(a, b, max(x))], c='black', label='least squares fit')
            text = r'Line: $\tau = \sigma_n $* tan( ' + str(round(alpha, 3)) + r' ) + ' + str(round(b, 2)) \
                + '\n' + r'$\alpha=' + str(round(alpha, 3)) + r', a=' + str(round(b, 2)) + r', \phi= $' + str(round(np.degrees(fi), 1)) + '\u00B0, C=' + str(round(coh, 2))\
                + '\n' + 'Sqr. Error: ' + str(round(E, 1))\
                + '\n' + 'Mean Sqr. Error: ' + str(round(E_per_n, 2))\
                + '\n' + 'Mean Error: ' + str(round(np.sqrt(E_per_n), 2))\
                + '\n' + 'Mean Rel. Sqr. Err.: ' + str(round(eps*100, 2)) + ' %'\
                + '\n' + 'N: ' + str(N)
            at = offsetbox.AnchoredText(text, loc='lower right', frameon=True)
            at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax.add_artist(at)
            if txt_annot:
                at2 = offsetbox.AnchoredText(
                    'GTM_ID', loc='upper left', frameon=True, prop=dict(fontweight='bold'))
                at2.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
                ax4.add_artist(at2)

            ax.set_title(plot_name)
            ax.legend(loc=2)
            ax.set_xlabel(r'$\sigma_n$ Normaalspanning')
            ax.set_ylabel(r'$\tau$ Schuifspanning')
            ax.set_xlim(xmin=0)
            ax.set_ylim(ymin=0)

            ax2.hist(x, round(N/4))
            ax3.hist(y, round(N/4))
            ax2.set_title(r'Histogrammen van $\sigma_n$ en $\tau$')
            ax2.set_ylabel('N')
            ax3.set_ylabel('N')
            ax2.set_xlabel(r'$\sigma_n = \frac{ \sigma_1 + \sigma_3 }{2}$')
            ax3.set_xlabel(r'$\tau = \frac{ \sigma_1 - \sigma_3 }{2}$')

            ax4.set_title('Residual Plot')
            ax4.set_ylabel('Residual Schuifspanning $\tau_r$')
            ax4.set_xlabel(r'$\sigma_n$ Normaalspanning')

            plt.tight_layout()
        else:
            fig = None
        return round(np.degrees(fi), 1), round(coh, 1), round(E), round(E_per_n, 1), round(eps*100, 1), N, fig

    def get_sdp(self, gtm_ids):
        """Querying compression tests\\samendrukkingsproeven"""
        if isinstance(gtm_ids, (list, tuple, pd.Series)):
            if len(gtm_ids) > 0:
                if all(isinstance(x, (int)) for x in gtm_ids):
                    values = list(gtm_ids)
                    chunks = [values[x:x+1000]
                              for x in range(0, len(values), 1000)]
                    df_list = []
                    for chunk in chunks:
                        values = chunk
                        bindValues = [':' + str(i + 1)
                                      for i in range(len(values))]
                        query = 'SELECT * FROM bis_sdp WHERE gtm_id IN ({})'.format(
                            ','.join(bindValues))
                        fetched, description = self.fetch(query, values)
                        if(len(fetched) > 0):
                            sdp_df = pd.DataFrame(fetched)
                            colnames = [desc[0] for desc in description]
                            sdp_df.columns = colnames
                            sdp_df.loc[:, 'VOLUMEGEWICHT_DROOG':] = \
                                sdp_df.loc[:, 'VOLUMEGEWICHT_DROOG':].apply(
                                    pd.to_numeric)
                            df_list.append(sdp_df)
                    sdp_df_all = pd.concat(df_list, ignore_index=True)
                    if sdp_df_all.empty is False:
                        return sdp_df_all
                    '''else:
                        raise ValueError('These selected boring(en): ' + str(values) + \
                            ' do not contain any SDP_Proeven.')'''
                else:
                    raise TypeError('Not all inputs are integers')
            else:
                raise ValueError('No gtm_ids were supplied.')
        else:
            raise TypeError('Input is not a list or tuple')

    def get_sdp_result(self, gtm_ids):
        """Querying sdp_results"""
        if isinstance(gtm_ids, (list, tuple, pd.Series)):
            if len(gtm_ids) > 0:
                if all(isinstance(x, (int)) for x in gtm_ids):
                    values = list(gtm_ids)
                    chunks = [values[x:x+1000]
                              for x in range(0, len(values), 1000)]
                    df_list = []
                    for chunk in chunks:
                        values = chunk
                        bindValues = [':' + str(i + 1)
                                      for i in range(len(values))]
                        query = 'SELECT * FROM bis_sdp_resultaten WHERE gtm_id IN ({})'.format(
                            ','.join(bindValues))
                        fetched, description = self.fetch(query, values)
                        if(len(fetched) > 0):
                            sdp_result_df = pd.DataFrame(fetched)
                            colnames = [desc[0] for desc in description]
                            sdp_result_df.columns = colnames
                            sdp_result_df.loc[:, 'LOAD':] = \
                                sdp_result_df.loc[:, 'LOAD':].apply(
                                    pd.to_numeric)
                            df_list.append(sdp_result_df)
                    sdp_result_df_all = pd.concat(df_list, ignore_index=True)
                    if sdp_result_df_all.empty is False:
                        return sdp_result_df_all
                    '''else:
                        raise ValueError('These selected boring(en): ' + str(values) + \
                            ' do not contain any SDP_results.')'''
                else:
                    raise TypeError('Not all inputs are integers')
            else:
                raise ValueError('No gtm_ids were supplied.')
        else:
            raise TypeError('Input is not a list or tuple')

    # NOT USED
    def join_trx_with_trx_results(self, gtm_ids, proef_type='CD'):
        if isinstance(gtm_ids, (list, tuple, pd.Series)):
            if len(gtm_ids) > 0:
                if all(isinstance(x, (int)) for x in gtm_ids):

                    values = tuple(gtm_ids)
                    bindValues = [':' + str(i + 1) for i in range(len(values))]
                    proef_type = list(proef_type)
                    bindProef = [':p' + str(i + 1)
                                 for i in range(len(proef_type))]
                    bindAll = bindValues + bindProef
                    values = values + proef_type
                    bindDict = dict(zip(bindAll, values))
                    query = 'SELECT bis_trx_proeven.gtm_id, volumegewicht_droog, volumegewicht_nat, ' \
                        + 'watergehalte, terreinspanning, bezwijksnelheid, trx_result.trx_volgnr, ea, '\
                        + 'coh, fi FROM bis_trx_proeven ' \
                        + 'INNER JOIN bis_trx_proef_result ON bis_trx_proeven.gtm_id = bis_trx_proef_result.gtm_id '\
                        + 'AND bis_trx_proeven.trx_volgnr = bis_trx_proef_result.trx_volgnr '\
                        + 'WHERE proef_type = ({}) AND bis_trx_proeven.gtm_id IN ({})'.format(
                            ','.join(bindProef), ','.join(bindValues))
                    fetched, description = self.fetch(query, bindDict)
                    if(len(fetched) > 0):
                        trx_df = pd.DataFrame(fetched)
                        colnames = [desc[0] for desc in description]
                        trx_df.columns = colnames
                        trx_df.VOLUMEGEWICHT_NAT = trx_df.VOLUMEGEWICHT_NAT.astype(
                            float)
                        return trx_df
                    else:
                        raise ValueError(
                            'These selected boring(en): ' + str(values) + ' do not contain any trx + trx_result.')
                else:
                    raise TypeError('not all inputs are ints')
            else:
                raise ValueError('No gtm_ids were supplied.')
        else:
            raise TypeError('Input is not a list or tuple')
