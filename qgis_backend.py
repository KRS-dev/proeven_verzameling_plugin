'''
Script: qgis_backend.py
Repository: https://github.com/KRS-dev/proeven_verzameling
Author: Kevin Schuurman
E-mail: kevinschuurman98@gmail.com
Summary: Base functions for qgis_frontend.py, Querying the data from the proeven verzameling database.
'''
import pandas as pd
import numpy as np
import psycopg2 as psy
import matplotlib.pyplot as plt
import matplotlib.offsetbox as offsetbox

# Database Connection
def fetch (query, data):
    ## Using a PostgreSQL database
    with psy.connect(
        host = "localhost",
        database = "bis",
        user = "postgres",
        password = "admin"
        ) as dbcon:
    ## Using an Oracle database:
        '''
        with cora.connect(
        user = "",
        password = "",
        dsn = ""*
            ) as dbcon:
        # *Can be a: 
        # 1. Oracle Easy Connect string
        # 2. Oracle Net Connect Descriptor string
        # 3. Net Service Name mapping to connect description
        '''

        cur = dbcon.cursor()
        cur.execute(query, data)
        # Print('SQL QUERY: \n' + query )
        fetched = cur.fetchall()
        description = cur.description
        return fetched, description

# Getting the loc_id's from the Qgislayer
def get_loc_ids(QgisLayer):
    loc_ids = []
    features = QgisLayer.selectedFeatures()

    if len(features) > 0:
        print(str(len(features)) + ' points were selected')
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
def get_meetpunten( loc_ids ):
    if isinstance( loc_ids, ( list, tuple, pd.Series) ):
        if len( loc_ids ) > 0:
            if( all( isinstance(x, int) for x in loc_ids )):
                values = tuple(loc_ids)
                query = 'SELECT * FROM graf_loc_aanduidingen '\
                    + 'INNER JOIN meetpunten ON meetpunten.mpt_id = graf_loc_aanduidingen.loc_id '\
                    + 'WHERE graf_loc_aanduidingen.loc_id IN %s'
                fetched, description = fetch(query, (values,))

                if (0 < len(fetched)):
                    meetp_df = pd.DataFrame(fetched)
                    colnames = [desc[0] for desc in description]
                    meetp_df.columns = colnames
                    meetp_df.gds_id = meetp_df.gds_id.fillna(0)
                    meetp_df.gds_id = pd.to_numeric(
                        meetp_df.gds_id, downcast='integer')
                    return meetp_df
                else:
                    raise ValueError(
                        'These selected geometry points do not contain valid loc_ids: ' + str(values))
            else:
                raise TypeError('not all inputs are integers')
        else:
            raise ValueError('No bor_ids were supplied.')
    else:
        raise TypeError('Input is not a list or tuple')

# Querying geodossiers
def get_geo_dossiers(gds_ids):
    if isinstance(gds_ids, (list, tuple, pd.Series)):
        if len(gds_ids) > 0:
            if(all(isinstance(x, int) for x in gds_ids)):
                values = tuple( gds_ids )
                query = 'SELECT * FROM geo_dossiers WHERE gds_id IN %s'
                fetched, description = fetch(query, (values,))
                if ( 0 < len( fetched )):
                    geod_df = pd.DataFrame(fetched)
                    colnames = [ desc[0] for desc in description ]
                    geod_df.columns = colnames
                    return geod_df
                else:
                    raise ValueError('The selected gds_ids: ' + str(values) + \
                    ' do not contain any geodossiers.')
            else:
                raise TypeError('not all inputs are integers')
        else:
            raise ValueError('No bor_ids were supplied.') 
    else:
        raise TypeError('Input is not a list or tuple')    

# Querying geotechnische monsters
def get_geotech_monsters( bor_ids ):
    if isinstance( bor_ids, ( list, tuple, pd.Series) ):
        if len( bor_ids ) > 0:
            if( all( isinstance( x, (int)) for x in bor_ids )):
                values = tuple(bor_ids)
                query = 'SELECT * FROM geotech_monsters WHERE bor_id IN %s'
                fetched, description = fetch(query, (values,))
                if( len( fetched ) > 0 ):
                    g_mon_df = pd.DataFrame( fetched )
                    colnames = [ desc[0] for desc in description ]
                    g_mon_df.columns = colnames
                    g_mon_df['z_coordinaat_laag'] = pd.to_numeric( g_mon_df['z_coordinaat_laag'] )
                    return g_mon_df
                else:
                    raise ValueError('These selected boring(en): ' + str(values) + \
                    ' do not contain any triaxiaal proeven.')
            else:
                raise TypeError('not all inputs are integers')
        else:
            raise ValueError('No bor_ids were supplied.') 
    else:
        raise TypeError('Input is not a list or tuple')

# Filter on height of the Geotechnical monsters        
def select_on_z_coord( g_mon_df, zmax, zmin ):
    if isinstance( g_mon_df, pd.DataFrame ):
        new_g_mon_df = g_mon_df[( zmax > g_mon_df.z_coordinaat_laag ) & ( g_mon_df.z_coordinaat_laag > zmin )]
        return new_g_mon_df
    else:
         raise TypeError('No pandas dataframe was supplied')

# Querying TRX_proeven
def get_trx( gtm_ids, proef_type = ('CD') ):
    if isinstance( gtm_ids, ( list, tuple, pd.Series ) ):
        if all( any( x == i for i in ('CU','CD','UU') ) for x in proef_type ):
            if len( gtm_ids ) > 0:
                if all( isinstance( x, ( int )) for x in gtm_ids ):
                    values = tuple(gtm_ids)
                    proef_type = tuple(proef_type)
                    query = 'SELECT * FROM trx WHERE proef_type IN %s AND gtm_id IN %s'
                    fetched, description = fetch(query, (proef_type, values))
                    if( len( fetched ) > 0 ):
                        trx_df = pd.DataFrame(fetched)
                        colnames = [desc[0] for desc in description]
                        trx_df.columns = colnames
                        trx_df[['volumegewicht_droog','volumegewicht_nat','watergehalte','terreinspanning','bezwijksnelheid']] = \
                        trx_df[['volumegewicht_droog','volumegewicht_nat','watergehalte','terreinspanning','bezwijksnelheid']].apply(pd.to_numeric)
                        trx_df.volumegewicht_nat = trx_df.volumegewicht_nat.astype(float)
                        return trx_df
                    else:
                        print('These selected boring(en): ' + str(values) + \
                            ' do not contain any triaxiaal proeven with proef_type: ' + str(proef_type))
                else:
                    raise TypeError('not all inputs are integers')
            else:
                raise ValueError('No gtm_ids were supplied.')
        else:
            raise TypeError('Only CU, CD and UU or a combination are allowed as proef_type')
    else:
        raise TypeError('Input is not a list or tuple')

# Filter on Volumetric weight      
def select_on_vg( trx_df, Vg_max = 20, Vg_min = 17, soort ='nat' ):
    #Volume gewicht y in kN/m3
    if isinstance(trx_df, pd.DataFrame):
        if soort == 'nat':
            new_trx_df = trx_df[(Vg_max >= trx_df.volumegewicht_nat) & (trx_df.volumegewicht_nat >= Vg_min)]
        elif soort == 'droog':
            new_trx_df = trx_df[(Vg_max >= trx_df.volumegewicht_droog) & (trx_df.volumegewicht_droog >= Vg_min)]
        else:
            raise TypeError('\'' + soort + '\' is not allowed as argument for soort,\
                 only \'nat\' and \'droog\' are allowed.')
        return new_trx_df
    else:
        raise TypeError('No pandas dataframe was supplied')

# Querying TRX_results
def get_trx_result( gtm_ids ):
    if isinstance(gtm_ids, ( list, tuple, pd.Series ) ):  
        if len( gtm_ids ) > 0:    
            if all(isinstance(x, (int)) for x in gtm_ids):
                values = tuple( gtm_ids )
                query = 'SELECT * FROM trx_result WHERE gtm_id IN %s' 
                fetched, description = fetch(query, (values,))
                if( len( fetched ) > 0 ):
                    trx_result_df = pd.DataFrame(fetched)
                    colnames = [desc[0] for desc in description]
                    trx_result_df.columns = colnames
                    trx_result_df[['ea','coh','fi']] = trx_result_df[['ea','coh','fi']].apply(pd.to_numeric)
                    return trx_result_df
                else:
                    print('These selected boring(en): ' + str(values) + \
                        ' do not contain any trx_results.')
            else:
                raise TypeError('Not all inputs are integers')
        else:
            raise ValueError('No gtm_ids were supplied.')
    else:
         raise TypeError('Input is not a list or tuple')   

# Querying TRX_deelproeven
def get_trx_dlp( gtm_ids ):
    if isinstance(gtm_ids, ( list, tuple, pd.Series ) ):   
        if len( gtm_ids ) > 0: 
            if all(isinstance(x, (int)) for x in gtm_ids):
                values = tuple( gtm_ids )
                query = 'SELECT * FROM trx_dlp WHERE gtm_id IN %s' 
                fetched, description = fetch(query, (values,))
                if( len( fetched ) > 0 ):
                    trx_dlp = pd.DataFrame(fetched)
                    colnames = [desc[0] for desc in description]
                    trx_dlp.columns = colnames
                    trx_dlp.loc[:,'eps50':] = trx_dlp.loc[:,'eps50':].apply(pd.to_numeric)
                    return trx_dlp
                else:
                    print('These selected geomonsters: ' + str(values) + \
                        ' do not contain any trx_dlp.')
            else:
                raise TypeError('Not all inputs are integers')
        else:
            raise ValueError('No gtm_ids were supplied.')
    else:
         raise TypeError('Input is not a list or tuple')   

# Querying TRX_dlp_results
def get_trx_dlp_result( gtm_ids ):
    if isinstance(gtm_ids, ( list, tuple, pd.Series ) ): 
        if len( gtm_ids ) > 0:
            if all(isinstance(x, (int)) for x in gtm_ids):
                values = tuple( gtm_ids )
                query = 'SELECT * FROM trx_dlp_result WHERE gtm_id IN %s' 
                fetched, description = fetch(query, (values,))
                if( len( fetched ) > 0 ):
                    trx_dlp_result = pd.DataFrame(fetched)
                    colnames = [desc[0] for desc in description]
                    trx_dlp_result.columns = colnames
                    trx_dlp_result.rename(columns={'tpr_ea':'ea'},inplace=True)
                    trx_dlp_result.loc[:,'ea':] = trx_dlp_result.loc[:,'ea':].apply(pd.to_numeric)
                    return trx_dlp_result
                else:
                    print('These selected boring(en): ' + str(values) + \
                        ' do not contain any trx_results.')
            else:
                raise TypeError('Not all inputs are integers')
        else:
            raise ValueError('No gtm_ids were supplied.')
    else:
         raise TypeError('Input is not a list or tuple')   

# Filter on ea/strain        
def select_on_ea( trx_result, ea = 2 ):
     if isinstance(trx_result, pd.DataFrame): 
         new_trx_result_ea = trx_result[ ea == trx_result.ea ]
         return new_trx_result_ea
     else:
         raise TypeError('No pandas dataframe was supplied')

# Calculating averages and standard deviations on TRX_results
def get_average_per_ea( df_trx_result, ea = 5):
    if isinstance( df_trx_result, pd.DataFrame ):
        df_trx_temp = select_on_ea(df_trx_result, ea)
        mean_coh = round( np.mean( df_trx_temp['coh'] ), 1 )
        mean_fi = round( np.mean( df_trx_temp['fi'] ), 1 )
        std_coh = round( np.std( df_trx_temp['coh'] ),1 )
        std_fi = round( np.std( df_trx_temp['fi'] ), 1 )
        N = int(len(df_trx_temp.index))
        return mean_fi,std_fi,mean_coh,std_coh,N
    else:
        raise TypeError('No pandas dataframe was supplied.')

# Creating least square fits on TRX_dlp_results
def get_least_squares( 
    df_trx_dlp_result, 
    plot_name = 'Lst_Sqrs_name', 
    ea = 2, 
    show_plot = True
    ):
    df = select_on_ea( df_trx_dlp_result, ea)
    data_full = (df.p, df.q)
    ### Begin Least Squares fitting van een 'linear regression'
    x, y = data_full
    N = len(x)
    x_m = np.mean(x)
    x_quadm = np.sum(x*x)/N
    y_m = np.mean(y)
    yx_quadm = np.sum(x*y)/N

    a = (yx_quadm - y_m*x_m)/(x_quadm - x_m**2) # Hellings Coefficient
    b = y_m - a*x_m   # Start Coefficent/cohesie
    alpha = np.arctan(a) 
    fi = np.arcsin(a)
    coh = b/np.cos(fi)

    def func(a,b,x):
        return a*x + b

    y_res = y - func(a,b,x)

    E = np.sum(y_res**2) # Abs. Squared Error
    E_per_n = E/N # Mean Squared Error 
    eps = np.mean(y_res**2/y**2) # Normalised/Relative Error average for all points
    ### Einde Least Squares fitting

    
    if show_plot:
        dlp1, dlp2, dlp3 = df[(df.deelproef_nummer == 1)], df[(df.deelproef_nummer == 2)], df[(df.deelproef_nummer == 3)]
        data_colors = ((dlp1.p, dlp1.q, dlp1.gtm_id), (dlp2.p, dlp2.q, dlp2.gtm_id), (dlp3.p, dlp3.q, dlp3.gtm_id))
        colors = ('red', 'green', 'blue')
        dlp_label = ('dlp 1', 'dlp 2', 'dlp 3')

        fig = plt.figure(figsize=(14,7))
        gs = fig.add_gridspec(2,3,width_ratios = [2,1,2])
        ax = fig.add_subplot(gs[0:,0])
        ax2 = fig.add_subplot(gs[0,1])
        ax3 = fig.add_subplot(gs[1,1])
        ax4 = fig.add_subplot(gs[0:,2], sharex=ax)
        ## Plotten verschillende deelproeven
        txt_annot = False
        for data, color, lab in zip(data_colors, colors, dlp_label):
            x2, y2, gtm_ids = data
            y_res = y2 - func(a,b,x2)
            ax.scatter(x2, y2, alpha=0.8, c=color, edgecolors='none', s=30, label=lab)
            ax4.scatter(x2, y_res, alpha=0.8, c=color, edgecolors='none', s=30, label=lab)
            for i in x2.index:
                if y_res[i]**2 > 3*E_per_n:
                    txt_annot = True
                    ax4.annotate(gtm_ids[i], xy=(x2[i], y_res[i]), xycoords='data',weight='bold')
        
        ax.plot([min(x), max(x)], [func(a,b,min(x)), func(a,b,max(x))], c='black', label='least squares fit')
        text = r'Line: $\tau = \sigma_n $* tan( ' + str(round(alpha,3)) + r' ) + ' + str(round(b,2)) \
            + '\n' + r'$\alpha=' + str(round(alpha,3)) + r', a=' + str(round(b,2)) + r', \phi= $' + str(round(np.degrees(fi),1)) + '\u00B0, C=' + str(round(coh,2))\
            + '\n' + 'Squared Error: ' + str(round(E,1))\
            + '\n' + 'Mean Squared Error: ' + str(round(E_per_n,2))\
            + '\n' + 'Mean Error: ' + str(round(np.sqrt(E_per_n),2))\
            + '\n' + 'Mean Rel Error: ' + str(round(eps*100, 2)) + ' %'\
            + '\n' + 'N: ' + str(N)
        at = offsetbox.AnchoredText(text, loc='lower right', frameon=True)
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax.add_artist(at)
        if txt_annot:
            at2 = offsetbox.AnchoredText('GTM_ID', loc='upper left', frameon=True, prop=dict(fontweight='bold'))
            at2.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax4.add_artist(at2)

        ax.set_title(plot_name)
        ax.legend(loc=2)
        ax.set_xlabel(r'$\sigma_n$ Normaalspanning')
        ax.set_ylabel(r'$\tau$ Schuifspanning')
        ax.set_xlim(xmin=0)
        ax.set_ylim(ymin=0)

        ax2.hist(x,round(N/4))
        ax3.hist(y,round(N/4))
        ax2.set_title(r'Histogrammen van $\sigma_n$ en $\tau$')
        ax2.set_ylabel('N')
        ax3.set_ylabel('N')
        ax2.set_xlabel(r'$\sigma_n = \frac{ \sigma_1 + \sigma_3 }{2}$')
        ax3.set_xlabel(r'$\tau = \frac{ \sigma_1 - \sigma_3 }{2}$')

        ax4.set_title('Residual Plot')
        ax4.set_ylabel('Residual Schuifspanning $\tau_r$')
        ax4.set_xlabel(r'$\sigma_n$ Normaalspanning')

        plt.tight_layout()
        if show_plot:
            plt.show()
    return round(np.degrees(fi),1), round(coh,1), round(E), round(E_per_n,1), round(eps*100,1), N

# Querying compression tests\samendrukkingsproeven
def get_sdp( gtm_ids ):
    if isinstance(gtm_ids, ( list, tuple, pd.Series ) ): 
        if len( gtm_ids ) > 0:
            if all(isinstance(x, (int)) for x in gtm_ids):
                values = tuple( gtm_ids )
                query = 'SELECT * FROM sdp WHERE gtm_id IN %s'
                fetched, description = fetch(query, (values,))
                if( len( fetched ) > 0 ):
                    sdp_df = pd.DataFrame(fetched)
                    colnames = [desc[0] for desc in description]
                    sdp_df.columns = colnames
                    sdp_df.loc[:,'volumegewicht_droog':] = \
                        sdp_df.loc[:,'volumegewicht_droog':].apply(pd.to_numeric)
                    return sdp_df
                else:
                    print('These selected boring(en): ' + str(values) + \
                        ' do not contain any SDP_Proeven.')
            else:
                raise TypeError('Not all inputs are integers')
        else:
            raise ValueError('No gtm_ids were supplied.')
    else:
         raise TypeError('Input is not a list or tuple')   

# Querying sdp_results
def get_sdp_result( gtm_ids ):
    if isinstance(gtm_ids, ( list, tuple, pd.Series ) ):
        if len( gtm_ids ) > 0: 
            if all(isinstance(x, (int)) for x in gtm_ids):
                values = tuple( gtm_ids )
                query = 'SELECT * FROM sdp_result WHERE gtm_id IN %s'
                fetched, description = fetch(query, (values,))
                if( len( fetched ) > 0 ):
                    sdp_result_df = pd.DataFrame(fetched)
                    colnames = [desc[0] for desc in description]
                    sdp_result_df.columns = colnames
                    sdp_result_df.loc[:,'load':] = \
                        sdp_result_df.loc[:,'load':].apply(pd.to_numeric)
                    return sdp_result_df
                else:
                    print('These selected boring(en): ' + str(values) + \
                        ' do not contain any SDP_results.')
            else:
                raise TypeError('Not all inputs are integers')
        else:
            raise ValueError('No gtm_ids were supplied.')
    else:
         raise TypeError('Input is not a list or tuple')   
# NOT USED
def join_trx_with_trx_results( gtm_ids, proef_type = 'CD' ):
    if isinstance(gtm_ids, ( list, tuple, pd.Series ) ):
        if len(gtm_ids) > 0:
            if all( isinstance( x, ( int )) for x in gtm_ids ):

                values = tuple(gtm_ids)
                #values_str = '(' + ','.join(str(i) for i in values).strip(',') + ')'
                query = 'SELECT trx.gtm_id, volumegewicht_droog, volumegewicht_nat, ' \
                    + 'watergehalte, terreinspanning, bezwijksnelheid, trx_result.trx_volgnr, ea, '\
                    + 'coh, fi FROM trx ' \
                    + 'INNER JOIN trx_result ON trx.gtm_id = trx_result.gtm_id '\
                    + 'AND trx.trx_volgnr = trx_result.trx_volgnr '\
                    + 'WHERE proef_type = %s AND trx.gtm_id IN %s'
                fetched, description = fetch(query, (proef_type, values))
                if( len( fetched ) > 0 ):
                    trx_df = pd.DataFrame(fetched)
                    colnames = [desc[0] for desc in description]
                    trx_df.columns = colnames
                    trx_df.volumegewicht_nat = trx_df.volumegewicht_nat.astype(float)
                    return trx_df
                else:
                    raise ValueError('These selected boring(en): ' + str(values) + ' do not contain any trx + trx_result.')
            else:
                raise TypeError('not all inputs are ints')    
        else:
            raise ValueError('No gtm_ids were supplied.')
    else:    
        raise TypeError('Input is not a list or tuple')     
