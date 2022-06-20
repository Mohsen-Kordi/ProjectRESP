
#Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import fft, fftpack
import scipy.signal as sg

###############################################################################
###############################  Functions  ###################################
###############################################################################


def load_df(_db , _deliminator, **kwargs):
    
    """
    load_df returns useful columns of data file.

    Parameters
    ----------
    _db : String
        Data file name.
    _deliminator : String
        Deliminator of the columns in _db.
    _col : List[String], optional
        List of columns which are returned.
        The default is ['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ', 'MagnX', 'MagnY', 'MagnZ'].
        'all' means do not drop any column.

    Returns
    -------
    _df : pandas.Dataframe
        Returns columns from _db file.
    _fs : integer
        Sampling frequency.
    _timestamp : [pandas.datetime]
        Sampling start and end time in a 2x1 list.

    """
    
    _col = kwargs.get('col' , ['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ', 'MagnX', 'MagnY', 'MagnZ'])
    
    _df = pd.read_csv(_db, delimiter = _deliminator)
    _fs = _df['Log Freq'][0]
    _timestamp = pd.to_datetime([_df.iloc[0]['Timestamp'] , _df.iloc[-1]['Timestamp']] , unit='s')
    if _col != 'all':
        _df = _df[_col]
    return _df , _fs , _timestamp






def tm_stamp(_fs , _timestamp):
    
    """
    Prints some information from data file.

    Parameters
    ----------
    _fs : int
        Sampling Frequency.
    _timestamp : list[Start time, end Time]
        Samples timestamp information.

    Returns
    -------
    None.

    """
    
    print(f'Sampling frequency is {_fs} Hz.\n')
    print(f'Sampling start and end times are {_timestamp[0]} and {_timestamp[1]} respectively.')
    print(f'Duration of sampling is {timestamp[1]-timestamp[0]} s.')    













def plt_df(_df, _fs, **kwargs):
    """
    Plots all columns of data.

    Parameters
    ----------
    _fs : int
        sampling frequency.
    _df : pandas.DataFrame
        data set to plot.
    **kwargs : TYPE
        Optional arguments as follow:
            y : list[string]
                List of y labels. The deault is _df.columns.
            x : string
                x label.
            c : list[color names]
                List of color for each row. The default is ['seagreen', 'seagreen', 'seagreen', 'royalblue', 'royalblue', 'royalblue', 'mediumpurple', 'mediumpurple', 'mediumpurple']
            w : int [start , stop]
                Start and end of selected window. The default is None (no window to show)
            fign : int [#rows , #columns]
                Number of subpolots. The default is [3 , 3].
            figs : int [width , hight]
                Figures' size. The default is [16 , 12].
            g : Bool
                Grid status. The default is True.

    Returns
    -------
    None.

    """
    _ylabels = kwargs.get('y', _df.columns)
    _xlabel = kwargs.get('x','Time(s)')
    _color = kwargs.get('c', ['seagreen', 'royalblue','mediumpurple'])
#    _color = kwargs.get('c', ['seagreen', 'seagreen', 'seagreen', 'royalblue', 'royalblue', 'royalblue', 'mediumpurple', 'mediumpurple', 'mediumpurple'])
    _wnd = kwargs.get('w', None)
    _fign = kwargs.get('fign', [3 , 3])
    _figs = kwargs.get('figs', [16 , 12])
    _grid = kwargs.get('g',True)
    
    x = np.linspace(0 , _df.shape[0] / _fs , _df.shape[0])
    
    fig, axs = plt.subplots(_fign[0] , _fign[1], figsize = (_figs[0], _figs[1]))
    for i, ax in enumerate(axs.flat):
        ax.plot(x, _df[_ylabels[i]], color = _color[i//_fign[0]])
        ax.set_xlabel(_xlabel)
        ax.set_ylabel(_ylabels[i])
        ax.grid(_grid)
        if _wnd != None:
            ax.axvline(x = _wnd[0], color = 'r')
            ax.axvline(x = _wnd[1], color = 'r')



def calib_df(_df , _col  , **kwargs):
    """
    This function calibrates dataframe usoing calibration coefficients and offset values.

    Parameters
    ----------
    _df : pandas.DataFrame
        Main data.
    _col : TYPE
        DESCRIPTION.

    **kwargs : TYPE
        Optional arguments as follow:
            c : np.array
                List of calibration coefficients. The default is np.eye(len(_col)).
            o : np.array
                List of offset values. The default is np.zeros(len(_col)).

    Returns
    -------
    _df : TYPE
        DESCRIPTION.

    """

    _calib = kwargs.get('c', np.eye(len(_col)))
    _offset = kwargs.get('o', np.zeros(len(_col)))
    
    _df[_col] = np.dot(_df[_col] , _calib.T) + _offset
    return _df

    




def statistic_df(_df):
    """
    Returns statistical and correlation analysis results of data.

    Parameters
    ----------
    _df : pandas.DataFrame
        Data to be analysed.

    Returns
    -------
    _statistic : pandas.DataFrame
        Statistical analysis result.
    _corr : pandas.DataFrame
        Correlation analysis result.

    """
    
    
    
    # _statistic = pd.DataFrame({'Mean': _df.mean(),
    #                       'Median': _df.median(),
    #                       'Variance': _df.var(),
    #                       'Standard deviation': _df.std(),
    #                       '25% percentile': _df.quantile(q = 0.25),
    #                       '75% percentile': df.quantile(q = 0.75)})
    
    
    _statistic = _df.describe()
    _corr = _df.corr()
    
    return _statistic , _corr



def Fanalysis_df(_df, _fs, **kwargs):
    """
    Calculates FFT of the given dataframe.

    Parameters
    ----------
    _df : pandas.DataFrame
        Input data to calculate fft on it.
    _fs : int
        Sampling frequency.
    **kwargs : string
        Optional arguments as follow:
            rb : Boolean
                If True the bias will be removed (set power of freq=0 to zero). Default is True.
            pf : Boolean
                If True negetive frequencies and their related FFT removed from return result. Default is False.

    Returns
    -------
    _fft_df : pandas.DataFrame
        FFT of input dataframe.
    _fft_power : pandas.DataFrame
        Power of FFT of input dataframe.
    _freq : list
        List of frequency of FFT.

    """
    
    _remove_bias = kwargs.get('rb' , True)
    _pos_freq = kwargs.get('pf' , False)
    
    
    _fft_df = _df.apply(np.fft.fft)
    _freq = fft.fftfreq(_fft_df.shape[0] , d=1/_fs)   
    
    if _remove_bias:
        _fft_df.loc[0] = 0

    if _pos_freq:
        _fft_df = _fft_df.drop(_fft_df[_freq <= 0].index)
        _freq = _freq[np.where(_freq > 0)]    

    _fft_power = _fft_df.apply(np.abs)
    
   
    return _fft_df, _fft_power, _freq







###############################################################################
#############################  Main Analysis  #################################
###############################################################################

df , fs , timestamp = load_df('center_sternum.txt', "\t")

print('Sampling frequency: ' , fs ,' Hz\n')
print('Sampling start time: {}\nSampling end time: {}\nDuration: {}.'.format(timestamp[0],timestamp[1],timestamp[1]-timestamp[0]))
print('Raw data:\n\n' , df)

# Configuration Information

Acc = ['AccX','AccY','AccZ']
Magn = ['MagnX','MagnY','MagnZ']
Gyro = ['GyroX','GyroY','GyroZ']



gyro_offset = np.array([-2.242224,2.963463,-0.718397])

acc_cal = np.array([[1.000966,-0.002326418,-0.0006995499],
                    [-0.002326379,0.9787045,-0.001540918],
                    [-0.0006995811,-0.001540928,1.00403]])
acc_offset = np.array([-3.929942,-13.74679,60.67546])

magn_cal = np.array([[0.9192851,-0.02325168,0.003480837],
                   [-0.02325175,0.914876,0.004257396],
                   [0.003481006,0.004257583,0.8748001]])
magn_offset = np.array([-95.67974,-244.9142,17.71132])







plt_df(df , fs , w = [10 , 70])


df_sel = df[2000:14000].copy()
print('Selected part of data (60 sec.):\n\n' , df_sel)


df_sel = calib_df(df_sel , Acc , c = acc_cal , o = acc_offset)
df_sel = calib_df(df_sel , Magn , c = magn_cal , o = magn_offset)
df_sel = calib_df(df_sel , Gyro , o = gyro_offset)

print('Calibrated data:\n\n' , df)

plt_df(df_sel , fs , fign=[9,1] , figs=[20,16])


statistic_df , corr_df = statistic_df(df_sel)

print('Statistical Analysis:\n\n', statistic_df)



print('corroleation Coefficients:\n\n', corr_df)


fft_df  = Fanalysis_df(df_sel , fs)
print(fft_df)

fft_df_power = pd.DataFrame(np.abs(fft_df) , columns=fft_df.columns)

plt_df(fft_df_power , fs , fign=[9,1] , figs=[16,18] , x='Frequency (Hz)')


fft_df_real = pd.DataFrame(np.real(fft_df) , columns=fft_df.columns)
fft_df_imag = pd.DataFrame(np.imag(fft_df) , columns=fft_df.columns)

plt_df(fft_df_real , fs , fign=[9,1] , figs=[16,18] , x='Frequency (Hz)')
plt_df(fft_df_imag , fs , fign=[9,1] , figs=[16,18] , x='Frequency (Hz)')




###############################################################################
   
fft_freq = fft.fftfreq(fft_df.shape[0] , d=1/fs)
    
fft_filter = fft_df.copy()

fft_freq <1
fft_freq.max()
fft_filter[fft_freq>1]=0
fft_filter
fft_filter[fft_freq<0]=0

fft_df_real = pd.DataFrame(np.real(fft_filter) , columns=fft_df.columns)
fft_df_imag = pd.DataFrame(np.imag(fft_filter) , columns=fft_df.columns)

plt_df(fft_df_real , fs , fign=[9,1] , figs=[16,18] , x='Frequency (Hz)')
plt_df(fft_df_imag , fs , fign=[9,1] , figs=[16,18] , x='Frequency (Hz)')

sig_filter = pd.DataFrame(np.real(fft.ifft(fft_filter)),columns=df_sel.columns)

plt_df(sig_filter , fs , fign=[9,1] , figs=[16,18] , x='Frequency (Hz)')


