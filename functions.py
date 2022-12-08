# Required function for running this assignment
# Written by Mehdi Rezvandehy


from tensorflow.keras.models import Sequential
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from sklearn.preprocessing import MinMaxScaler
from pandas import concat
import numpy as np
import pandas as pd
import math
from tensorflow.random import set_seed
from matplotlib.offsetbox import AnchoredText
from IPython.display import display, Math, Latex
from matplotlib.ticker import PercentFormatter
import matplotlib.pyplot as plt
from IPython.display import HTML
set_seed(455)

import itertools

def df_Series_to_Supervised(df, n_input=90, n_output=10, remove_nan=True):
    """
    Convert DataFrame time series to a supervised learning dataset.
    
    df: DataFrame time series
    n_input: Input observations as training features (X)
    n_output: Output observations target (y)
    remove_nan: drop missing values (NaN)
    
    """
    Vars=df.columns
    Clms, Nms = [],[]
    # input sequence (t-n, ... t-1)
    for i in range(n_input, 0, -1):
        Clms.append(df.shift(i))
        Nms+=[('Var%d(t-%d)' % (j+1, i)) for j in range(len(Vars))]
    # predict sequence (t, t+1, ... t+n)
    for i in range(0, n_output):
        Clms.append(df.shift(-i))
        if i == 0:
            Nms+= [('Var%d(t)' % (j+1)) for j in range(len(Vars))]
        else:
            Nms+=[('Var%d(t+%d)' % (j+1, i)) for j in range(len(Vars))]
    # concat all
    df_f = concat(Clms, axis=1)
    df_f.columns = Nms
    # drop rows with NaN values if needed
    if remove_nan:
        df_f.dropna(inplace=True)
    return df_f  

################################################################################

def time_series_forecst(n_input,n_output,nsim, model,X_train,clm='value',scale=True,
                        no='date',data_out=False,deep_learning=False,epochs=10,
                        activation='tanh',hl1=50,hl2=35,**kwargs):
    
    """
    Predict time series by suppervised learning
    
    n_input       : number of input features (should be number of samples in one complete series)
    n_output      : number of output targets
    nsim:         : number of simulations (realizations)
    model         : predictive algorithms 
    X_train       : training time series as Pandas DataFrame
    clm           : column name for the feature in X_train DataFrame
    no            : date for the feature in X_train DataFrame
    data_out      : True, it returns xtrain and ytrain
    deep_learning : deep_learning model that can be rnn, lstm
    epochs        : number of epochs for deep lerining
    hl1, hl2      : hidden layer 1 and 2 for deep learning
    activation    : activation function for deep learning
    **kwargs      : individual test sets (for example, test1=October, test2=November, test3=December...)
    """
    
    def dl_rnn(n_out,seed=42):
        """
        Recurrent Neural Network
        """
        np.random.seed(seed)
        tf.random.set_seed(seed)    
        model_rnn = keras.models.Sequential()
        model_rnn.add(keras.layers.SimpleRNN(hl1, activation=activation, input_shape=(None,1)))
        model_rnn.add(keras.layers.Dense(hl2, activation=activation))
        model_rnn.add(keras.layers.Dense(n_out))
        # Compile model
        model_rnn.compile(loss="mse", optimizer="RMSprop")
        return model_rnn    
    #
    def dl_lstm(n_out,seed=42):
        """
        LSTM (Long Short-Term Memory)
        """
        np.random.seed(seed)
        tf.random.set_seed(seed)    
        model_lstm = keras.models.Sequential()
        model_lstm.add(keras.layers.LSTM(hl1, activation=activation, input_shape=(None,1)))
        model_lstm.add(keras.layers.Dense(hl2, activation=activation))
        model_lstm.add(keras.layers.Dense(n_out))
        # Compile model
        model_lstm.compile(loss="mse", optimizer="RMSprop")
        return model_lstm       
    #    
    def dl_gru(n_out,seed=42):
        """
        GRU (Gated Recurrent Unit)
        """
        np.random.seed(seed)
        tf.random.set_seed(seed)    
        model_lstm = keras.models.Sequential()
        model_lstm.add(keras.layers.GRU(hl1, activation=activation, input_shape=(None,1)))
        model_lstm.add(keras.layers.Dense(hl2, activation=activation))
        model_lstm.add(keras.layers.Dense(n_out))
        # Compile model
        model_lstm.compile(loss="mse", optimizer="RMSprop")
        return model_lstm      
    
    # Make a copy of data
    X_trainCopy=X_train.copy()
    
    # Scale data
    if scale:
        scl = MinMaxScaler(feature_range=(0, 1))
        tmp_values = X_trainCopy[clm].values.reshape(-1,1)
        scalery    = scl.fit(tmp_values)
        X_trainCopy[[clm]] = scalery.transform(tmp_values)
    
    # Convert time series to suppervised learning:
    train_ts= df_Series_to_Supervised(X_trainCopy[[clm]], n_input=n_input,
                                  n_output=n_output, remove_nan=True)
    target_clmn=train_ts.columns[-n_output:]
    x_train_ts = train_ts.drop(target_clmn, axis = 1)
    y_train_ts = train_ts[target_clmn]

    input_sample_df=X_trainCopy[clm]
    
    if (n_input%n_output!=0):
        raise ValueError('Remainder of n_input/n_output is not zero, please increase or decrease n_output')
        
    if (len(kwargs.items())>1 and nsim==1):
        raise ValueError('nsim should be >1 for perdiction next series')        
    
    n_loops=int(n_input/n_output)
    
    pred_no_all=[]
    full_test_nsim_all=[]
    full_test_nsim_m=[]
    MAE_all=[]
    ir=0
    if deep_learning=='rnn':
        model=dl_rnn(seed=42,n_out=n_output)
        history=model.fit(x_train_ts.values, y_train_ts.values,epochs=epochs,verbose=0)
    elif deep_learning=='lstm':
        model=dl_lstm(seed=42,n_out=n_output)
        history=model.fit(x_train_ts.values, y_train_ts.values,epochs=epochs,verbose=0)        
    elif deep_learning=='gru':
        model=dl_lstm(seed=42,n_out=n_output)
        history=model.fit(x_train_ts.values, y_train_ts.values,epochs=epochs,verbose=0)             
    else:
        model=model.fit(x_train_ts.values, y_train_ts.values)        
    for key, test in kwargs.items():
        test_list=test[clm].tolist()
        # First series to predict
        if (ir==0):
            tmp_no=[]
            full_test_nsim=[]
            for isim in range(nsim):
                full_test=[]
                for ilp in range(n_loops):
                    if ((n_input-ilp*n_output)>0):
                        # Get the data to predict. For the first loop (ilp=0), latest data series 
                        # with size "n_input" is selected to predict "n_output" as target.                        
                        X_new=input_sample_df.iloc[-(n_input-ilp*n_output):].to_list()+full_test
                        
                        # Retrain model again to calculate uncertainty
                        if deep_learning=='rnn':
                            model=dl_rnn(seed=43+isim,n_out=n_output)
                            model.fit(x_train_ts.values, y_train_ts.values,epochs=epochs,verbose=0)
                        elif deep_learning=='lstm':    
                            model=dl_lstm(seed=43+isim,n_out=n_output)
                            model.fit(x_train_ts.values, y_train_ts.values,epochs=epochs,verbose=0)        
                        elif deep_learning=='gru':    
                            model=dl_lstm(seed=43+isim,n_out=n_output)
                            model.fit(x_train_ts.values, y_train_ts.values,epochs=epochs,verbose=0)                                 
                        else:
                            model=model.fit(x_train_ts.values, y_train_ts.values)   
                            
                        # Predict series    
                        pred=model.predict(np.array(np.array(X_new).reshape(1,n_input)))
                        if (isim==0):
                            tmp=list(test[no][ilp*n_output:(ilp+1)*n_output])
                            tmp_no.append(tmp)
                        if ((n_input-ilp*n_output)<n_output):
                            full_test=full_test+list(pred[0][:(n_input-ilp*n_output)])
                        else:
                            full_test=full_test+list(pred[0])
                
                # Back transform to original space
                if scale:
                    full_test_=scalery.inverse_transform(np.array(full_test).reshape(-1, 1))
                    full_test=[iback[0] for iback in full_test_ ]                        
                    
                full_test_nsim.append(full_test)
            
            # Calculate average of realizations
            if (nsim>1):
                full_test_m=[sum(full_test_nsim[j][i]/nsim for j in range(nsim)) for i in range(len(full_test_nsim[0]))]
                full_test_nsim_m.append(full_test_m)
                
                # Calculate Mean Square Error
                MAE=np.nanmean([abs(full_test_m[i]-test_list[i]) for i in range(len(full_test))])  
                MAE_all.append(MAE)
            else:
                # Calculate Mean Square Error
                MAE=np.nanmean([abs(full_test[i]-test_list[i]) for i in range(len(full_test))])  
                MAE_all.append(MAE)                
            
            # Get index for prediction
            pred_no_all.append(list(itertools.chain(*tmp_no))) 
            full_test_nsim_all.append(full_test_nsim)  
            
            # Replace missing series with the predicted ones. This series used to predict next series
            # if it is requested             
            if len(kwargs.items())>1 and nsim>1:
                new_series=[full_test_nsim_m[ir][i] if math.isnan(test_list[i]) else test_list[i] 
                           for i in range(len(full_test_nsim_m[ir]))] 
                test_new=test.copy() 
                test_new[clm]=new_series      
                
                # Add the predicted series and actual values to training series
                # The new training set is used to predict next test series
                X_train_Added=pd.concat([X_train,test_new])  
          
        else:
            # Back transform data
            if scale:
                X_train_Added_copy=X_train_Added.copy()
                scl = MinMaxScaler(feature_range=(0, 1))
                tmp_values = X_train_Added[clm].values.reshape(-1,1)
                scalery    = scl.fit(tmp_values)                
                X_train_Added[[clm]] = scalery.transform(tmp_values)  
                new_series= X_train_Added[clm].tolist()

            
            # Convert new time series to suppervised learning
            train_ts= df_Series_to_Supervised(X_train_Added[[clm]], n_input=n_input,
                                          n_output=n_output, remove_nan=True)
            target_clmn=train_ts.columns[-n_output:]
            x_train_ts = train_ts.drop(target_clmn, axis = 1)
            y_train_ts = train_ts[target_clmn]         
            
            tmp_no=[]
            full_test_nsim=[]
            for isim in range(nsim):
                full_test=[]
                for ilp in range(n_loops):
                    if ((n_input-ilp*n_output)>=0):
                        # Get the data to predict. For the first loop (ilp=0), latest data series 
                        # with size "n_input" is selected to predict "n_output" as target.                        
                        X_new=new_series[-(n_input-ilp*n_output):]+full_test
                        
                        # Fit new model based on prediction from previous series
                        # the model is retrain over loop to calculate uncertainty in 
                        # prediction                        
                        if deep_learning=='rnn':
                            model=dl_rnn(seed=43+isim,n_out=n_output)
                            model.fit(x_train_ts.values, y_train_ts.values,epochs=epochs,verbose=0)   
                        elif deep_learning=='lstm':    
                            model=dl_lstm(seed=43+isim,n_out=n_output)
                            model.fit(x_train_ts.values, y_train_ts.values,epochs=epochs,verbose=0)    
                        elif deep_learning=='gru':    
                            model=dl_lstm(seed=43+isim,n_out=n_output)
                            model.fit(x_train_ts.values, y_train_ts.values,epochs=epochs,verbose=0)                              
                        else:
                            model=model.fit(x_train_ts.values, y_train_ts.values)                     
                        
                        # Predict series
                        pred=model.predict(np.array(np.array(X_new).reshape(1,n_input)))
                        
                        if ((n_input-ilp*n_output)<n_output):
                            full_test=full_test+list(pred[0][:(n_input-ilp*n_output)])
                        else:
                            full_test=full_test+list(pred[0])  
                            
                # Back transform data
                if scale:
                    full_test_=scalery.inverse_transform(np.array(full_test).reshape(-1, 1))
                    full_test=[iback[0] for iback in full_test_ ]
                    # Calculate Mean Square Error
                    MAE=np.nanmean([abs(full_test_m[i]-test_list[i]) for i in range(len(full_test))])  
                    MAE_all.append(MAE)
                else:
                    # Calculate Mean Square Error
                    MAE=np.nanmean([abs(full_test[i]-test_list[i]) for i in range(len(full_test))])  
                    MAE_all.append(MAE)                                         

                full_test_nsim.append(full_test)
            
            # Calculate mean of realizations
            if (nsim>1):
                full_test_m=[sum(full_test_nsim[j][i]/nsim for j in range(nsim)) for i in range(len(full_test_nsim[0]))]              
                
            pred_no_tmp=[max(pred_no_all[ir-1])+i+1 for i in range(n_input)]
            pred_no_all.append(pred_no_tmp) 
            full_test_nsim_all.append(full_test_nsim)       
            if (nsim> 1): full_test_nsim_m.append(full_test_m) 
            
            # Replace missing series with the predicted ones. This series used to predict next series
            # if requested
            new_series=[full_test_nsim_m[ir][i] if math.isnan(test_list[i]) else test_list[i] 
                       for i in range(len(full_test_nsim_m[ir]))] 
            test_new=test.copy() 
            test_new[clm]=new_series            
            
            # Add the series to training series
            X_train_Added=pd.concat([X_train_Added_copy,test_new]) 
            
        ir+=1   
    if data_out:
        return pred_no_all,full_test_nsim_all,full_test_nsim_m,np.nanmean(MAE_all),x_train_ts,y_train_ts
    
    else:
        return pred_no_all,full_test_nsim_all,full_test_nsim_m,np.nanmean(MAE_all)


#################################################################################################### 

class EDA_plot:
    def histplt (val: list,bins: int,title: str,xlabl: str,ylabl: str,xlimt: list,
                 ylimt: list=False, loc: int =1,legend: int=1,axt=None,days: int=False,
                 class_: int=False,scale: int=1,int_: int=0,nsplit: int=1,
                 font: int=5,color: str='b') -> None :
        
        """ Make histogram of data"""
        
        ax1 = axt or plt.axes()
        font = {'size'   : font }
        plt.rc('font', **font) 
        
        #val=val[~np.isnan(val)]
        val=np.array(val)
        plt.hist(val, bins=bins, weights=np.ones(len(val)) / len(val),ec='black',color=color)
        n=len(val)
        Mean=np.nanmean(val)
        Median=np.nanmedian(val)
        SD=np.sqrt(np.nanvar(val))
        Max=np.nanmax(val)
        Min=np.nanmin(val)
    
        if (int_==0):
            txt='n=%.0f\nMean=%0.3f\nMedian=%0.3f\nσ=%0.3f\nMax=%0.3f\nMin=%0.3f'
        else:
            txt='n=%.0f\nMean=%0.3f\nMedian=%0.3f\nMax=%0.3f\nMin=%0.3f'        
        anchored_text = AnchoredText(txt %(n,Mean,Median,SD,Max,Min), borderpad=0, 
                                     loc=loc,prop={ 'size': font['size']*scale})    
        if(legend==1): ax1.add_artist(anchored_text)
        if (scale): plt.title(title,fontsize=font['size']*(scale+0.15))
        else:       plt.title(title)
        plt.xlabel(xlabl,fontsize=font['size']) 
        ax1.set_ylabel('Frequency',fontsize=font['size'])
        if (scale): ax1.set_xlabel(xlabl,fontsize=font['size']*scale)
        else:       ax1.set_xlabel(xlabl)
    
        try:
            xlabl
        except NameError:
            pass    
        else:
            if (scale): plt.xlabel(xlabl,fontsize=font['size']*scale) 
            else:        plt.xlabel(xlabl)   
            
        try:
            ylabl
        except NameError:
            pass      
        else:
            if (scale): plt.ylabel(ylabl,fontsize=font['size']*scale)  
            else:         plt.ylabel(ylabl)  
            
        if (class_==True): plt.xticks([0,1])
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        ax1.grid(linewidth='0.1')
        if days:plt.xticks(range(0,45,nsplit),y=0.01, fontsize=8.6)  
        plt.xticks(fontsize=font['size']*scale)    
        plt.yticks(fontsize=font['size']*scale)   
        try:
            xlimt
        except NameError:
            pass  
        else:
            plt.xlim(xlimt) 
            
        try:
            ylimt
        except NameError:
            pass  
        else:
            plt.ylim(ylimt)        
    
    ######################################################################### 
                        
    def CDF_plot(data_var: list,nvar: int,label:str,colors:str,title:str,xlabel:str,
                 ylabel:str='Cumulative Probability', bins: int =1000,xlim: list=(0,100),
                 ylim: list=(0,0.01),linewidth: float =2.5,loc: int=0,axt=None,
                 x_ftze: float=12,y_ftze: float=12,tit_ftze: float=12,leg_ftze: float=9) -> None:
        
        """
        Cumulative Distribution Function
         
        """
        ax1 = axt or plt.axes() 
        def calc(data:[float])  -> [float]:
            var_mean=int(np.nanmean(data))
            var_median=int(np.nanmedian(data))
            var_s=int(np.nanstd(data))
            var_n=len(data)
            val_=np.array(data)
            counts, bin_edges = np.histogram(val_, bins=bins,density=True)
            cdf = np.cumsum(counts)
            tmp=max(cdf)
            cdf=cdf/float(tmp)
            return var_mean,var_median,var_s,var_n,bin_edges,cdf
               
        if nvar==1:
            var_mean,var_median,var_s,var_n,bin_edges,cdf=calc(data_var)
            if label:
                label_=f'{label} : n={var_n} \n mean= {var_mean}\n median= {var_median}\n $\sigma$={var_s}'
                plt.plot(bin_edges[1:], cdf,color=colors, linewidth=linewidth,
                    label=label_)                
            else:
                plt.plot(bin_edges[1:], cdf,color=colors, linewidth=linewidth)

        else:    
            # Loop over variables
            for i in range (nvar):
                data = data_var[i]
                var_mean,var_median,var_s,var_n,bin_edges,cdf=calc(data)
                label_=f'{label[i]} : n={var_n}, mean= {var_mean}, median= {var_median}, $\sigma$={var_s}'
                plt.plot(bin_edges[1:], cdf,color=colors[i], linewidth=linewidth,
                        label=label_)
         
        plt.xlabel(xlabel,fontsize=x_ftze, labelpad=6)
        plt.ylabel(ylabel,fontsize=y_ftze)
        plt.title(title,fontsize=tit_ftze)
        if label:
            plt.legend(loc=loc,fontsize=leg_ftze,markerscale=1.2)
        
        ax1.grid(linewidth='0.2')
        plt.xlim(xlim) 
        plt.ylim(ylim)         

    ######################################################################### 
            
    def CrossPlot (x:list,y:list,title:str,xlabl:str,ylabl:str,loc:int,
                   xlimt:list,ylimt:list,axt=None,scale: float=0.8,alpha: float=0.6,
                   markersize: float=6,marker: str='ro', fit_line: bool=False) -> None:
        """
        Cross plto between two variables
         
        """
        ax1 = axt or plt.axes()
        x=np.array(x)
        y=np.array(y)    
        no_nan=np.where((~np.isnan(x)) & (~np.isnan(y)))[0]
        Mean_x=np.mean(x)
        SD_x=np.sqrt(np.var(x)) 
        #
        n_x=len(x)
        n_y=len(y)
        Mean_y=np.mean(y)
        SD_y=np.sqrt(np.var(y)) 
        corr=np.corrcoef(x[no_nan],y[no_nan])
        n_=len(no_nan)
        #txt=r'$\rho_{x,y}=$%.2f'+'\n $n=$%.0f '
        #anchored_text = AnchoredText(txt %(corr[1,0], n_),borderpad=0, loc=loc,
        #                         prop={ 'size': font['size']*0.95, 'fontweight': 'bold'})  
        
        txt=r'$\rho_{x,y}}$=%.2f'+'\n $n$=%.0f \n $\mu_{x}$=%.0f \n $\sigma_{x}$=%.0f \n '
        txt+=' $\mu_{y}$=%.0f \n $\sigma_{y}$=%.0f'
        anchored_text = AnchoredText(txt %(corr[1,0], n_x,Mean_x,SD_x,Mean_y,SD_y), loc=4,
                                prop={ 'size': font['size']*1.1, 'fontweight': 'bold'})    
            
        ax1.add_artist(anchored_text)
        Lfunc1=np.polyfit(x,y,1)
        vEst=Lfunc1[0]*x+Lfunc1[1]    
        try:
            title
        except NameError:
            pass  # do nothing! 
        else:
            plt.title(title,fontsize=font['size']*(scale))   
    #
        try:
            xlabl
        except NameError:
            pass  # do nothing! 
        else:
            plt.xlabel(xlabl,fontsize=font['size']*scale)            
    #
        try:
            ylabl
        except NameError:
            pass  # do nothing! 
        else:
            plt.ylabel(ylabl,fontsize=font['size']*scale)        
            
        try:
            xlimt
        except NameError:
            pass  # do nothing! 
        else:
            plt.xlim(xlimt)   
    #        
        try:
            ylimt
        except NameError:
            pass  # do nothing! 
        else:
            plt.ylim(ylimt)   
          
        plt.plot(x,y,marker,markersize=markersize,alpha=alpha)  
        if fit_line:
            ax1.plot(x, vEst,'k-',linewidth=2)   
        ax1.grid(linewidth='0.1') 
        plt.xticks(fontsize=font['size']*0.85)    
