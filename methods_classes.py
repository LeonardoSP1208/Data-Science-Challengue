## Para usar y operar con DataFrames, arrays y objetos de tiempo
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Para pruebas de hipótesis estadística
from scipy import stats

## Para imputación univariada
from sklearn.impute import SimpleImputer

## Para elaborar clases personalizadas
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

## Para el balanceo de datos
import sys
import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from collections import Counter

## Para medir el performance del modelo
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import plot_roc_curve, auc, roc_auc_score, confusion_matrix 
from sklearn.metrics import classification_report, recall_score, f1_score, precision_score

## Para Validación Cruzada
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

## Para el WoE de OptBinning
from optbinning import OptimalBinning

## Para la transformación de variables (yeo-jhonson, Quantil-Transform o ninguna)
## y para la estandarización de las mismas
from sklearn import preprocessing

## Para la codificación de variables categóricas
import category_encoders as ce

## Para la imputación multivariada 
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
from missingpy import MissForest

## Para guardar y cargar las clases junto al mejor modelo
import pickle

## Separa variables numéricas de categóricas
def vars_cont_cat(data):
    cols_num = []
    cols_cat = []
    for col in data:
        if (data[col].dtype.kind in 'bifc'):
            cols_num.append(col) 
        else:
            cols_cat.append(col)
    df_var_cont = data[cols_num]
    df_var_cat = data[cols_cat]
    return data[cols_num], data[cols_cat]  


### Hipotesis para verificar relación entre variables predictoras continuas y el target
def Mann_Whitney(df, df_cont, target):
    lista_UMW = []
    col_UMW = []
    for col in df_cont.columns:
        var_exp_0 = df[df[target]==0][col].dropna()
        var_exp_1 = df[df[target]==1][col].dropna()
        if (df[col].count()>=20):
            try:
                UMW2, p_value = stats.mannwhitneyu(var_exp_0, var_exp_1, 
                                                   use_continuity = True, alternative = 'two-sided')
                lista_UMW.append([UMW2, p_value])
                col_UMW.append(col)
            except:
                pass
        else:
            try:
                UMW2, p_value = stats.mannwhitneyu(var_exp_0, var_exp_1, 
                                                   use_continuity = False, alternative = 'two-sided')
                lista_UMW.append([UMW2, p_value])
                col_UMW.append(col)
            except:
                pass
    hipotesis_indep = pd.DataFrame(lista_UMW, columns=['Estadístico','p-valor'], index=col_UMW)
    return hipotesis_indep


### Hipotesis para verificar relación entre variables predictoras categóricas y el target
def Chi_Square(df, df_cat, target):
    lista_Chi = []
    col_Chi = []
    for col in df_cat.columns:
        try:
            Chi2, p_value, dof, expect = stats.chi2_contingency(pd.crosstab(df[col],
                                                                            df[target]).values)
            if ((expect<5).sum()/expect.shape[0]<=0.2):
                    lista_Chi.append([Chi2, p_value])
                    col_Chi.append(col)
            else:
                try:
                    fisher2, p_value2 = stats.fisher_exact(pd.crosstab(df[col],
                                                                       df[target]).values)
                    lista_Chi.append([fisher2, p_value2])
                    col_Chi.append(col)
                except:
                    lista_Chi.append([Chi2, p_value])
                    col_Chi.append(col)
        except:
            pass
    hipotesis_indep2 = pd.DataFrame(lista_Chi,columns=['Estadístico','p-valor'],index=col_Chi)
    return hipotesis_indep2


### Identificar outliers por variable
def encontrar_outliers(data: pd.core.frame.DataFrame, col: str):
    """Returns a array with the outliers of variable 'col' in 'data'"""
    outliers = list()
    q1 = data[col].quantile(0.25)
    q3 = data[col].quantile(0.75)
    iqr = q3 - q1
    lower_limit = q1 - (1.5 * iqr)
    upper_limit = q3 + (1.5 * iqr)
    
    for out in data[col]:
        if (out > upper_limit) or (out < lower_limit):
            outliers.append(out)
    return np.array(outliers)


def graficos(data: pd.core.frame.DataFrame, col: str, hue: str):
    """Returns plots: boxplot and histplot"""
    fix,(ax1,ax2)=plt.subplots(1,2,figsize=(15,5))
    sns.boxplot(y=data[col],ax=ax1)
    ax1.set_ylabel=col
    ax1.set_title('Gráfico de cajas de {}'.format(col))
    sns.histplot(data, x=col, hue=hue, kde=True, ax=ax2)
    ax2.set_title('Distribution plot of {}'.format(col))
    plt.show()

    
def analisis_numerico(data: pd.core.frame.DataFrame, col: str, hue: str):
    """Returns outliers and plots"""
    q1 = data[col].quantile(0.25)
    q3 = data[col].quantile(0.75)
    iqr = q3 - q1
    lower_limit = q1 - (1.5 * iqr)
    upper_limit = q3 + (1.5 * iqr)
    if data[col].min() >= lower_limit:
        lower_limit = data[col].min()
    if data[col].max() <= upper_limit:
        upper_limit = data[col].max()
    print("Cantidad de outliers", col, ":", len(encontrar_outliers(data, col)))
    print("Media", col, ":", format(data[col].mean()))
    print("Mediana", col, ":", format(data[col].median()))
    print("Q1", col, ":", format(q1))
    print("Q3", col, ":", format(q3))
    print("LI", col, ":", format(lower_limit))
    print("LS", col, ":", format(upper_limit))
    print("cuantil 0.01", col, ":", format(data[col].quantile(0.01)))
    print("cuantil 0.99", col, ":", format(data[col].quantile(0.99)))
    print("min", col, ":", format(data[col].min()))
    print("max", col, ":", format(data[col].max()))
    graficos(data, col, hue)


## Creamos una clase que cree variables dummie eliminando una para evitar multicolinealidad,
## con .fit codificamos las variables categóricas y con .transform codificamos y eliminamos
## según se hizo en .fit 
class OneHotEncoder_DropOne:
    def fit(self, data):
        enc = ce.OneHotEncoder(use_cat_names=True)
        enc.fit(data)
        new_cols = data.columns.tolist()
        for col in data.columns:
            if (data[col].dtype.kind in 'O'):
                new_cols.remove(col+'_'+np.unique(data[col].dropna())[0])
        self.enc, self.new_cols = enc,new_cols
        
    def transform(self, data):
        new_data = self.enc.transform(data)
        data_trans = new_data[self.new_cols]
        return(data_trans)
    
    def save(self, fileName):
        """Save thing to a file."""
        f = open(fileName, "wb")
        pickle.dump(self,f)
        f.close()
        
    def load(fileName):
        """Return a thing loaded from a file."""
        f = open(fileName, "rb")
        obj = pickle.load(f)
        f.close()
        return obj
    # make load a static method
    load = staticmethod(load)

## Construimos una clase que impute datos en valores perdidos según el tipo de variable
## si es categórica imputa la moda y si es numérica imputa la mediana
class ImputerUniv():
    def fit(self,data):
        data_cont, data_cat = vars_cont_cat(data)
        var_cont, var_cat = data_cont.columns, data_cat.columns
        
        imp_cont = SimpleImputer(missing_values=np.nan, strategy='median')
        imp_cat = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        imp_cont.fit(data_cont)
        imp_cat.fit(data_cat)
        self.var_cont, self.var_cat  = var_cont, var_cat
        self.imp_cont, self.imp_cat  = imp_cont, imp_cat
        
    def transform(self,data):
        data_cont_trans = pd.DataFrame(self.imp_cont.transform(data[self.var_cont]), 
                                       columns=self.var_cont,
                                       index=data.index)
        data_cat_trans = pd.DataFrame(self.imp_cat.transform(data[self.var_cat]), 
                                      columns=self.var_cat, 
                                      index=data.index)
        data_trans = pd.concat([data_cont_trans,data_cat_trans],axis=1)
        return data_trans
    
    def save(self, fileName):
        """Save thing to a file."""
        f = open(fileName, "wb")
        pickle.dump(self,f)
        f.close()
        
    def load(fileName):
        """Return a thing loaded from a file."""
        f = open(fileName, "rb")
        obj = pickle.load(f)
        f.close()
        return obj
    # make load a static method
    load = staticmethod(load)

    
## Creamos una clase que tiene como finalidad transformar las variables originales a valores WoE, 
## para ello creamos un .fit que entrena con una data y un .transform que devuelve la data de entrada 
## a valores WoE según los criterios que tuvo en el entrenamiento la primera data en .fit.
class WoeExp(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0, bins=10, show_woe=False):
        self.threshold = threshold
        self.bins = bins
        self.show_woe = show_woe
    
    def fit(self, X, y):
        # Empty Dataframe
        newDF, woeDF = pd.DataFrame(), pd.DataFrame()

        # Extract Column Names
        data = pd.concat([X, y], axis=1)
        cols = data.columns

        # Run WOE and IV on all the independent variables
        intervals = []
        WoEs = []
        cols_intervals = []
        col_WoE = []
        for ivars in cols[~cols.isin([y.name])]:
            if (data[ivars].dtype.kind in 'bifc') and (len(np.unique(data[ivars]))>self.bins):
                binned_x = pd.qcut(data[ivars], self.bins,  duplicates='drop')
                int_var = pd.IntervalIndex(np.unique(binned_x))
                d0 = pd.DataFrame({'x': binned_x, 'y': data[y.name]})
                intervals.append(int_var) 
                cols_intervals.append(ivars)
            else:
                d0 = pd.DataFrame({'x': data[ivars], 'y': data[y.name]})
                
            d = d0.groupby("x", as_index=False).agg({"y": ["count", "sum"]})
            d.columns = ['Cutoff', 'N', 'Events']
            d['% of Events'] = np.maximum(d['Events'], 0.5) / d['Events'].sum()
            d['Non-Events'] = d['N'] - d['Events']
            d['% of Non-Events'] = np.maximum(d['Non-Events'], 0.5) / d['Non-Events'].sum()
            d['WoE'] = np.log(d['% of Events']/d['% of Non-Events'])
            d['IV'] = d['WoE'] * (d['% of Events'] - d['% of Non-Events'])
            d.insert(loc=0, column='Variable', value=ivars)
            #print("Information value of " + ivars + " is " + str(round(d['IV'].sum(),6)))
            temp = pd.DataFrame({"Variable" : [ivars], "IV" : [d['IV'].sum()]}, 
                                columns = ["Variable", "IV"])
            newDF = pd.concat([newDF,temp], axis=0)
            woeDF = pd.concat([woeDF,d], axis=0)
            d2 = d[["Cutoff",'WoE']]
            col_WoE.append(ivars)
            WoEs.append(d2.values)
        
            #Show WOE Table
            if self.show_woe == True:
                print(d)
                print("\n")
        
        self.col_WoE, self.WoEs, self.cols_intervals  = col_WoE, WoEs, cols_intervals
        self.intervals, self.woeDF, self.newDF  = intervals, woeDF, newDF
        
        if self.show_woe == True:
            return newDF, woeDF
        return self
    
    def transform(self, X):
        
        #Extract Column Names
        cols = X.columns
        data_without_cont = X.drop(self.cols_intervals,axis=1)
        
        for col in self.cols_intervals:
            name_cont = self.cols_intervals[self.cols_intervals.index(col)]
            int_cont = self.intervals[self.cols_intervals.index(col)]
            data_without_cont[col] = pd.cut(X[col], int_cont)
            
        data_WoE = data_without_cont

        for ivars in cols:
            
            name_ivar = self.col_WoE[self.col_WoE.index(ivars)]
            WoE_ivar = self.WoEs[self.col_WoE.index(ivars)]
            df_WoE = pd.DataFrame(WoE_ivar,columns=[name_ivar,'WoE_'+name_ivar])
            data_WoE = pd.merge(data_WoE, df_WoE, on= name_ivar, how='left')
            data_WoE['WoE_'+name_ivar] = np.where(data_WoE['WoE_'+name_ivar].isnull(),
                                            self.woeDF[self.woeDF['Variable']==name_ivar]['WoE'].iloc[-1],
                                            data_WoE['WoE_'+name_ivar])
                
        X_WoE_ultimate = data_WoE.drop(cols,axis=1)
        X_WoE_ultimate.index = X.index
        
        features_total_IV = self.newDF
        var_select_IV = features_total_IV[features_total_IV['IV']>=self.threshold]
        X_WoE_ultimate_threshold = X_WoE_ultimate[('WoE_'+var_select_IV['Variable']).array]
        return X_WoE_ultimate_threshold
    
    def save(self, fileName):
        """Save thing to a file."""
        f = open(fileName, "wb")
        pickle.dump(self,f)
        f.close()
        
    def load(fileName):
        """Return a thing loaded from a file."""
        f = open(fileName, "rb")
        obj = pickle.load(f)
        f.close()
        return obj
    # make load a static method
    load = staticmethod(load)
    

# Creamos una clase que tiene como finalidad transformar las variables originales a valores WoE o 
# dummie, para ello creamos un .fit que entrena con una data y un .transform que devuelve 
# la data de entrada  a valores WoE o dummie según los criterios que tuvo en el entrenamiento 
# la primera data en .fit.
class WoeOptBining(BaseEstimator, TransformerMixin):
    def __init__(self, trans_dtype, threshold=0, bins=10, show_woe=False):
        self.threshold = threshold
        self.bins = bins
        self.show_woe = show_woe
        self.trans_dtype = trans_dtype
    
    def fit(self, X, y):
        #Empty Dataframe
        df_IV, df_WoE = pd.DataFrame(), pd.DataFrame()
        #Run WOE and IV on all the independent variables
        cols = [] ######
        optb_list = [] ######
        tables = [] ######
        for col in X.columns:
            if (X[col].dtype.kind in 'bifc'):
                optb = OptimalBinning(name=col, dtype="numerical", solver="cp")
            else:
                optb = OptimalBinning(name=col, dtype="categorical", solver="mip")
            optb.fit(X[col], y)
            bin_table = optb.binning_table.build()
            optb_list.append(optb)
            tables.append(bin_table)
            cols.append(col)
            #Show WOE Table 
            if self.show_woe == True:
                print("#################################################### ",col,
                      " ####################################################")
                print(bin_table)
                optb.binning_table.plot(metric='woe')
                print('\n')
            
            temp = pd.DataFrame({"Variable":[col],
                                 "IV":[bin_table[bin_table.index=='Totals']['IV'][0]]},
                                columns = ["Variable", "IV"]) 
            df_IV = pd.concat([df_IV, temp], axis=0)
            
        df_WoE = pd.concat(tables, keys=cols, axis=0)
        self.optb_list, self.cols, self.tables = optb_list, cols, tables
        self.df_IV, self.df_WoE = df_IV, df_WoE
        
        if self.show_woe == True:
            return df_IV, df_WoE
        return self
    
    def transform(self, X):
        data_trans = pd.DataFrame()
        if (self.trans_dtype=='WoE'):
            for col in X.columns:
                optb = self.optb_list[self.cols.index(col)]
                col_trans = optb.transform(X[col], metric='woe')
                df_col_trans = pd.DataFrame(col_trans, columns=[col])
                data_trans = pd.concat([data_trans, df_col_trans], axis=1)
            data_trans.index = X.index
            features_total_IV = self.df_IV
            var_select_IV = features_total_IV[features_total_IV['IV']>=self.threshold]
            data_trans = data_trans[(var_select_IV['Variable']).array]
        elif (self.trans_dtype=='dummies'):
            for col in X.columns:
                optb = self.optb_list[self.cols.index(col)]
                dftable = self.tables[self.cols.index(col)]
                col_trans = optb.transform(X[col], metric='bins')
                df_col_trans = pd.DataFrame(col_trans, columns=[col])
                dftable_drop = dftable[dftable['Count']!=0]
                dftable_drop_min = dftable_drop[dftable_drop['IV']==min(dftable_drop['IV'])]
                try:
                    bin_min_WoE = dftable_drop_min['Bin'].values[0]
                    df_dummies = pd.get_dummies(df_col_trans,drop_first=False)
                    df_dummies = df_dummies.drop([col+'_'+bin_min_WoE], axis=1)
                except:
                    bin_min_WoE = dftable_drop_min['Bin'].values[0][0]
                    df_dummies = pd.get_dummies(df_col_trans,drop_first=False)
                    for col_dummie in df_dummies.columns:
                        if bin_min_WoE in col_dummie:
                            df_dummies = df_dummies.drop([col_dummie], axis=1)
                data_trans = pd.concat([data_trans,df_dummies],axis=1)
            data_trans.index = X.index
        return(data_trans)
    
    def save(self, fileName):
        """Save thing to a file."""
        f = open(fileName, "wb")
        pickle.dump(self,f)
        f.close()
        
    def load(fileName):
        """Return a thing loaded from a file."""
        f = open(fileName, "rb")
        obj = pickle.load(f)
        f.close()
        return obj
    # make load a static method
    load = staticmethod(load)
    
    
## Creamos una clase que transforme las variables por dos métodos (Yeo-Jhonson y Quantil Trnasform)
class TransformVar(BaseEstimator, TransformerMixin):
    def __init__(self, method=None):
        self.method = method
        
    def fit(self, X, y=None):
        data_new = X.reset_index(drop=True)
        data_cont = pd.DataFrame()
        for col in data_new.columns:
            if (data_new[col].dtype.kind in 'bifc'):
                data_cont[col] = data_new[col]
        if (self.method=='yeo-johnson'):
            data_scaler = preprocessing.PowerTransformer(method = 'yeo-johnson', standardize = False)
        elif (self.method=='QuantileTransformer'):
            data_scaler = preprocessing.QuantileTransformer(output_distribution='normal', random_state=0)
        else:
            return self
        data_scaled = data_scaler.fit_transform(data_cont)
        col_fit = data_cont.columns
        self.data_scaler, self.col_fit = data_scaler, col_fit
        return self
    
    def transform(self, X):

        if (self.method == 'yeo-johnson') or (self.method == 'QuantileTransformer'):
            data_new = X.reset_index(drop=True)
            data_cont = pd.DataFrame()
            data_cat = pd.DataFrame()
            for col in data_new.columns:
                if (data_new[col].dtype.kind in 'bifc'):
                    data_cont[col] = data_new[col]
                else:
                    data_cat[col] = data_new[col]
            cont_trans = self.data_scaler.transform(data_cont[self.col_fit])
            data_cont_trans = pd.DataFrame(cont_trans, columns=self.col_fit)
            data_trans = pd.concat([data_cont_trans, data_cat], axis=1)
            data_trans.index = X.index
        else:
            data_trans = X
        return data_trans[X.columns]
    
    def save(self, fileName):
        """Save thing to a file."""
        f = open(fileName,"wb")
        pickle.dump(self,f)
        f.close()
        
    def load(fileName):
        """Return a thing loaded from a file."""
        f = open(fileName,"rb")
        obj = pickle.load(f)
        f.close()
        return obj
    # make load a static method
    load = staticmethod(load)
    

## Creamos un clase que detecte outliers con .fit y posteriormente los reemplace con .transform
class TreatmentOutliers(BaseEstimator, TransformerMixin):    
    def __init__(self, thresholds=None):
        self.thresholds = thresholds
        
    def fit(self, X, y=None):
        lowers = []
        uppers = []
        cols_fit = []
        for col in X.columns:
            if (X[col].dtype.kind in 'bifc'):
                q1 = X[col].quantile(0.25)
                q3 = X[col].quantile(0.75)
                iqr = q3-q1
                lower_mustache = q1-(1.5*iqr)
                upper_mustache = q3+(1.5*iqr)
                if X[col].min() >= lower_mustache:
                    lower_mustache = X[col].min()
                if X[col].max() <= upper_mustache:
                    upper_mustache = X[col].max()
                
                if self.thresholds:
                    lower_threshold = X[col].quantile(self.thresholds["lower"])
                    upper_threshold = X[col].quantile(self.thresholds["upper"])
                    if lower_threshold >= lower_mustache:
                        lower_threshold = lower_mustache
                    if upper_threshold <= upper_mustache:
                        upper_threshold = upper_mustache
                else:
                    lower_threshold = lower_mustache
                    upper_threshold = upper_mustache
                
                lowers.append(lower_threshold)
                uppers.append(upper_threshold)
                cols_fit.append(col)
        self.cols_fit, self.lowers, self.uppers  = cols_fit, lowers, uppers
        return self
    
    def transform(self,X):
        data_new = X.reset_index(drop=True)
        data_cont = pd.DataFrame()
        data_cat = pd.DataFrame()
        for col in X.columns:
            if (data_new[col].dtype.kind in 'bifc'):
                name_var = self.cols_fit[self.cols_fit.index(col)]
                lower = self.lowers[self.cols_fit.index(col)]
                upper = self.uppers[self.cols_fit.index(col)]
                imput_lower = np.where(data_new[name_var] < lower, lower, data_new[name_var])
                imput_lower_upper = np.where(imput_lower > upper, upper, imput_lower)
                data_cont[name_var] = imput_lower_upper
            else:
                data_cat[col]=data_new[col]
        data_trans=pd.concat([data_cont,data_cat],axis=1)
        data_trans.index = X.index
        return(data_trans[X.columns])
    
    def save(self, fileName):
        """Save thing to a file."""
        f = open(fileName,"wb")
        pickle.dump(self,f)
        f.close()
        
    def load(fileName):
        """Return a thing loaded from a file."""
        f = open(fileName,"rb")
        obj = pickle.load(f)
        f.close()
        return obj
    # make load a static method
    load = staticmethod(load)


## Creamos un clase que detecte variables con desviación nula con .fit 
## y posteriormente los elimine con .transform 
class DropConstants(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X_cont, X_cat = vars_cont_cat(X)
        col_X_cont_desv_null = X_cont.describe().T[X_cont.describe().T['std']<1.0E-10].index
        self.col_X_cont_desv_null = col_X_cont_desv_null
        return self
    
    def transform(self, X):
        X_trans = X.drop(self.col_X_cont_desv_null, axis=1)
        return X_trans
    
    def save(self, fileName):
        """Save thing to a file."""
        f = open(fileName, "wb")
        pickle.dump(self,f)
        f.close()
        
    def load(fileName):
        """Return a thing loaded from a file."""
        f = open(fileName, "rb")
        obj = pickle.load(f)
        f.close()
        return obj
    # make load a static method
    load = staticmethod(load)


## Creamos un clase con el estandarizador para que incluya en sus inputs
## y outputs las variables categóricas
class Estandarice(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        data_cont = pd.DataFrame()
        data_cat = pd.DataFrame()
        for col in X.columns:
            if (X[col].dtype.kind in 'bifc'):
                data_cont[col] = X[col]
        data_fit_std = preprocessing.StandardScaler().fit(data_cont)
        col_fit = data_cont.columns
        self.data_fit_std, self.col_fit = data_fit_std, col_fit
        return self
        
    def transform(self, X):
        data_cont = pd.DataFrame()
        data_cat = pd.DataFrame()
        for col in X.columns:
            if (X[col].dtype.kind in 'bifc'):
                data_cont[col] = X[col]
            else:
                data_cat[col] = X[col]
        cont_trans = self.data_fit_std.transform(data_cont[self.col_fit])
        data_cont_trans = pd.DataFrame(cont_trans, columns=self.col_fit)
        data_cont_trans.index = X.index
        data_cat.index = X.index
        data_trans = pd.concat([data_cont_trans, data_cat], axis=1)
        return(data_trans[X.columns])
    
    def save(self, fileName):
        """Save thing to a file."""
        f = open(fileName, "wb")
        pickle.dump(self,f)
        f.close()
        
    def load(fileName):
        """Return a thing loaded from a file."""
        f = open(fileName, "rb")
        obj = pickle.load(f)
        f.close()
        return obj
    # make load a static method
    load = staticmethod(load)
    

## Creamos un clase con codificador de variables categóricas por 4 métodos
## 2 métodos para modelos clásicos y 2 para modelos de ML
class CategoryEncoders(BaseEstimator, TransformerMixin):
    def __init__(self, method='WoEEncoder', threshold=0, bins=10, min_samples_leaf=1):
        self.method = method
        self.threshold = threshold
        self.bins = bins
        self.min_samples_leaf = min_samples_leaf
    
    def fit(self, X, y):
        cat = []
        cont = []
        for col in X.columns:
            if (X[col].dtype.kind not in 'bifc'):
                cat.append(col)
            else:
                cont.append(col)
        # Codificación usando WoE
        if (self.method=='WoEEncoder'):
            me_WoEEncoder = WoeOptBining('WoE', self.threshold, self.bins)
            me_WoEEncoder.fit(X[cat], y)
            self.me_WoEEncoder, self.cat, self.cont = me_WoEEncoder, cat, cont
        
        # Codificación usando Modelos Generalizados Mixtos   
        if (self.method=='GLMMEncoder'):
            GLMM = ce.GLMMEncoder(handle_missing='return_nan', binomial_target=True)
            GLMM.fit(X[cat], y)
            self.GLMM, self.cat, self.cont = GLMM, cat, cont
        
        # Codificación usando Cat Boost  
        if (self.method=='CatBoostEncoder'):
            CB = ce.CatBoostEncoder(handle_missing='return_nan')
            CB.fit(X[cat], y)
            self.CB, self.cat, self.cont = CB, cat, cont
        
        # Codificación usando Target Encoder  
        if (self.method=='TargetEncoder'):
            TE = ce.TargetEncoder(handle_missing='return_nan', min_samples_leaf=self.min_samples_leaf)
            TE.fit(X[cat], y)
            self.TE, self.cat, self.cont = TE, cat, cont
        
        return self
    
    def transform(self, X):
        data_cat = X[self.cat]
        data_cont = X[self.cont]
        if (self.method=='WoEEncoder'):
            data_cat_trans = self.me_WoEEncoder.transform(data_cat)
            data_cat_trans.index = X.index
        
        if (self.method=='GLMMEncoder'):
            data_cat_trans = self.GLMM.transform(data_cat)
            data_cat_trans.index = X.index
        
        if (self.method=='CatBoostEncoder'):
            data_cat_trans = self.CB.transform(data_cat)
            data_cat_trans.index = X.index
        
        if (self.method=='TargetEncoder'):
            data_cat_trans = self.TE.transform(data_cat)
            data_cat_trans.index = X.index
        
        data_trans = pd.concat([data_cat_trans, data_cont], axis=1)
        
        return data_trans
    
    def save(self, fileName):
        """Save thing to a file."""
        f = open(fileName, "wb")
        pickle.dump(self,f)
        f.close()
        
    def load(fileName):
        """Return a thing loaded from a file."""
        f = open(fileName, "rb")
        obj = pickle.load(f)
        f.close()
        return obj
    # make load a static method
    load = staticmethod(load)

    
## Creamos una clase que impute datos de forma multivariada para datos numéricos 
## e impute la moda para datos categóricos, con .fit entrenamos y .transform imputamos
class ImputerMultiVar(BaseEstimator, TransformerMixin):
    def __init__(self, method='Iterative_Imputer', max_iter=10, n_nearest_features=None, 
                 initial_strategy='mean', n_neighbors=5, weights='uniform',
                 criterion = ('mse','gini'), n_estimators=100, max_depth=None):
        self.method = method
        self.max_iter = max_iter
        self.n_nearest_features = n_nearest_features
        self.initial_strategy = initial_strategy
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.criterion = criterion
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        
    def fit(self, X, y=None):
        data_cont = pd.DataFrame()
        data_cat = pd.DataFrame()
        for col in X.columns:
            if (X[col].dtype.kind in 'bifc'):
                 data_cont[col] = X[col]
            else:
                 data_cat[col] = X[col]
        if (self.method == 'Iterative_Imputer'):
            imp_cont = IterativeImputer(max_iter=self.max_iter, 
                                        n_nearest_features=self.n_nearest_features,
                                        initial_strategy=self.initial_strategy,
                                        random_state=0)
            imp_cont.fit(data_cont)
            self.imp_cont = imp_cont
        elif (self.method == 'KNN_Imputer'):
            imp_cont = KNNImputer(n_neighbors=self.n_neighbors,
                                  weights=self.weights)
            imp_cont.fit(data_cont)
            self.imp_cont = imp_cont
        elif (self.method == 'Forest_Imputer'):
            imp_cont = MissForest(max_iter=self.max_iter, 
                                  criterion=self.criterion,
                                  n_estimators=self.n_estimators,
                                  max_depth=self.max_depth, 
                                  random_state=0)
            imp_cont.fit(data_cont)
            self.imp_cont = imp_cont
        else:
            print('Error: Only method Iterative_Imputer, KNN_Imputer or Forest_Imputer')
        
        col_cont_fit = data_cont.columns
        col_cat_fit = data_cat.columns
        self.col_cont_fit, self.col_cat_fit = col_cont_fit, col_cat_fit
        return self
        
    def transform(self, X):
        data_cont = X[self.col_cont_fit]
        if ((self.method=='Iterative_Imputer') or (self.method=='KNN_Imputer') or 
            (self.method=='Forest_Imputer')):
            data_trans_cont = pd.DataFrame(self.imp_cont.transform(data_cont),
                                           columns=self.col_cont_fit)
            data_trans_cont.index = X.index
            data_trans_cat = X[self.col_cat_fit]
            data_trans_cat.index = X.index
            data_trans = pd.concat([data_trans_cont, data_trans_cat], axis=1)
        else:
            print('Error: Only method Iterative_Imputer, KNN_Imputer or Forest_Imputer')
        return(data_trans[X.columns])
    
    def save(self, fileName):
        """Save thing to a file."""
        f = open(fileName,"wb")
        pickle.dump(self,f)
        f.close()
        
    def load(fileName):
        """Return a thing loaded from a file."""
        f = open(fileName,"rb")
        obj = pickle.load(f)
        f.close()
        return obj
    # make load a static method
    load = staticmethod(load)


## Creamos una función que muestre el performance de cada modelo según cada partición de la data
def score_datas(model, X_train, y_train, X_test, y_test, name_model):
    predictions_train = model.predict(X_train)
    predictions_test = model.predict(X_test)
    pred = {'train': predictions_train, 'test':predictions_test}
    predict_proba_train = model.predict_proba(X_train)
    predict_proba_test = model.predict_proba(X_test)
    pred_proba = {'train': predict_proba_train, 'test':predict_proba_test}
    AUC_train = roc_auc_score(y_train, predict_proba_train[:,1])
    AUC_test = roc_auc_score(y_test,predict_proba_test[:,1]) 
    Gini_train = 2*AUC_train-1
    Gini_test = 2*AUC_test-1
    Precision_train = precision_score(y_train,predictions_train)
    Precision_test = precision_score(y_test,predictions_test)
    Recall_train = recall_score(y_train,predictions_train)
    Recall_test = recall_score(y_test,predictions_test)
    F1_train = f1_score(y_train, predictions_train)
    F1_test = f1_score(y_test, predictions_test)
    res = pd.DataFrame([[F1_train,F1_test],[Precision_train,Precision_test],
                  [Recall_train,Recall_test],[Gini_train,Gini_test]],
                 index= ['F1 Score','Precision','Recall','GINI'],
                 columns=['Data Train '+name_model,'Data Test '+name_model]).T

    print("################################################### TRAIN SCORE ###################################################")
    print('F1 Train score: ',F1_train)
    print('GINI Train score: ',Gini_train)
    print(classification_report(y_train, predictions_train))
    confusion_matrix_train = pd.crosstab(y_train, predictions_train, 
                                         rownames=['Actual'], colnames=['Predicted'])
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.heatmap(confusion_matrix_train, annot=True,ax = axes[0])
    plot_roc_curve(model, X_train, y_train, ax = axes[1])
    axes[0].set_title("Matriz de Confusión Data Train")
    axes[1].set_title("Curva ROC Data Train")
    plt.show()
    print('\n')
    print("################################################### TEST SCORE ###################################################")
    print('F1 Test score: ',F1_test)
    print('GINI Test score: ',Gini_test)
    print(classification_report(y_test, predictions_test))
    confusion_matrix_test = pd.crosstab(y_test, predictions_test, 
                                        rownames=['Actual'], colnames=['Predicted'])
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.heatmap(confusion_matrix_test, annot=True,ax = axes[0])
    plot_roc_curve(model, X_test, y_test,ax = axes[1])
    axes[0].set_title("Matriz de Confusión Data Test")
    axes[1].set_title("Curva ROC Data Test")
    plt.show()
    print('\n')
    print(res)
    
    return res, pred, pred_proba


def results_cv(grid_result):
    """Mostramos los Score tanto de test como de train en cada 
       validación para todas las combinaciones"""
    if type(grid_result)==dict:
        best_results = pd.DataFrame()
        for key in grid_result.keys():
            results = pd.DataFrame(grid_result[key].cv_results_)
            cols_scores = []
            for col in results.columns:
                if (("mean_train_score" in col) or ("mean_test_score" in col) or 
                    ("std_train_score" in col) or ("std_test_score" in col)):
                    cols_scores.append(col)
            results_train_test = results[cols_scores]
            results_train_test_ord = results_train_test.sort_values(['mean_test_score'], 
                                                                    ascending=False)
            best_results_key = results_train_test_ord[:1]
            best_results_key.index = ['the_best_'+ key]
            best_results = pd.concat([best_results, best_results_key], axis=0)
    else:    
        results = pd.DataFrame(grid_result.cv_results_)
        cols_scores = []
        for col in results.columns:
            if (("mean_train_score" in col) or ("mean_test_score" in col) or 
                ("std_train_score" in col) or ("std_test_score" in col)):
                cols_scores.append(col)
        results_train_test = results[cols_scores]
        results_train_test_ord = results_train_test.sort_values(['mean_test_score'], 
                                                                ascending=False)
        best_results = results_train_test_ord[:1]
        best_results.index = ['the_best']
    return best_results


## Creamos una función que elimine columnas de "data_trans" que no están en "data_ref" y 
## que cree columnas que no están en "data_trans" pero si en "data_ref"
def create_delete_cols(data_ref, data_trans):
    notin_data_ref = []
    for col in data_trans.columns:
        if col not in data_ref.columns:
            notin_data_ref.append(col)
    data_1 = data_trans.drop(notin_data_ref,axis=1)
    for col in data_ref.columns:
        if col not in data_1.columns:
            data_1[col] = 0
    return(data_1)


def cross_validation_RLog_WoE(X_train, y_train, class_weight, class_WoE, cv, n_iter):
    """Realiza validación cruzada con RandomizedSearchCV usando un canalizador que
       contiene una clase que tranforma la data a valores WoE y un modelo de RLog"""
    # Canalizador
    pipe = Pipeline(steps=[('WoE', class_WoE),
                           ('RLog', LogisticRegression(class_weight = class_weight))
                          ]
                   )

    # Muestra de Parámetros
    WoE_threshold = [0.02, 0.1, 0.3]
    WoE_bins = [10, 15, 20]
    max_iter = [10, 50, 100, 500, 1000]
    C = [.01, .05, 1, 2, 4, 6, 8, 10]
    solver = ['saga', 'liblinear']
    penalty = ['l1', 'l2']

    # Diccionario que contiene los parámetros de la clase WoE y RLog
    parameters = dict(WoE__threshold = WoE_threshold,
                      WoE__bins = WoE_bins,
                      RLog__max_iter = max_iter, 
                      RLog__C = C,
                      RLog__solver = solver, 
                      RLog__penalty = penalty)

    # Validación cruzada con RandomizedSearchCV y treno con las datas de train
    grid = RandomizedSearchCV(estimator=pipe, param_distributions=parameters,
                              cv=cv, return_train_score=True, n_iter=n_iter)
    grid_result = grid.fit(X_train, y_train)
    return grid_result


def cross_validation_RLog_Treated(X_train, y_train, class_TransformVar,
                                  class_CategoryEncoders, class_Estandarice,
                                  section, cv, n_iter):
    """Realiza validación cruzada con RandomizedSearchCV usando un canalizador que
       contiene clases que tratan la data (transforma, trata outliers, estandariza variables 
       numéricas, codifica variables categóricas e imputa en datos nulos con técnicas 
       multivariadas) y un modelo de RLog"""
    # Canalizador
    pipe = Pipeline(steps=[('TransVar', class_TransformVar),
                           ('CatEnc', class_CategoryEncoders),
                           ('Estandarice', class_Estandarice),
                           ('RLog', LogisticRegression(class_weight = None))
                          ]
                   )

    #======== Section = 'WoE_IterImp'
    # Diccionario que contiene los parámetros de la clase TransformVar, TreatmentOutliers,
    # Estandarice, CategoryEncoders (WoEEncoder), ImputerMultiVar(Iterative_Imputer) y
    # del modelo de regresión logística usando penalización 'elasticnet', 'l2' y 'l1'.
    parameters_RLog1 = dict(TransVar__method = ['yeo-johnson', 'QuantileTransformer', None],
                            CatEnc__method = ['WoEEncoder'],
                            CatEnc__threshold = [0.02, 0.1], 
                            CatEnc__bins = [10, 15, 20],
                            RLog__max_iter = [10, 50, 100, 500, 1000], 
                            RLog__C = [.01, .05, 1, 2, 4, 6, 8, 10], 
                            RLog__penalty = ['elasticnet'], 
                            RLog__l1_ratio = [ 0.1, 0.3, 0.5, 0.7, 0.9],
                            RLog__solver = ['saga'])
    
    parameters_RLog2 = dict(TransVar__method = ['yeo-johnson', 'QuantileTransformer', None],
                            CatEnc__method = ['WoEEncoder'],
                            CatEnc__threshold = [0.02, 0.1], 
                            CatEnc__bins = [10, 15, 20],
                            RLog__max_iter = [10, 50, 100, 500, 1000], 
                            RLog__C = [.01, .05, 1, 2, 4, 6, 8, 10], 
                            RLog__penalty = ['l2'],
                            RLog__l1_ratio = [ 0.1, 0.3, 0.5, 0.7, 0.9],
                            RLog__solver = ['saga', 'liblinear', 'newton-cg', 'lbfgs', 'sag'])
    
    parameters_RLog3 = dict(TransVar__method = ['yeo-johnson', 'QuantileTransformer', None],
                            CatEnc__method = ['WoEEncoder'],
                            CatEnc__threshold = [0.02, 0.1], 
                            CatEnc__bins = [10, 15, 20], 
                            RLog__max_iter = [10, 50, 100, 500, 1000], 
                            RLog__C = [.01, .05, 1, 2, 4, 6, 8, 10], 
                            RLog__penalty = ['l1'],
                            RLog__l1_ratio = [ 0.1, 0.3, 0.5, 0.7, 0.9],
                            RLog__solver = ['saga', 'liblinear'])
    
    #======== Section = 'GLMM_IterImp'
    # Diccionario que contiene los parámetros de la clase TransformVar, TreatmentOutliers,
    # Estandarice, CategoryEncoders (GLMMEncoder), ImputerMultiVar(Iterative_Imputer) y
    # del modelo de regresión logística usando penalización 'elasticnet', 'l2' y 'l1'.
    parameters_RLog10 = dict(TransVar__method = ['yeo-johnson', 'QuantileTransformer', None],
                            CatEnc__method = ['GLMMEncoder'], 
                            RLog__max_iter = [10, 50, 100, 500, 1000], 
                            RLog__C = [.01, .05, 1, 2, 4, 6, 8, 10], 
                            RLog__penalty = ['elasticnet'], 
                            RLog__l1_ratio = [ 0.1, 0.3, 0.5, 0.7, 0.9],
                            RLog__solver = ['saga'])
    
    parameters_RLog11 = dict(TransVar__method = ['yeo-johnson', 'QuantileTransformer', None],
                            CatEnc__method = ['GLMMEncoder'],
                            RLog__max_iter = [10, 50, 100, 500, 1000], 
                            RLog__C = [.01, .05, 1, 2, 4, 6, 8, 10], 
                            RLog__penalty = ['l2'],
                             RLog__l1_ratio = [ 0.1, 0.3, 0.5, 0.7, 0.9],
                            RLog__solver = ['saga', 'liblinear', 'newton-cg', 'lbfgs', 'sag'])
    
    parameters_RLog12 = dict(TransVar__method = ['yeo-johnson', 'QuantileTransformer', None],
                            CatEnc__method = ['GLMMEncoder'],
                            RLog__max_iter = [10, 50, 100, 500, 1000], 
                            RLog__C = [.01, .05, 1, 2, 4, 6, 8, 10], 
                            RLog__penalty = ['l1'],
                            RLog__l1_ratio = [ 0.1, 0.3, 0.5, 0.7, 0.9],
                            RLog__solver = ['saga', 'liblinear'])
    
    # Guardamos los parámetros de cada sección en un diccionario
    parameters_WoE_IterImp = {'elastic_net': parameters_RLog1, 'l2': parameters_RLog2, 
                              'l1': parameters_RLog3}
    parameters_GLMM_IterImp = {'elastic_net': parameters_RLog10, 'l2': parameters_RLog11, 
                               'l1': parameters_RLog12}
    
    # Validación cruzada con RandomizedSearchCV por cada una de las secciones y 
    # treno con las datas de train
    if section=='WoE_IterImp':
        grid_result = {}
        for parameters in parameters_WoE_IterImp.keys():
            grid = RandomizedSearchCV(estimator=pipe, 
                                      param_distributions=parameters_WoE_IterImp[parameters],
                                      cv=cv, return_train_score=True, n_iter=n_iter)
            grid_result[parameters] = grid.fit(X_train, y_train)
        return grid_result
    
    if section=='GLMM_IterImp':
        grid_result = {}
        for parameters in parameters_GLMM_IterImp.keys():
            grid = RandomizedSearchCV(estimator=pipe, 
                                      param_distributions=parameters_GLMM_IterImp[parameters],
                                      cv=cv, return_train_score=True, n_iter=n_iter)
            grid_result[parameters] = grid.fit(X_train, y_train)
        return grid_result
    

def cross_validation_XGB_Treated(X_train, y_train, class_TransformVar,
                                 class_CategoryEncoders, class_Estandarice,
                                 section, cv, n_iter):
    """Realiza validación cruzada con RandomizedSearchCV usando un canalizador que
       contiene clases que tratan la data (transforma, trata outliers, estandariza variables 
       numéricas, codifica variables categóricas e imputa en datos nulos con técnicas 
       multivariadas) y un model XGB"""
    # Canalizador
    pipe = Pipeline(steps=[('TransVar', class_TransformVar),
                           ('CatEnc', class_CategoryEncoders),
                           ('Estandarice', class_Estandarice),
                           ('XGB', XGBClassifier(n_jobs=15,
                                                 objective = 'binary:logistic'
                                                )
                           )
                          ]
                   )
    
    #======== Section = 'CatBoost_IterImp'
    # Diccionario que contiene los parámetros de la clase TransformVar, TreatmentOutliers,
    # Estandarice, CategoryEncoders (CatBoostEncoder), ImputerMultiVar(Iterative_Imputer) y
    # del modelo XGB.
    parameters_CatBoost_IterImp = dict(TransVar__method = ['yeo-johnson', 'QuantileTransformer', None],
                                        CatEnc__method = ['CatBoostEncoder'], 
                                        XGB__max_depth = [2, 4], 
                                        XGB__learning_rate = [0.001, 0.01, 0.1, 0.2, 0.3],
                                        XGB__subsample = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                                        XGB__colsample_bytree = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                                        XGB__colsample_bylevel = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                                        XGB__min_child_weight = [0.5, 1.0, 3.0, 5.0, 7.0, 10.0, 13, 21],
                                        XGB__gamma = [0, 0.25, 0.5, 1.0],
                                        XGB__reg_lambda = [10.0, 50.0, 100.0, 500.0],
                                        XGB__n_estimators = [50, 100, 200, 500])
    
    #======== Section = 'Target_IterImp'
    # Diccionario que contiene los parámetros de la clase TransformVar, TreatmentOutliers,
    # Estandarice, CategoryEncoders (TargetEncoder), ImputerMultiVar(Iterative_Imputer) y
    # del modelo XGB.
    parameters_Target_IterImp = dict(TransVar__method = ['yeo-johnson', 'QuantileTransformer', None],
                                    CatEnc__method = ['TargetEncoder'],
                                    CatEnc__min_samples_leaf = [1, 2, 4, 6],
                                    XGB__max_depth = [2, 4], 
                                    XGB__learning_rate = [0.001, 0.01, 0.1, 0.2, 0.3],
                                    XGB__subsample = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                                    XGB__colsample_bytree = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                                    XGB__colsample_bylevel = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                                    XGB__min_child_weight = [0.5, 1.0, 3.0, 5.0, 7.0, 10.0, 13, 21],
                                    XGB__gamma = [0, 0.25, 0.5, 1.0],
                                    XGB__reg_lambda = [10.0, 50.0, 100.0, 500.0],
                                    XGB__n_estimators = [50, 100, 200, 500])
    
    # Validación cruzada con RandomizedSearchCV por cada una de las secciones y 
    # treno con las datas de train
    if section=='CatBoost_IterImp':
        grid = RandomizedSearchCV(estimator=pipe, 
                                  param_distributions=parameters_CatBoost_IterImp,
                                  cv=cv, return_train_score=True, n_iter=n_iter)
        grid_result = grid.fit(X_train, y_train)
        return grid_result
    
    if section=='Target_IterImp':
        grid = RandomizedSearchCV(estimator=pipe, 
                                  param_distributions=parameters_Target_IterImp,
                                  cv=cv, return_train_score=True, n_iter=n_iter)
        grid_result = grid.fit(X_train, y_train)
        return grid_result    
    
    
## Creamos una función que muestre la comparación de los resultados  
## en todos los experimentos tanto de train como de test 
def summary(data_results):
    ind_test = []
    ind_train = []
    for ind in data_results.index:
        if 'Data Test' in ind:
            ind_test.append(ind)
        else:
            ind_train.append(ind)
    data_results_test = data_results.T[ind_test].T
    data_results_train = data_results.T[ind_train].T
    new_ind_test = []
    new_ind_train = []
    new_col_test = []
    new_col_train = []
    for ind in data_results_test.index:
        ind = ind.replace("Data Test ","")
        new_ind_test.append(ind)
    for col in data_results_test.columns:
        col = col + " Test"
        new_col_test.append(col)
    for ind in data_results_train.index:
        ind = ind.replace("Data Train ","")
        new_ind_train.append(ind)
    for col in data_results_train.columns:
        col = col + " Train"
        new_col_train.append(col)
    data_results_test.index = new_ind_test    
    data_results_train.index = new_ind_train
    data_results_test.set_axis(new_col_test,axis='columns', inplace=True)
    data_results_train.set_axis(new_col_train,axis='columns', inplace=True)
    data_results_train_test = pd.concat([data_results_train,data_results_test],axis=1)
    data_results_train_test['Dif_GINI'] = (data_results_train_test['GINI Train'] -
                                           data_results_train_test['GINI Test'])
    return(data_results_train_test)
     