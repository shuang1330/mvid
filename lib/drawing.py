import numpy as np
import matplotlib.pyplot as plt

def hist_compare(datatable_pd, col_name):
    '''
    compare the column(col_name) value for different classes
    in the database(datatable_pd) and plot histograms/bar plot 
    for the pathogenic and benign class
    
    currently only support 'Pathogenic' and 'Benign' classes
    no legend in the plot
    '''
    col = datatable_pd[col_name]
    col_patho = col[datatable_pd['INFO']=='Pathogenic']
    col_benig = col[datatable_pd['INFO']=='Benign']
    if datatable_pd[col_name].dtype == np.object:
        x = datatable_pd[col_name].unique()
        y_path = []
        y_beni = []
        for name in x:
            y_name = datatable_pd['INFO'][datatable_pd[col_name]==name]
            y_path.append(y_name[y_name=='Pathogenic'].count())
            y_beni.append(y_name[y_name=='Benign'].count())
        plt.bar([str(name) for name in x],y_path,alpha=0.5)
        plt.bar([str(name) for name in x],y_beni,alpha=0.5)
    else:
        n,bins,patches = plt.hist(col_patho[~np.isnan(col_patho)],50,alpha=0.5)
        n,bins,patches = plt.hist(col_benig[~np.isnan(col_benig)],50,alpha=0.5)
    plt.show()