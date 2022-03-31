import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_geom_eval(a,b,mode="bond",s=0.01):
  

    
  fig, axs = plt.subplots(1,2,figsize=(16,5))
  axs = axs.flatten()
  
  # scatter plot
  ax = axs[0]
  ax.scatter(a,b,s=s)
  #sns.kdeplot(a,b,fill=True)
  if mode == "bond":
    ax.set_xlim(1,1.8)
    ax.set_ylim(1,1.8)
    units = "(ang.)"
  elif mode == "angle":
    ax.set_xlim(50,140)
    ax.set_ylim(50,140)
    units = "(deg.)"
  ax.plot([0,200],[0,200],color="black")
  ax.set_xlabel("Reference "+units,fontsize=14)
  ax.set_ylabel("Predicted "+units,fontsize=14)
  
  
  # histogram
  ax = axs[1]
  sns.histplot(a-b,ax=ax,kde=True,stat="density")
  if mode == "bond":
    ax.set_xlim(-3,3)
  elif mode == "angle":
    ax.set_xlim(-20,20)
  ax.set_xlabel("Error (reference-predicted) "+units,fontsize=14)
  ax.set_ylabel("Density estimate",fontsize=14)
  mae = np.abs(a-b).mean()
  ax.set_title("MAE:"+str(mae))
