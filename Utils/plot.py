import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_box_and_hist(data, columns, length=5, dpi=100):
    if len(columns) > 1:
        fig, axs = plt.subplots(len(columns),2,sharex=False, figsize=(15, length*len(columns)), dpi=dpi)
        for i, ax in enumerate(axs):
            
            sns.boxplot(data=data, x=columns[i], zorder=2, ax=ax[0])
            ax[0].set_title(f'Boxplot of {columns[i].replace("_", " ").title()}')
            ax[0].grid(axis='x', linestyle='-', alpha=0.7, zorder=1)
            ax[0].set_xlabel(f'{columns[i].replace("_", " ").title()}')

            sns.histplot(data=data, x=columns[i], zorder=2, ax=ax[1], bins=50, kde=True)
            ax[1].set_title(f'Histogram of {columns[i].replace("_", " ").title()}')
            ax[1].set_xlabel(f'{columns[i].replace("_", " ").title()}')
            ax[1].grid( linestyle='-', alpha=0.7, zorder=1)
        fig.tight_layout()
        fig.show()
    
    elif len(columns) == 1:
        fig, ax = plt.subplots(len(columns),2,sharex=False, figsize=(15, length*len(columns)), dpi=dpi)
        sns.boxplot(data=data, x=columns[0], zorder=2, ax=ax[0])
        ax[0].set_title(f'Boxplot of {columns[0].replace("_", " ").title()}')
        ax[0].grid(axis='x', linestyle='-', alpha=0.7, zorder=1)
        ax[0].set_xlabel(f'{columns[0].replace("_", " ").title()}')

        sns.histplot(data=data, x=columns[0], zorder=2, ax=ax[1], bins=50, kde=True)
        ax[1].set_title(f'Histogram of {columns[0].replace("_", " ").title()}')
        ax[1].set_xlabel(f'{columns[0].replace("_", " ").title()}')
        ax[1].grid( linestyle='-', alpha=0.7, zorder=1)
        fig.tight_layout()
        fig.show()
    else:
        print("there's no column to display")
    

    
def plot_missing_values(data):
    missing_values = data.isna().sum(axis=0).sort_values(ascending=False)
    missing_values = missing_values[missing_values > 0]
    
    if len(missing_values) == 0:
        print("There's no missing value column!")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot (x=missing_values.values, y=missing_values.index, zorder=2, legend=False, ax=ax)
    #plt.xticks(rotation=90)
    ax.set_title('Missing Values Count')
    ax.set_ylabel('Variables') 
    ax.set_xlabel('Count')
    ax.set_xlim(0, missing_values.max() + 50)
    ax.grid(axis='x', linestyle='-', alpha=0.4, zorder=1)
    for p in ax.patches:
        ax.annotate(format(p.get_width(), '.0f'), 
                        (p.get_width(), p.get_y() + p.get_height() / 2.), 
                        ha='center',
                        va='center', 
                        xytext=(10, 0),
                        textcoords='offset points',
                        zorder=3)
        
    fig.tight_layout()

    return fig, ax

def DisplayConfusion(y_true, y_predicted, labels, normalize=None):
    try:
        cm = confusion_matrix(y_true, y_predicted,
                              labels=labels, normalize=normalize)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=labels)
        disp.plot()
        disp.ax_.set_title("Confusion matrix")
        plt.show()

    except:
        raise




