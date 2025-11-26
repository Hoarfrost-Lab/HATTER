import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import pandas as pd
import math


def make_loss_plot(epoch_losses, val_losses=None, file_path='./lossplot.png', model_name='lookingglassv2'):
    df = pd.DataFrame()

    df['loss'] = epoch_losses
    if val_losses is not None:
        df['eval_loss'] = val_losses
        ylim_max = max([max(epoch_losses), max(val_losses)])
    else:
        ylim_max = max(epoch_losses)
    
    df['epoch'] = list(range(len(epoch_losses)))
    
    # plot the data using seaborn
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    #ax.set(xticks=list(range(0,len(epoch_losses)+1, 5)))
    #ax.set(xticks=list(range(0,len(epoch_losses)+1, 500)))
    ax.set_ylim(0.0, ylim_max)

    sns.lineplot(x='epoch', y='loss', data=df, label='Loss')

    if val_losses is not None:
        sns.lineplot(x='epoch', y='eval_loss', data=df, label='Eval Loss')

    plt.title('Training Loss over Epochs for {}'.format(model_name))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(title="Metrics", loc='best')
    plt.savefig(file_path)

def make_accuracy_plot(accuracies, balanced_accuracies=None, file_path='./accplot.png', model_name='lookingglassv2'):
    df = pd.DataFrame()

    df['accuracy'] = accuracies
    if balanced_accuracies is not None:
        df['balanced_accuracy'] = balanced_accuracies
    
    df['epoch'] = list(range(len(accuracies)))
    
    # plot the data using seaborn
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(figsize=(8, 6))
    #ax.set_xlim(0, len(accuracies))
    #ax.set(xticks=list(range(0,len(accuracies)+1, 5)))
    #ax.set(xticks=list(range(0,len(accuracies)+1, 500)))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax.set_ylim(0.0, 1.0)

    sns.lineplot(x='epoch', y='accuracy', data=df, label='Accuracy')
    
    if balanced_accuracies is not None:
        sns.lineplot(x='epoch', y='balanced_accuracy', data=df, label='Balanced Accuracy')

    plt.title('Accuracy over Epochs for {}'.format(model_name))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(title="Metrics", loc='best')
    plt.savefig(file_path)

def make_metrics_plot(aucs=None, prcs=None, precs=None, recs=None, f1s=None, hps=None, hrs=None, hf1s=None, file_path='./metrics_plot.png', model_name='lookingglassv2'):
    df = pd.DataFrame()

    if aucs is not None:
        df['auc'] = aucs
    if prcs is not None:
        df['prc'] = prcs
    if precs is not None:
        df['precision'] = precs
    if recs is not None:
        df['recall'] = recs
    if f1s is not None:
        df['f1-score'] = f1s
    if hps is not None:
        df['h-precision'] = hps
    if hrs is not None:
        df['h-recall'] = hrs
    if hf1s is not None:
        df['h-f1-score'] = hf1s
    
    df['epoch'] = list(range(len(df)))
    
    # plot the data using seaborn
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(figsize=(8, 6))
    #ax.set_xlim(0, len(aucs))
    #ax.set(xticks=list(range(0,len(aucs)+1, 5)))
    #ax.set(xticks=list(range(0,len(aucs)+1, 500)))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax.set_ylim(0.0, 1.0)

    if aucs is not None:
        sns.lineplot(x='epoch', y='auc', data=df, label='AUROC')
    if prcs is not None:
        sns.lineplot(x='epoch', y='prc', data=df, label='AUPRC')
    if precs is not None:
        sns.lineplot(x='epoch', y='precision', data=df, label='Precision')
    if recs is not None:
        sns.lineplot(x='epoch', y='recall', data=df, label='Recall')
    if f1s is not None:
        sns.lineplot(x='epoch', y='f1-score', data=df, label='F1-Score')
    if hps is not None:
        sns.lineplot(x='epoch', y='h-precision', data=df, label='h-Precision')
    if hrs is not None:
        sns.lineplot(x='epoch', y='h-recall', data=df, label='h-Recall')
    if hf1s is not None:
        sns.lineplot(x='epoch', y='h-f1-score', data=df, label='h-F1-Score')
    
    plt.title('Macro Metrics over Epochs for {}'.format(model_name))
    plt.xlabel('Epochs')
    plt.ylabel('Macro Metrics')
    plt.legend(title="Metrics", loc='best')
    plt.savefig(file_path)

def plot_pca_by_uncertainty(pca, X_pool, X_query, scores, instance=-1, path='.'):
    transformed_pool = pca.transform(X_pool)
    transformed_queries = pca.transform(X_query)

    if instance == -1:
        title = 'PCA by Uncertainty'
        save = path+'/pca_by_uncertainty.png'
    else:
        title = 'PCA by Uncertainty: Round {}'.format(instance)
        save = path+'/pca_by_uncertainty_round_{}.png'.format(instance)

    with plt.style.context('seaborn-v0_8-whitegrid'):
        plt.figure(figsize=(8, 8))
        plt.scatter(transformed_pool[:, 0], transformed_pool[:, 1], c=scores, cmap='viridis', s=50)
        plt.colorbar()
        plt.scatter(transformed_queries[:, 0], transformed_queries[:, 1], c='r', s=200, label='queried')
        plt.title(title)
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.legend()
        plt.savefig(save)

def plot_pca_by_class(pca, label_encoder, X_pool, y_pool, X_query, y_query, instance=-1, path='.'):
    transformed_pool = pca.transform(X_pool)
    transformed_queries = pca.transform(X_query)
    transformed_y_pool = label_encoder.transform(y_pool)
    transformed_y_query = label_encoder.transform(y_query)

    if instance == -1:
        title = 'PCA by Class'
        save = path+'/pca_by_class.png'
    else:
        title = 'PCA by Class: Round {}'.format(instance)
        save = path+'/pca_by_class_round_{}.png'.format(instance)

    with plt.style.context('seaborn-v0_8-whitegrid'):
        plt.figure(figsize=(8, 8))
        plt.scatter(transformed_pool[:, 0], transformed_pool[:, 1], c=transformed_y_pool, cmap='viridis', s=50)
        plt.scatter(transformed_queries[:, 0], transformed_queries[:, 1], c=transformed_y_query, cmap='viridis', s=200)
        plt.title(title)
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.savefig(save)

def plot_pca_by_learner(pca, X_pool, X_query, committee, instance=1, path='.', n_members=3):
    transformed_pool = pca.transform(X_pool)
    transformed_queries = pca.transform(X_query)

    with plt.style.context('seaborn-v0_8-whitegrid'):
        plt.figure(figsize=(n_members*7, 7))
        for learner_idx, learner in enumerate(committee):
            plt.subplot(1, n_members, learner_idx + 1)
            plt.scatter(x=transformed_pool[:, 0], y=transformed_pool[:, 1], c=learner.predict(X_pool), cmap='viridis', s=50)
            plt.scatter(x=transformed_queries[:, 0], y=transformed_queries[:, 1], c=learner.predict(X_query), cmap='viridis', s=200)
            plt.title('Learner no. %d Initial Predictions' % (learner_idx + 1))
            plt.xlabel('PCA 1')
            plt.ylabel('PCA 2')
        
        plt.savefig(path+'/pca_by_learner_round_{}.png'.format(instance))
