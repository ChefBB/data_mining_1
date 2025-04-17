"""
This module serves as the central file for outlier detection in the project.

It provides the core functionality and utilities required to identify and handle
outliers in datasets. The methods implemented here are designed to work with 
various types of data and can be customized for specific use cases. This file 
acts as the backbone for outlier detection workflows, ensuring consistency 
and reusability across the project.
"""

import pandas as pd

import plotly.express as px
from plotly import graph_objects as go
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt


feats_to_keep = [
    'startYear', 'runtimeMinutes',
    'totalCredits', 'reviewsTotal',
    'numRegions', 'ratingCount',
    'castNumber', 'companiesNumber', 'averageRating', 'externalLinks',
    'writerCredits', 'directorsCredits', 'totalMedia',
    'totalNominations',
    'regions_freq_enc', 'regions_EU', 'regions_NA', 'regions_AS',
    'regions_AF', 'regions_OC', 'regions_SA', 'regions_UNK',
    'countryOfOrigin_freq_enc', 'countryOfOrigin_NA', 'countryOfOrigin_AF',
    'countryOfOrigin_AS', 'countryOfOrigin_EU', 'countryOfOrigin_OC',
    'countryOfOrigin_SA', 'countryOfOrigin_UNK',
    # 'endYear', 
]


def outlier_detection(train: pd.DataFrame, test: pd.DataFrame | None=None) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    !!!TODO!!!
    
    !!! THIS IS A PLACEHOLDER FUNCTION !!!
    
    Applies outlier detection on the training dataset and optionally on the test dataset.

    Parameters
    ----------
    train : pd.DataFrame
        The training dataset.
        
    test : pd.DataFrame | None
        The testing dataset. If None, only the training dataset is processed.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame | None]
        The processed training dataset and the processed testing dataset (if provided).
    """
    
    # Placeholder for outlier detection logic
    # This should be replaced with actual implementation
    # For now, we just return the datasets as they are
    return train if test is None else (train, test)


def plot_3d_outliers(
    df: pd.DataFrame, title_type: str = None, feats: list[str] = feats_to_keep,
    outlier_col: str = 'outlier'):
    """
    Draws an interactive 3D scatter plot of the PCA components, highlighting outliers in red and others in blue.
    Allows filtering by titleType.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data to be plotted.
    title_type : str, optional
        The titleType to filter the data. If None, the entire dataset is used.
    feats : list[str], optional
        The features to use for PCA. Defaults to feats_to_keep_iso_forest.

    Returns
    -------
    None
    """
    # Filter by titleType if specified
    if title_type:
        df = df[df['titleType'] == title_type]

    # Apply PCA to reduce to 3 components
    pca = PCA(n_components=3, random_state=42)
    pca_components = pca.fit_transform(df[feats])
    df_pca = pd.DataFrame(pca_components, columns=['PCA1', 'PCA2', 'PCA3'], index=df.index)

    # Add outlier and titleType information
    df_pca['outlier'] = df[outlier_col]
    df_pca['titleType'] = df['titleType']

    # Plot using Plotly
    fig = px.scatter_3d(
        df_pca,
        x='PCA1',
        y='PCA2',
        z='PCA3',
        color=df_pca['outlier'].map({1: 'blue', -1: 'red'}),
        symbol='titleType',
        title="3D PCA Scatter Plot with Outliers Highlighted",
        labels={'color': 'Outlier', 'symbol': 'Title Type'},
        opacity=0.3,  # Correct parameter for transparency
    )

    # Update the legend to separate columns for titleType and outlier
    # Add PCA loading vectors to the plot
    loadings = pca.components_.T * 10  # Scale up the loadings for better visibility
    # Update the legend to separate columns for titleType and outlier
    # Add PCA loading vectors to the plot
    for i, feature in enumerate(feats):
        fig.add_trace(
            go.Scatter3d(
                x=[0, loadings[i, 0]],
                y=[0, loadings[i, 1]],
                z=[0, loadings[i, 2]],
                mode='lines+text',
                line=dict(color='black', width=2),
                text=[None, feature],
                textposition='top center',
                name=f"Loading: {feature}"
            )
        )

    # Update layout for legend and other settings
    # fig.update_layout(
    #     legend=dict(
    #         title=dict(text="Legend"),
    #         tracegroupgap=10,
    #         itemsizing='constant',
    #         orientation="h",
    #         xanchor="center",
    #         x=0.5,
    #         y=-0.1,
    #     )
    # )
    fig.update_traces(marker_size=3)
    fig.show()
    
    
def plot_pairplot_pca(df: pd.DataFrame, feats: list[str] = feats_to_keep, outlier_col: str = 'outlier'):
    """
    Plots a pairplot of the first 4 principal components using seaborn, highlighting outliers.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data to be plotted.
    feats : list[str], optional
        The features to use for PCA. Defaults to feats_to_keep.
    outlier_col : str, optional
        The column indicating outliers. Defaults to 'outlier'.

    Returns
    -------
    None
    """
    # Apply PCA to reduce to 4 components
    pca = PCA(n_components=5, random_state=42)
    pca_components = pca.fit_transform(df[feats])
    df_pca = pd.DataFrame(pca_components, columns=['PCA1', 'PCA2', 'PCA3', 'PCA4', 'PC5'], index=df.index)

    # Add outlier information
    df_pca['outlier'] = df[outlier_col].map({1: 'Inlier', -1: 'Outlier'})

    # Plot using seaborn pairplot
    g = sns.pairplot(
        df_pca,
        vars=['PCA1', 'PCA2', 'PCA3', 'PCA4', 'PC5'],
        hue='outlier',
        palette={'Inlier': 'blue', 'Outlier': 'red'},
        diag_kind='kde',
        plot_kws={'alpha': 0.1, 's': 5, 'edgecolor': 'none'},
    )
    
    # Add grid to each graph
    for ax in g.axes.flatten():
        ax.grid(True)

    plt.suptitle("Pairplot of First 4 Principal Components", y=1.02)
    plt.show()
