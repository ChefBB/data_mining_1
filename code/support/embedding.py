"""
!!!NO FINAL SOLUTION TO EMBEDDING THE GENRE COLUMN YET!!!

This module contains functions to generate embeddings for the genres of movies using Word2Vec.
"""
import pandas as pd
from gensim.models import Word2Vec
import numpy as np
from node2vec import Node2Vec
import networkx as nx



def embedding(train: pd.DataFrame, test: pd.DataFrame | None=None) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Applies embedding to the training and testing datasets.
    This includes generating embeddings for the genres using Word2Vec and Node2Vec.

    Parameters
    ----------
    train : pd.DataFrame
        The training dataset.
    test : pd.DataFrame | None
        The testing dataset. Defaults to None.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame | None]: The training and testing datasets with embeddings.
    """
    one_hot_genres = pd.get_dummies(train['genres'].apply(pd.Series).stack()).groupby(level=0).sum()
    one_hot_genres = one_hot_genres.add_prefix('genre_').astype(int)
    train = pd.concat([train, one_hot_genres], axis=1)
    
    one_hot_titletype = pd.get_dummies(train['titleType'], prefix='titleType')
    train = pd.concat([train, one_hot_titletype], axis=1)
    
    if test is None:
        return train, None
    
    one_hot_genres = pd.get_dummies(test['genres'].apply(pd.Series).stack()).groupby(level=0).sum()
    one_hot_genres = one_hot_genres.add_prefix('genre_').astype(int)
    test = pd.concat([test, one_hot_genres], axis=1)
    
    one_hot_titletype = pd.get_dummies(test['titleType'], prefix='titleType')
    test = pd.concat([test, one_hot_titletype], axis=1)
    
    return train, test



# Step 2: Generate Embeddings for Each Sample
def get_w2v_model(df: pd.DataFrame, name: str='models/genre_w2v_model.model') -> Word2Vec:
    """
    Generate the Word2Vec model for the genres column in the given DataFrame.
    Save model to the specified name.
    ----------
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the genres.

    name : str
        The name of the model file.
        
    
    ----------
    Returns
    -------
    Word2Vec: The Word2Vec model.
    """
    w2v_model = Word2Vec(sentences=df['genre'], vector_size=100, window=3, min_count=1, workers=4, sg=0)
    w2v_model.save(name)
    
    return w2v_model


def get_genre_embedding(genre: list[str], w2v_model: Word2Vec) -> np.ndarray:
    """
    Generate the embedding for a single entry using the Word2Vec model.
    ----------
    Parameters
    ----------
    genre : list[str]
        The genre to generate the embedding for.

    w2v_model : Word2Vec
        The Word2Vec model.
        
    
    ----------
    Returns
    -------
    np.ndarray: The embedding for the genre.
    """
    # Get the embeddings for each word in the genre
    embeddings = [w2v_model.wv[word] for word in genre if word in w2v_model.wv]
    
    # Return the mean of the embeddings
    return np.mean(embeddings, axis=0) if embeddings else np.zeros(w2v_model.vector_size)


def get_node2vec_model(graph: nx.Graph, name: str='models/genre_node2vec_model.model') -> Node2Vec:
    """
    Generate the Node2Vec model for the given graph.
    Save model to the specified name.
    ----------
    Parameters
    ----------
    graph : nx.Graph
        The graph representing the genres.

    name : str
        The name of the model file.
        
    ----------
    Returns
    -------
    Node2Vec: The Node2Vec model.
    """
    node2vec = Node2Vec(graph, dimensions=100, walk_length=30, num_walks=200, workers=4)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    model.save(name)
    
    return model


def get_node_embedding(node: str, node2vec_model: Node2Vec) -> np.ndarray:
    """
    Generate the embedding for a single node using the Node2Vec model.
    ----------
    Parameters
    ----------
    node : str
        The node to generate the embedding for.

    node2vec_model : Node2Vec
        The Node2Vec model.
        
    ----------
    Returns
    -------
    np.ndarray: The embedding for the node.
    """
    if node in node2vec_model.wv:
        return node2vec_model.wv[node]
    else:
        return np.zeros(node2vec_model.wv.vector_size)

# Apply the function to generate embeddings for each row
# df['genre_embedding'] = df['genre'].apply(lambda x: get_genre_embedding(x, w2v_model))
