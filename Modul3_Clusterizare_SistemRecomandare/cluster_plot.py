from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import webbrowser
import plotly.graph_objects as go
import plotly.express as px
import nbformat
import pandas as pd
import numpy as np

class Plot():
    def __init__(self):
        pass

    def init_2d_plot(self, input_data, cluster_labels, product_links, unscaled_prices=None, cluster_centers=None):
        df = pd.DataFrame({'Performance(PCA) Score': input_data[:, 0],
                    'Relevance Score': input_data[:, 1],
                    'Cluster Label': cluster_labels,
                    'Link':product_links})

        text=df[['Cluster Label', 'Link']].apply(lambda x: [x[0], x[1]], axis=1)
        data_points = go.Scatter(
            x=df['Performance(PCA) Score'],
            y=df['Relevance Score'],
            mode='markers',
            marker=dict(
                color=df['Cluster Label'],
                colorscale='Jet',
                size=7,
                opacity=0.8
            ),
            hovertemplate='<b>Link:</b> %{text[1]} <br><b>Cluster:</b> %{text[0]}',
            text=df[['Cluster Label', 'Link']].apply(lambda x: [x[0], x[1]], axis=1),
            name='Products'
        )

        fig = go.Figure(data=[data_points])

        if cluster_centers is not None:
            cluster_centers = go.Scatter(
                x=cluster_centers[:, 0],
                y=cluster_centers[:, 1],
                mode='markers',
                marker=dict(
                    color='black',
                    size=10,
                    symbol='x'
                ),
                name='Cluster Centers'
            )
            fig.add_trace(cluster_centers)

            # fig = go.Figure(data=[data_points, cluster_centers])
        # else:

        fig.update_layout(title='Clusterization Results', xaxis_title='Performance(PCA) Score', yaxis_title='Relevance Score')
        fig.update_layout(hovermode='closest')

        fig.show()


    def init_3d_plot(self, input_data, unscaled_prices, cluster_labels, product_links, cluster_centers=None):
        fig = go.Figure(data=go.Scatter3d(
            x=input_data[:, 0],
            y=input_data[:, 1],
            z=unscaled_prices,
            mode='markers',
            marker=dict(
                color=cluster_labels,
                colorscale='Jet',
                size=5,
                opacity=1
            ),
            hovertext=[f'{product_links[i]}<br>Cluster:{cluster_labels[i]}' for i in range(len(product_links))],
        ))

        unique_labels, label_counts = np.unique(cluster_labels, return_counts=True)
        if cluster_centers is not None:
            fig.add_trace(go.Scatter3d(
                x=cluster_centers[0][:, 0],
                y=cluster_centers[0][:, 1],
                z=cluster_centers[1][:, 3],
                mode='markers',
                marker=dict(
                    color='red',
                    size=8,
                    symbol='cross',
                    opacity=1.0
                ),
                name='Cluster Centers',
                hovertext = [f'ID: {i}' for i in unique_labels],
            ))

        fig.update_layout(scene=dict(
            xaxis_title='Performance Score',
            yaxis_title='Relevance Score',
            zaxis_title='Price'
        ))

        fig.update_layout(title='Aglomerative Clustering')

        fig.show()


    def plot_dendrogram(self, model, **kwargs):
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # noduri frunza
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack(
            [model.children_, model.distances_, counts]
        ).astype(float)

        dendrogram(linkage_matrix, **kwargs)
