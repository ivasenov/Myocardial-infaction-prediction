# We are writing some simple visualization tools that take file name as an input and produce the desired plots as an output

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


def plot_single_frame(frame_id, data):

    frame = data[frame_id]
    xyz = frame[:, :3] 
    category = frame[:, 3]
    category = category.astype(int)

    fig = px.scatter_3d(
    x=xyz[:, 0],
    y=xyz[:, 1],
    z=xyz[:, 2],
    color=category,
    color_continuous_scale="Viridis",  # or try "Turbo", "Plotly3", etc.
    title="First Frame Colored by Category Attribution",
    height=800,
    width=900,
    opacity=0.7
    )
    fig.update_traces(marker=dict(size=2))
    fig.update_layout(scene=dict(aspectmode='data'))
    fig.show()

    return

def frame_colors(frame, cat_to_color):
    cats = frame[:, 3]
    # if categories are floats like 1.0, they still map OK
    return [cat_to_color[c] for c in cats]


def visualise_patient(data, speed = 1, include_background = False):

    nymph = data
    T, N, D = nymph.shape
    visible = include_background

    all_cats = np.unique(nymph[:, :, 3])
    palette = px.colors.qualitative.Dark24  # plenty of distinct colors
    cat_to_color = {cat: palette[i % len(palette)] for i, cat in enumerate(all_cats)}

    frames = []
    for t in range(T):
        frame = nymph[t]
        xyz = frame[:, :3]
        colors = frame_colors(frame, cat_to_color)
        trace = go.Scatter3d(
            x=xyz[:, 0],
            y=xyz[:, 1],
            z=xyz[:, 2],
            mode='markers',
            marker=dict(
                size=2,          # small marker for many points
                opacity=0.8,
                color=colors     # list of color hex strings
            ),
            hoverinfo='skip'     # change to 'text' and add text if you want hover labels
        )
        frames.append(go.Frame(data=[trace], name=str(t)))

    # initial data = frame 0
    init_xyz = nymph[0][:, :3]
    init_colors = frame_colors(nymph[0], cat_to_color)
    init_trace = go.Scatter3d(
        x=init_xyz[:, 0],
        y=init_xyz[:, 1],
        z=init_xyz[:, 2],
        mode='markers',
        marker=dict(size=2, opacity=0.8, color=init_colors),
        hoverinfo='skip'
    )

    # layout with play/slider controls
    layout = go.Layout(
        title="Nymph life - 3D animation (colored by category)",
        scene=dict(
            xaxis=dict(title='X', visible=visible),
            yaxis=dict(title='Y', visible=visible),
            zaxis=dict(title='Z', visible=visible),
            aspectmode='data'  # equal scaling
        ),
        width=1000,
        height=800,
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'y': 1.05,
            'x': 0.1,
            'xanchor': 'right',
            'yanchor': 'top',
            'buttons': [
                {
                    'label': 'Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 400*speed, 'redraw': True},
                        'fromcurrent': True,
                        'transition': {'duration': 0}
                    }]
                },
                {
                    'label': 'Pause',
                    'method': 'animate',
                    'args': [[None], {
                        'frame': {'duration': 0, 'redraw': False},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                }
            ]
        }],
        sliders=[{
            'active': 0,
            'y': -0.05,
            'x': 0.1,
            'len': 0.9,
            'pad': {'b': 10, 't': 50},
            'steps': [
                {
                    'args': [[str(k)], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}],
                    'label': str(k),
                    'method': 'animate'
                } for k in range(T)
            ]
        }]
    )

    fig = go.Figure(data=[init_trace], frames=frames, layout=layout)

    # show in notebook / web browser
    fig.show()

    return