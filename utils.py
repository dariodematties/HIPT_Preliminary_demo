import os
import random
import numpy as np

import torch

from sklearn.manifold import TSNE
import plotly.express as px

# Using dash for advanced image hovering (optional but recommended)
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import base64
import io
from PIL import Image

from concurrent.futures import ProcessPoolExecutor

def prepare_crops(tensor_dir = './Crops/', # Folder with crop_*.pt files
                  embedding_path = './output_features/output_features_2k.pt'): # Embedding file
    # First of all, check if vectors, crops and labels are already stored in respective files
    if os.path.exists('vectors_crops.npy') and os.path.exists('crops.npy') and os.path.exists('labels_crops.npy'):
        vectors = np.load('vectors_crops.npy')
        crops = np.load('crops.npy', allow_pickle=True)
        labels = np.load('labels_crops.npy', allow_pickle=True)
        return vectors, crops, labels

    # If not, proceed with loading and processing the data
    # Load embeddings
    embeddings_dict = torch.load(embedding_path)
    
    # Prepare data lists
    vectors, crops, labels = [], [], []
    
    
    # Iterate and preprocess data
    for filename, embedding in embeddings_dict.items():
        tensor_path = os.path.join(tensor_dir, filename)
        if not os.path.exists(tensor_path):
            continue
    
        tensor = torch.load(tensor_path)  # Load 2048x2048 tensor
        # For example, to reduce to 512x512 (1/4 of each dimension)
        stride = 8
        subsampled_tensor = tensor[::stride, ::stride]
    
        crops.append(subsampled_tensor)
        labels.append(f'{filename}')
        vectors.append(embedding.numpy())
    
    vectors = np.array(vectors)
    crops = np.array(crops)
    # Before returning, save the vectors, crops and labels to respective files for future use
    np.save('vectors_crops.npy', vectors)
    np.save('crops.npy', crops)
    np.save('labels_crops.npy', labels)

    return vectors, crops, labels

def prepare_patches(tensor_dir = './Crops/',  # Folder with crop_*.pt files
                 embedding_path = './output_features/output_features.pt'): 
    # First of all, check if vectors, patches and labels are already stored in respective files
    if os.path.exists('vectors_patches.npy') and os.path.exists('patches.npy') and os.path.exists('labels_patches.npy'):
        vectors = np.load('vectors_patches.npy')
        patches = np.load('patches.npy', allow_pickle=True)
        labels = np.load('labels_patches.npy', allow_pickle=True)
        return vectors, patches, labels
    # If not, proceed with loading and processing the data
    # Load embeddings
    embeddings_dict = torch.load(embedding_path)
    
    # Prepare data lists
    vectors, patches, labels = [], [], []
   
    # Iterate and preprocess data
    for filename, embedding in embeddings_dict.items():
        tensor_path = os.path.join(tensor_dir, filename)
        if not os.path.exists(tensor_path):
            continue
    
        tensor = torch.load(tensor_path)  # Load 2048x2048 tensor
        tensor_reshaped = tensor.reshape(8, 256, 8, 256)
        emb = embedding.reshape(8, 8, 384)
    
        for i in range(8):
            for j in range(8):
                vectors.append(emb[i, j].numpy())  # 384-dimensional vector
                patch = tensor_reshaped[i, :, j, :].numpy()
                patches.append(patch)
                labels.append(f'{filename}_patch_{i}_{j}')
    
    vectors = np.array(vectors)

    # Before returning, save the vectors, patches and labels to respective files for future use
    np.save('vectors_patches.npy', vectors)
    np.save('patches.npy', patches)
    np.save('labels_patches.npy', labels)

    return vectors, patches, labels

def plot_crops(vectors, crops, labels):
    # Apply t-SNE only if there is non already applied
    # check id vectors_2d_crops.npy file exists
    if os.path.exists('vectors_2d_crops.npy'):
        vectors_2d = np.load('vectors_2d_crops.npy')
    else:        # Apply t-SNE
        # Note: This can be computationally expensive for large datasets
        # Consider using PCA first to reduce dimensionality before t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        vectors_2d = tsne.fit_transform(vectors)
        np.save('vectors_2d_crops.npy', vectors_2d)

    # Normalize crops for visualization
    crops_normalized = [(crop - crop.min()) / (crop.max() - crop.min()) for crop in crops]
    
    # Create interactive plot
    fig = px.scatter(x=vectors_2d[:, 0], y=vectors_2d[:, 1], hover_name=labels)
    
    # Create Dash app
    #app = dash.Dash(__name__)
    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    
    app.layout = html.Div([
        dcc.Graph(id='scatter-plot', figure=fig, style={'width': '70%', 'display': 'inline-block'}),
        html.Img(id='hover-image', style={'width': '256px', 'height': '256px', 'display': 'inline-block', 'vertical-align': 'top', 'margin-left': '20px'})
    ])
    
    # Convert crop to image
    def crop_to_img(crop, cmap='viridis'):
        import matplotlib.pyplot as plt
        import numpy as np
        import io
        import base64
        from PIL import Image
    
        cmap_func = plt.get_cmap(cmap)
        crop_colored = cmap_func(crop.squeeze())[:, :, :3]  # Apply colormap and discard alpha channel
        img = (crop_colored * 255).astype(np.uint8)
    
        image_pil = Image.fromarray(img)
        buff = io.BytesIO()
        image_pil.save(buff, format="PNG")
        encoded_image = base64.b64encode(buff.getvalue()).decode()
        return f'data:image/png;base64,{encoded_image}'
    
    @app.callback(
        Output('hover-image', 'src'),
        Input('scatter-plot', 'hoverData')
    )
    def display_hover_image(hoverData):
        if hoverData is None:
            return dash.no_update
    
        point_idx = hoverData['points'][0]['pointIndex']
        return crop_to_img(crops_normalized[point_idx])

    return app

def plot_patches(vectors, patches, labels):
    # Apply t-SNE only if there is non already applied
    # check id vectors_2d_patches.npy file exists
    if os.path.exists('vectors_2d_patches.npy'):
        vectors_2d = np.load('vectors_2d_patches.npy')
    else:        # Apply t-SNE
        # Note: This can be computationally expensive for large datasets
        # Consider using PCA first to reduce dimensionality before t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        vectors_2d = tsne.fit_transform(vectors)
        np.save('vectors_2d_patches.npy', vectors_2d)
    
    # Normalize patches for visualization
    patches_normalized = [(patch - patch.min()) / (patch.max() - patch.min()) for patch in patches]
    
    # Create interactive plot
    fig = px.scatter(x=vectors_2d[:, 0], y=vectors_2d[:, 1], hover_name=labels)
    
    # Create Dash app
    #app = dash.Dash(__name__)
    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    
    app.layout = html.Div([
        dcc.Graph(id='scatter-plot', figure=fig, style={'width': '70%', 'display': 'inline-block'}),
        html.Img(id='hover-image', style={'width': '256px', 'height': '256px', 'display': 'inline-block', 'vertical-align': 'top', 'margin-left': '20px'})
    ])
    
    # Convert patch to image
    def patch_to_img(patch, cmap='viridis'):
        import matplotlib.pyplot as plt
        import numpy as np
        import io
        import base64
        from PIL import Image
    
        cmap_func = plt.get_cmap(cmap)
        patch_colored = cmap_func(patch.squeeze())[:, :, :3]  # Apply colormap and discard alpha channel
        img = (patch_colored * 255).astype(np.uint8)
    
        image_pil = Image.fromarray(img)
        buff = io.BytesIO()
        image_pil.save(buff, format="PNG")
        encoded_image = base64.b64encode(buff.getvalue()).decode()
        return f'data:image/png;base64,{encoded_image}'
    
    @app.callback(
        Output('hover-image', 'src'),
        Input('scatter-plot', 'hoverData')
    )
    def display_hover_image(hoverData):
        if hoverData is None:
            return dash.no_update
    
        point_idx = hoverData['points'][0]['pointIndex']
        return patch_to_img(patches_normalized[point_idx])

    return app
###############################################
######## parallel processing functions ########
###############################################
def process_single_embedding(args):
    filename, embedding, tensor_dir = args

    tensor_path = os.path.join(tensor_dir, filename)
    if not os.path.exists(tensor_path):
        return [], [], []

    tensor = torch.load(tensor_path)
    tensor_reshaped = tensor.reshape(8, 256, 8, 256)
    emb = embedding.reshape(8, 8, 384)

    vectors_local, patches_local, labels_local = [], [], []

    for i in range(8):
        for j in range(8):
            vectors_local.append(emb[i, j].numpy())
            patch = tensor_reshaped[i, :, j, :].numpy()
            patches_local.append(patch)
            labels_local.append(f'{filename}_patch_{i}_{j}')

    return vectors_local, patches_local, labels_local

def prepare_patches_parallel(tensor_dir='./Crops/', embedding_path='./output_features/output_features.pt', subsample_ratio=1.0):
    if os.path.exists('vectors_patches.npy') and os.path.exists('patches.npy') and os.path.exists('labels_patches.npy'):
        vectors = np.load('vectors_patches.npy')
        patches = np.load('patches.npy', allow_pickle=True)
        labels = np.load('labels_patches.npy', allow_pickle=True)
        return vectors, patches, labels

    embeddings_dict = torch.load(embedding_path)

    filenames = list(embeddings_dict.keys())
    if subsample_ratio < 1.0:
        num_samples = int(len(filenames) * subsample_ratio)
        filenames = random.sample(filenames, num_samples)

    args_list = [(filename, embeddings_dict[filename], tensor_dir) for filename in filenames]

    vectors, patches, labels = [], [], []

    with ProcessPoolExecutor() as executor:
        results = executor.map(process_single_embedding, args_list)

    for vec_local, patch_local, label_local in results:
        vectors.extend(vec_local)
        patches.extend(patch_local)
        labels.extend(label_local)

    vectors = np.array(vectors)

    np.save('vectors_patches.npy', vectors)
    np.save('patches.npy', patches)
    np.save('labels_patches.npy', labels)

    return vectors, patches, labels
