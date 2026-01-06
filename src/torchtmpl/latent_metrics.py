"""Latent space metrics for β-VAE evaluation.

Provides tools to evaluate:
1. Cluster quality (silhouette score, separation)
2. Interpolation quality (smoothness, coherence)
3. Latent space density (coverage, holes detection)
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.cluster import KMeans
from PIL import Image
import pathlib
import logging


def load_archetypes(archetypes_dir, model, device, max_height=2048):
    """Load and encode archetype wireframes.
    
    Args:
        archetypes_dir: Path to archetypes_png folder
        model: Trained VAE model
        device: torch.device
        max_height: Maximum image height
    
    Returns:
        latents: (N, latent_dim) encoded representations
        labels: (N,) archetype labels (0, 1, ..., K-1)
        label_names: List of archetype names
    """
    import torchvision.transforms as T
    
    archetypes_path = pathlib.Path(archetypes_dir)
    if not archetypes_path.exists():
        logging.warning(f"Archetypes directory not found: {archetypes_dir}")
        return None, None, None
    
    # Find all archetype images
    archetype_files = sorted(list(archetypes_path.glob("*_linear.png")))
    if len(archetype_files) == 0:
        logging.warning(f"No archetype images found in {archetypes_dir}")
        return None, None, None
    
    logging.info(f"Loading {len(archetype_files)} archetypes from {archetypes_dir}")
    
    # Transform pipeline (same as training)
    transform = T.Compose([
        T.Grayscale(num_output_channels=1),
        T.ToTensor(),
    ])
    
    latents = []
    labels = []
    label_names = []
    
    model.eval()
    with torch.no_grad():
        for idx, img_path in enumerate(archetype_files):
            # Extract archetype name
            archetype_name = img_path.stem.replace("_linear", "")
            label_names.append(archetype_name)
            
            try:
                # Load image
                img = Image.open(img_path).convert('L')
                
                # Resize if too tall
                w, h = img.size
                if h > max_height:
                    new_h = max_height
                    new_w = int(w * (new_h / h))
                    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                
                # Transform and encode
                img_tensor = transform(img).unsqueeze(0).to(device)
                _, mu, _ = model(img_tensor)
                
                latents.append(mu.cpu().numpy())
                labels.append(idx)
                
            except Exception as e:
                logging.warning(f"Failed to load {img_path.name}: {e}")
                continue
    
    if len(latents) == 0:
        return None, None, None
    
    latents = np.concatenate(latents, axis=0)  # (N, latent_dim)
    labels = np.array(labels)  # (N,)
    
    logging.info(f"Loaded {len(latents)} archetypes: {', '.join(label_names)}")
    return latents, labels, label_names


def compute_cluster_metrics(latents, labels):
    """Compute cluster quality metrics.
    
    Args:
        latents: (N, latent_dim) encoded representations
        labels: (N,) ground-truth archetype labels
    
    Returns:
        metrics: dict with silhouette_score, calinski_harabasz, etc.
    """
    if latents is None or len(latents) < 2:
        return {}
    
    metrics = {}
    
    # 1. Silhouette Score (−1 to 1, higher is better)
    # Measures how similar an object is to its own cluster vs other clusters
    if len(np.unique(labels)) > 1 and len(latents) > len(np.unique(labels)):
        try:
            sil_score = silhouette_score(latents, labels, metric='euclidean')
            metrics['silhouette_score'] = float(sil_score)
        except Exception as e:
            logging.warning(f"Silhouette score failed: {e}")
    
    # 2. Calinski-Harabasz Index (higher is better)
    # Ratio of between-cluster to within-cluster variance
    if len(np.unique(labels)) > 1:
        try:
            ch_score = calinski_harabasz_score(latents, labels)
            metrics['calinski_harabasz_score'] = float(ch_score)
        except Exception as e:
            logging.warning(f"Calinski-Harabasz failed: {e}")
    
    # 3. Average pairwise cluster distance
    # Measures separation between different archetypes
    try:
        unique_labels = np.unique(labels)
        centroids = []
        for label in unique_labels:
            cluster_latents = latents[labels == label]
            centroid = np.mean(cluster_latents, axis=0)
            centroids.append(centroid)
        centroids = np.array(centroids)
        
        # Pairwise distances between centroids
        from scipy.spatial.distance import pdist
        pairwise_dists = pdist(centroids, metric='euclidean')
        metrics['mean_cluster_distance'] = float(np.mean(pairwise_dists))
        metrics['min_cluster_distance'] = float(np.min(pairwise_dists))
        
    except Exception as e:
        logging.warning(f"Cluster distance failed: {e}")
    
    # 4. Intra-cluster variance (lower is better for tight clusters)
    try:
        total_variance = 0.0
        for label in np.unique(labels):
            cluster_latents = latents[labels == label]
            if len(cluster_latents) > 1:
                variance = np.var(cluster_latents, axis=0).mean()
                total_variance += variance
        metrics['mean_intra_cluster_variance'] = float(total_variance / len(np.unique(labels)))
    except Exception as e:
        logging.warning(f"Intra-cluster variance failed: {e}")
    
    return metrics


def compute_interpolation_quality(model, latent1, latent2, device, num_steps=10):
    """Compute smoothness of interpolation between two latents.
    
    Args:
        model: VAE decoder
        latent1: (latent_dim,) first latent code
        latent2: (latent_dim,) second latent code
        device: torch.device
        num_steps: Number of interpolation steps
    
    Returns:
        smoothness_score: Average pixel difference between consecutive frames
    """
    model.eval()
    
    # Linear interpolation
    alphas = np.linspace(0, 1, num_steps)
    reconstructions = []
    
    with torch.no_grad():
        for alpha in alphas:
            z_interp = (1 - alpha) * latent1 + alpha * latent2
            z_interp_torch = torch.from_numpy(z_interp).unsqueeze(0).float().to(device)
            
            # Decode
            recon = model.decode(z_interp_torch) if hasattr(model, 'decode') else model.decoder(z_interp_torch)
            reconstructions.append(recon.cpu())
    
    # Compute frame-to-frame differences
    diffs = []
    for i in range(len(reconstructions) - 1):
        diff = F.l1_loss(reconstructions[i], reconstructions[i+1], reduction='mean')
        diffs.append(diff.item())
    
    # Smoothness = low variance in differences (consistent change)
    smoothness = float(np.mean(diffs))
    variance = float(np.var(diffs))
    
    return {
        'interpolation_smoothness': smoothness,
        'interpolation_variance': variance
    }


def compute_latent_density_metrics(latents, n_neighbors=5):
    """Compute latent space density and coverage.
    
    Args:
        latents: (N, latent_dim) encoded representations
        n_neighbors: Number of neighbors for density estimation
    
    Returns:
        metrics: dict with density statistics
    """
    if latents is None or len(latents) < n_neighbors:
        return {}
    
    metrics = {}
    
    try:
        from sklearn.neighbors import NearestNeighbors
        
        # 1. Average distance to k-nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, metric='euclidean').fit(latents)
        distances, _ = nbrs.kneighbors(latents)
        
        # Exclude self (distance 0)
        distances = distances[:, 1:]
        
        metrics['mean_knn_distance'] = float(np.mean(distances))
        metrics['std_knn_distance'] = float(np.std(distances))
        
        # 2. Coverage: percentage of latent space volume used
        # Approximate by ratio of convex hull volume to hypercube volume
        from scipy.spatial import ConvexHull
        if latents.shape[1] <= 10:  # Only for low-dimensional spaces
            try:
                hull = ConvexHull(latents)
                metrics['convex_hull_volume'] = float(hull.volume)
            except Exception:
                pass
        
        # 3. Effective dimensionality (PCA explained variance)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(10, latents.shape[1]))
        pca.fit(latents)
        
        # Number of components explaining 95% variance
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        n_components_95 = int(np.argmax(cumsum >= 0.95) + 1)
        metrics['effective_dimensionality_95'] = n_components_95
        
    except Exception as e:
        logging.warning(f"Density metrics failed: {e}")
    
    return metrics


def log_latent_space_visualization(model, archetypes_dir, device, writer, epoch, max_height=2048):
    """Generate and log comprehensive latent space visualizations to TensorBoard.
    
    Args:
        model: Trained VAE
        archetypes_dir: Path to archetypes
        device: torch.device
        writer: TensorBoard SummaryWriter
        epoch: Current epoch number
        max_height: Max image height
    """
    import matplotlib.pyplot as plt
    
    # Load archetypes
    latents, labels, label_names = load_archetypes(archetypes_dir, model, device, max_height)
    
    if latents is None:
        logging.warning("Skipping latent visualization: no archetypes loaded")
        return
    
    # 1. Cluster Metrics
    cluster_metrics = compute_cluster_metrics(latents, labels)
    for key, value in cluster_metrics.items():
        writer.add_scalar(f"latent/{key}", value, epoch)
    
    logging.info(f"Epoch {epoch} Cluster Metrics: {cluster_metrics}")
    
    # 2. Interpolation Quality (between first and last archetype)
    if len(latents) >= 2:
        try:
            interp_metrics = compute_interpolation_quality(
                model, latents[0], latents[-1], device, num_steps=10
            )
            for key, value in interp_metrics.items():
                writer.add_scalar(f"latent/{key}", value, epoch)
            logging.info(f"Epoch {epoch} Interpolation Metrics: {interp_metrics}")
        except Exception as e:
            logging.warning(f"Interpolation quality failed: {e}")
    
    # 3. Latent Density
    density_metrics = compute_latent_density_metrics(latents, n_neighbors=5)
    for key, value in density_metrics.items():
        writer.add_scalar(f"latent/{key}", value, epoch)
    
    logging.info(f"Epoch {epoch} Density Metrics: {density_metrics}")
    
    # 4. t-SNE Visualization with labels
    try:
        from sklearn.manifold import TSNE
        import matplotlib.cm as cm
        
        tsne = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto")
        z_embedded = tsne.fit_transform(latents)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Color by archetype
        colors = cm.rainbow(np.linspace(0, 1, len(label_names)))
        for idx, (label_name, color) in enumerate(zip(label_names, colors)):
            mask = labels == idx
            ax.scatter(z_embedded[mask, 0], z_embedded[mask, 1], 
                      c=[color], label=label_name, s=200, alpha=0.8, edgecolors='black')
        
        ax.set_title(f"Latent Space t-SNE (Epoch {epoch})", fontsize=16)
        ax.set_xlabel("t-SNE Component 1")
        ax.set_ylabel("t-SNE Component 2")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
        ax.grid(True, alpha=0.3)
        
        writer.add_figure("latent/tsne_visualization", fig, epoch)
        plt.close(fig)
        
    except Exception as e:
        logging.warning(f"t-SNE visualization failed: {e}")
