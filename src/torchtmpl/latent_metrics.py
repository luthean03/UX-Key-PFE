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
    
    latents = []
    labels = []
    label_names = []
    images = []  # Stocker aussi les images
    
    model.eval()
    with torch.inference_mode():
        for idx, img_path in enumerate(archetype_files):
            # Extract archetype name
            archetype_name = img_path.stem.replace("_linear", "")
            label_names.append(archetype_name)
            
            try:
                # Load image
                img = Image.open(img_path).convert('L')
                
                # === PREPROCESSING IDENTIQUE AU TRAINING ===
                # CROP si trop grand (comme dans data.py ligne 71-78)
                w, h = img.size
                if h > max_height:
                    # Crop déterministe au centre (comme validation)
                    top = (h - max_height) // 2
                    img = img.crop((0, top, w, top + max_height))
                # ===========================================
                
                # Transform to tensor
                import torchvision.transforms.functional as TF
                img_tensor = TF.to_tensor(img).unsqueeze(0)

                # Pad to multiple of 32 (same logic as padded_masked_collate)
                stride = 32
                _, _, h, w = img_tensor.unsqueeze(0).shape if img_tensor.dim() == 3 else img_tensor.shape
                pad_h = ((h + stride - 1) // stride) * stride
                pad_w = ((w + stride - 1) // stride) * stride

                padded = torch.zeros(1, img_tensor.shape[1], pad_h, pad_w)
                mask = torch.zeros(1, 1, pad_h, pad_w)
                # copy top-left like collate
                padded[0, :, :h, :w] = img_tensor[0]
                mask[0, 0, :h, :w] = 1.0

                # Encode using the padded tensor and mask (like training/validation)
                _, mu, _ = model(padded.to(device), mask=mask.to(device))

                latents.append(mu.cpu().numpy())
                labels.append(idx)
                # Save tuple: (padded_image_cpu, mask_cpu, orig_h, orig_w)
                images.append((padded.cpu(), mask.cpu(), h, w))  # Sauvegarder l'image et son masque
                
            except Exception as e:
                logging.warning(f"Failed to load {img_path.name}: {e}")
                continue
    
    if len(latents) == 0:
        return None, None, None, None
    
    latents = np.concatenate(latents, axis=0)  # (N, latent_dim)
    labels = np.array(labels)  # (N,)
    # Ne pas concat les images car elles ont des largeurs différentes
    # Garder comme liste de tensors (1, 1, H, W)
    
    logging.info(f"Loaded {len(latents)} archetypes: {', '.join(label_names)}")
    return latents, labels, label_names, images


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
    n_clusters = len(np.unique(labels))
    if n_clusters > 1 and len(latents) > n_clusters:
        try:
            sil_score = silhouette_score(latents, labels, metric='euclidean')
            metrics['silhouette_score'] = float(sil_score)
        except Exception as e:
            logging.warning(f"Silhouette score failed: {e}")
    
    # 2. Calinski-Harabasz Index (higher is better)
    # Ratio of between-cluster to within-cluster variance
    # Requires: 2 <= n_clusters < n_samples
    if 2 <= n_clusters < len(latents):
        try:
            ch_score = calinski_harabasz_score(latents, labels)
            metrics['calinski_harabasz_score'] = float(ch_score)
        except Exception as e:
            logging.warning(f"Calinski-Harabasz failed: {e}")
    elif n_clusters >= len(latents):
        logging.debug(f"Skipping Calinski-Harabasz: n_clusters ({n_clusters}) >= n_samples ({len(latents)})")
    
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
        valid_clusters = 0
        for label in np.unique(labels):
            cluster_latents = latents[labels == label]
            if len(cluster_latents) > 1:
                variance = np.var(cluster_latents, axis=0).mean()
                total_variance += variance
                valid_clusters += 1
        
        if valid_clusters > 0:
            metrics['mean_intra_cluster_variance'] = float(total_variance / valid_clusters)
        else:
            # Tous les clusters ont 1 seul échantillon (variance=0)
            metrics['mean_intra_cluster_variance'] = 0.0
            logging.debug(f"All clusters have single sample (variance=0)")
    except Exception as e:
        logging.warning(f"Intra-cluster variance failed: {e}")
    
    return metrics


# SLERP est centralisé dans utils.py
from .utils import slerp_numpy as slerp


# ⚠️  INTERPOLATION TENSORBOARD REMOVED
# Use the dedicated interpolate() function in main.py instead


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


def log_latent_space_visualization(model, train_loader, archetypes_dir, device, writer, epoch, max_height=2048, max_samples=1000):
    """Generate and log comprehensive latent space visualizations to TensorBoard.
    
    Args:
        model: Trained VAE
        train_loader: DataLoader du dataset d'entraînement
        archetypes_dir: Path to archetypes (pour k-means avec k connu)
        device: torch.device
        writer: TensorBoard SummaryWriter
        epoch: Current epoch number
        max_height: Max image height
        max_samples: Nombre max de samples du train set à encoder
    """
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    
    # 1. Encoder le dataset d'entraînement complet
    logging.info(f"Encoding training dataset (max {max_samples} samples)...")
    model.eval()
    train_latents = []
    train_indices = []
    
    with torch.inference_mode():
        for i, (inputs, _, masks) in enumerate(train_loader):
            if i >= max_samples:
                break
            inputs = inputs.to(device)
            masks = masks.to(device)
            _, mu, _ = model(inputs, mask=masks)
            train_latents.append(mu.cpu().numpy())
            train_indices.append(i)
    
    if len(train_latents) == 0:
        logging.warning("No training samples encoded, skipping visualization")
        return
    
    train_latents = np.concatenate(train_latents, axis=0)  # (N, latent_dim)
    logging.info(f"Encoded {len(train_latents)} training samples")
    
    # 2. Charger archetypes pour déterminer k
    archetype_latents, archetype_labels, archetype_names, archetype_images = load_archetypes(archetypes_dir, model, device, max_height)
    
    if archetype_latents is None:
        logging.warning("No archetypes loaded, using k=15 by default")
        k = 15
        archetype_names = [f"Cluster_{i}" for i in range(k)]
    else:
        k = len(archetype_names)
        logging.info(f"Loaded {k} archetypes: {', '.join(archetype_names)}")
    
    # 3. Appliquer k-means sur le dataset d'entraînement
    logging.info(f"Applying k-means with k={k} on training latents...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    train_cluster_labels = kmeans.fit_predict(train_latents)
    
    # 4. Cluster Metrics sur dataset d'entraînement
    cluster_metrics = compute_cluster_metrics(train_latents, train_cluster_labels)
    for key, value in cluster_metrics.items():
        writer.add_scalar(f"latent/{key}", value, epoch)
    
    logging.info(f"Epoch {epoch} Cluster Metrics (k-means on train): {cluster_metrics}")
    
    # 5. Si archetypes disponibles, vérifier correspondance cluster ↔ archetype
    if archetype_latents is not None:
        logging.info("Checking cluster-archetype correspondence...")
        # Assigner chaque archetype au cluster le plus proche (centroïd)
        archetype_cluster_assignments = kmeans.predict(archetype_latents)
        
        # Compter combien d'archetypes par cluster
        cluster_to_archetypes = {}
        for arch_idx, cluster_id in enumerate(archetype_cluster_assignments):
            if cluster_id not in cluster_to_archetypes:
                cluster_to_archetypes[cluster_id] = []
            cluster_to_archetypes[cluster_id].append(archetype_names[arch_idx])
        
        # Logger la correspondance
        for cluster_id in range(k):
            archs = cluster_to_archetypes.get(cluster_id, [])
            if len(archs) == 0:
                logging.info(f"  Cluster {cluster_id}: ❌ No archetype assigned")
            elif len(archs) == 1:
                logging.info(f"  Cluster {cluster_id}: ✅ {archs[0]}")
            else:
                logging.info(f"  Cluster {cluster_id}: ⚠️  Multiple archetypes: {', '.join(archs)}")
    
    # ⚠️  INTERPOLATION TENSORBOARD REMOVED
    # Use the dedicated interpolate() function in main.py instead
    
    # 7. Latent Density
    density_metrics = compute_latent_density_metrics(train_latents, n_neighbors=5)
    for key, value in density_metrics.items():
        writer.add_scalar(f"latent/{key}", value, epoch)
    
    logging.info(f"Epoch {epoch} Density Metrics: {density_metrics}")
    
    # 8. t-SNE/PCA Visualization du dataset d'entraînement avec clusters k-means
    try:
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        import matplotlib.cm as cm
        
        n_samples = len(train_latents)
        perplexity = min(30.0, max(5.0, n_samples / 3))  # Heuristique: perplexity ≈ n_samples/3
        
        # === GÉNÉRER PCA ET t-SNE ===
        logging.info(f"Generating PCA visualization...")
        pca = PCA(n_components=2, random_state=42)
        z_pca = pca.fit_transform(train_latents)
        
        # Calculer t-SNE seulement si assez de samples
        if n_samples >= 50:
            logging.info(f"Generating t-SNE visualization with perplexity={perplexity:.1f} (n_samples={n_samples})")
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, 
                       init="pca", learning_rate="auto")
            z_tsne = tsne.fit_transform(train_latents)
            visualizations = [("PCA", z_pca, pca), ("t-SNE", z_tsne, None)]
        else:
            logging.info(f"Skipping t-SNE (n_samples={n_samples} < 50)")
            visualizations = [("PCA", z_pca, pca)]
        
        # Générer une figure pour chaque visualisation
        colors = cm.tab20(np.linspace(0, 1, k))
        
        for viz_method, z_embedded, fitted_model in visualizations:
            fig, ax = plt.subplots(figsize=(14, 10))
            
            # Colorier par cluster k-means
            for cluster_id in range(k):
                mask = train_cluster_labels == cluster_id
                if np.sum(mask) > 0:
                    # Nom du cluster (archetype correspondant si disponible)
                    if archetype_latents is not None and cluster_id in cluster_to_archetypes:
                        archs = cluster_to_archetypes[cluster_id]
                        label = f"C{cluster_id}: {archs[0]}" if len(archs) == 1 else f"C{cluster_id}: {len(archs)} archs"
                    else:
                        label = f"Cluster {cluster_id}"
                    
                    ax.scatter(z_embedded[mask, 0], z_embedded[mask, 1], 
                              c=[colors[cluster_id]], label=label, s=30, alpha=0.6, edgecolors='none')
            
            # Superposer les archetypes (étoiles colorées par cluster)
            if archetype_latents is not None:
                if fitted_model is not None:  # PCA
                    arch_embedded = fitted_model.transform(archetype_latents)
                else:  # t-SNE - utiliser KNN
                    from sklearn.neighbors import NearestNeighbors
                    nbrs = NearestNeighbors(n_neighbors=1).fit(train_latents)
                    _, indices = nbrs.kneighbors(archetype_latents)
                    arch_embedded = z_embedded[indices.flatten()]
                
                # Colorier chaque archetype selon son cluster k-means
                for i, (name, cluster_id) in enumerate(zip(archetype_names, archetype_cluster_assignments)):
                    ax.scatter(arch_embedded[i, 0], arch_embedded[i, 1], 
                              c=[colors[cluster_id]], marker='*', s=500, 
                              edgecolors='white', linewidths=2, zorder=10)
                    
                    # Annoter l'archetype
                    ax.annotate(name, (arch_embedded[i, 0], arch_embedded[i, 1]),
                               fontsize=8, ha='center', va='bottom', 
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
            
            ax.set_title(f"Latent Space {viz_method} - Training Dataset (Epoch {epoch}, n={n_samples}, k={k})", fontsize=14)
            ax.set_xlabel(f"{viz_method} Component 1")
            ax.set_ylabel(f"{viz_method} Component 2")
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9, ncol=2)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            # Sauvegarder dans TensorBoard avec un tag différent pour chaque méthode
            tag = "latent/pca_train_kmeans" if viz_method == "PCA" else "latent/tsne_train_kmeans"
            writer.add_figure(tag, fig, epoch)
            plt.close(fig)
            logging.info(f"✅ {viz_method} visualization saved to TensorBoard")
        
    except Exception as e:
        logging.warning(f"Visualization failed: {e}")
        import traceback
        logging.debug(traceback.format_exc())

