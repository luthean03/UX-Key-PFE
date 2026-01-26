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
                
                # Crop if image is taller than max_height (center crop, same as validation)
                w, h = img.size
                if h > max_height:
                    top = (h - max_height) // 2
                    img = img.crop((0, top, w, top + max_height))
                
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
                # Store padded image, mask, and original dimensions
                images.append((padded.cpu(), mask.cpu(), h, w))
                
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


# Import SLERP from centralized location
from .utils import slerp_numpy as slerp


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


def create_interactive_3d_visualization(z_embedded_3d, cluster_labels, archetype_embedded, archetype_names, 
                                       archetype_cluster_labels, train_images, colors, k, 
                                       viz_method, epoch, n_samples, latent_dim, output_path):
    """Create interactive 3D visualization with Plotly and image thumbnails.
    
    Args:
        z_embedded_3d: (N, 3) array of 3D projections (PCA or t-SNE)
        cluster_labels: (N,) cluster assignments
        archetype_embedded: (K, 3) archetype projections
        archetype_names: List of archetype names
        archetype_cluster_labels: (K,) archetype cluster assignments
        train_images: List of torch tensors (images)
        colors: Matplotlib colormap for clusters
        k: Number of clusters
        viz_method: 'PCA' or 't-SNE'
        epoch: Current epoch
        n_samples: Number of samples
        latent_dim: Latent dimension
        output_path: Path to save HTML file
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import base64
        from io import BytesIO
        
        logging.info(f"Creating interactive 3D {viz_method} visualization with Plotly...")
        
        # Helper: convert tensor image to base64
        def tensor_to_base64(img_tensor, max_size=200):
            """Convert torch tensor (1, H, W) to base64 PNG."""
            # Convert to numpy and scale to 0-255
            img_np = (img_tensor.squeeze().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np, mode='L')
            
            # Resize to max_size for thumbnails
            h, w = pil_img.size
            if h > max_size or w > max_size:
                ratio = min(max_size / h, max_size / w)
                new_size = (int(h * ratio), int(w * ratio))
                pil_img = pil_img.resize(new_size, Image.LANCZOS)
            
            # Convert to base64
            buffered = BytesIO()
            pil_img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return f"data:image/png;base64,{img_str}"
        
        # Prepare data traces for each cluster
        fig = go.Figure()
        
        # Add training samples (one trace per cluster for legend)
        for cluster_id in range(k):
            mask = cluster_labels == cluster_id
            if np.sum(mask) == 0:
                continue
            
            # Get cluster points
            cluster_points = z_embedded_3d[mask]
            cluster_indices = np.where(mask)[0]
            
            # Prepare hover text with image thumbnails
            hover_texts = []
            for idx in cluster_indices[:min(len(cluster_indices), len(train_images))]:
                if idx < len(train_images):
                    img_base64 = tensor_to_base64(train_images[idx])
                    hover_text = f"<img src='{img_base64}' width='150'><br>Sample {idx}<br>Cluster {cluster_id}"
                    hover_texts.append(hover_text)
                else:
                    hover_texts.append(f"Sample {idx}<br>Cluster {cluster_id}")
            
            # Convert matplotlib color to RGB
            color_rgb = tuple(int(c * 255) for c in colors[cluster_id][:3])
            
            fig.add_trace(go.Scatter3d(
                x=cluster_points[:, 0],
                y=cluster_points[:, 1],
                z=cluster_points[:, 2],
                mode='markers',
                name=f'Cluster {cluster_id}',
                marker=dict(
                    size=4,
                    color=f'rgb{color_rgb}',
                    opacity=0.6,
                    line=dict(width=0)
                ),
                text=hover_texts,
                hovertemplate='%{text}<extra></extra>',
                customdata=cluster_indices
            ))
        
        # Add archetypes as stars
        if archetype_embedded is not None and len(archetype_embedded) > 0:
            for i, (name, cluster_id) in enumerate(zip(archetype_names, archetype_cluster_labels)):
                color_rgb = tuple(int(c * 255) for c in colors[cluster_id][:3])
                
                fig.add_trace(go.Scatter3d(
                    x=[archetype_embedded[i, 0]],
                    y=[archetype_embedded[i, 1]],
                    z=[archetype_embedded[i, 2]],
                    mode='markers+text',
                    name=f'★ {name}',
                    marker=dict(
                        size=15,
                        color=f'rgb{color_rgb}',
                        symbol='diamond',
                        line=dict(color='white', width=2)
                    ),
                    text=[name],
                    textposition='top center',
                    textfont=dict(size=10, color='black'),
                    hovertemplate=f'<b>{name}</b><br>Cluster {cluster_id}<extra></extra>'
                ))
        
        # Update layout
        fig.update_layout(
            title=f"Interactive 3D {viz_method} - K-means on {latent_dim}D Latent Space<br>Epoch {epoch}, n={n_samples}, k={k}",
            scene=dict(
                xaxis_title=f'{viz_method} Component 1',
                yaxis_title=f'{viz_method} Component 2',
                zaxis_title=f'{viz_method} Component 3',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=1200,
            height=800,
            hovermode='closest',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        # Save as HTML
        fig.write_html(str(output_path))
        logging.info(f"[OK] Interactive 3D {viz_method} saved to {output_path}")
        
        return fig
        
    except ImportError:
        logging.warning("Plotly not installed. Install with: pip install plotly")
        return None
    except Exception as e:
        logging.warning(f"Failed to create interactive 3D visualization: {e}")
        import traceback
        logging.debug(traceback.format_exc())
        return None


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
    import base64
    from io import BytesIO
    
    # 1. Encoder le dataset d'entraînement complet
    logging.info(f"Encoding training dataset (max {max_samples} samples)...")
    model.eval()
    train_latents = []
    train_indices = []
    train_images = []  # Store images for interactive visualization
    
    with torch.inference_mode():
        for i, (inputs, targets, masks) in enumerate(train_loader):
            if i >= max_samples:
                break
            # Use targets (clean images) for latent space visualization
            # We want to visualize the latent space of real data, not augmented/noisy versions
            targets = targets.to(device)
            masks = masks.to(device)
            _, mu, _ = model(targets, mask=masks)
            train_latents.append(mu.cpu().numpy())
            train_indices.append(i)
            # Store first image of batch for visualization (resize for efficiency)
            img_tensor = targets[0].cpu()  # (1, H, W)
            train_images.append(img_tensor)
    
    if len(train_latents) == 0:
        logging.warning("No training samples encoded, skipping visualization")
        return
    
    train_latents = np.concatenate(train_latents, axis=0)  # (N, latent_dim)
    latent_dim = train_latents.shape[1]  # Get actual latent dimension from data
    logging.info(f"Encoded {len(train_latents)} training samples (latent_dim={latent_dim})")
    
    # 2. Charger archetypes pour déterminer k
    archetype_latents, archetype_labels, archetype_names, archetype_images = load_archetypes(archetypes_dir, model, device, max_height)
    
    if archetype_latents is None:
        logging.warning("No archetypes loaded, using k=15 by default")
        k = 15
        archetype_names = [f"Cluster_{i}" for i in range(k)]
    else:
        k = len(archetype_names)
        logging.info(f"Loaded {k} archetypes: {', '.join(archetype_names)}")
    
    # 3. Compute latent density metrics (on full latent space)
    density_metrics = compute_latent_density_metrics(train_latents, n_neighbors=5)
    for key, value in density_metrics.items():
        writer.add_scalar(f"latent/{key}", value, epoch)
    
    logging.info(f"Epoch {epoch} Density Metrics: {density_metrics}")
    
    # 4. K-means clustering on latent space before dimensionality reduction
    logging.info(f"Applying k-means (k={k}) on FULL {latent_dim}D latent space...")
    kmeans_full = KMeans(n_clusters=k, random_state=42, n_init=10)
    train_cluster_labels_full = kmeans_full.fit_predict(train_latents)
    
    # Compute cluster metrics on full space
    cluster_metrics_full = compute_cluster_metrics(train_latents, train_cluster_labels_full)
    for key, value in cluster_metrics_full.items():
        writer.add_scalar(f"latent/full_{latent_dim}d_{key}", value, epoch)
    
    logging.info(f"Epoch {epoch} Full {latent_dim}D Cluster Metrics (k-means on {latent_dim}D): {cluster_metrics_full}")
    
    # 5. Assign archetypes to clusters (on full latent space)
    cluster_to_archetypes_full = {}
    archetype_cluster_assignments_full = None
    
    if archetype_latents is not None:
        logging.info(f"Assigning archetypes to clusters in full {latent_dim}D space...")
        archetype_cluster_assignments_full = kmeans_full.predict(archetype_latents)
        
        # Count archetypes per cluster
        for arch_idx, cluster_id in enumerate(archetype_cluster_assignments_full):
            if cluster_id not in cluster_to_archetypes_full:
                cluster_to_archetypes_full[cluster_id] = []
            cluster_to_archetypes_full[cluster_id].append(archetype_names[arch_idx])
        
        # Log cluster-archetype correspondence
        logging.info(f"Full {latent_dim}D Cluster-Archetype Correspondence:")
        for cluster_id in range(k):
            archs = cluster_to_archetypes_full.get(cluster_id, [])
            if len(archs) == 0:
                logging.info(f"  Cluster {cluster_id}: [X] No archetype assigned")
            elif len(archs) == 1:
                logging.info(f"  Cluster {cluster_id}: [OK] {archs[0]}")
            else:
                logging.info(f"  Cluster {cluster_id}: [!] Multiple archetypes: {', '.join(archs)}")
    
    # 6. t-SNE/PCA Visualization with k-means clusters from full latent space
    try:
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        import matplotlib.cm as cm
        
        n_samples = len(train_latents)
        perplexity = min(30.0, max(5.0, n_samples / 3))  # Adaptive perplexity based on sample size
        
        # Determine output directory for interactive HTML files
        if writer is not None:
            log_dir = pathlib.Path(writer.log_dir)
            interactive_dir = log_dir / "interactive_viz"
            interactive_dir.mkdir(exist_ok=True)
        else:
            interactive_dir = pathlib.Path("./interactive_viz")
            interactive_dir.mkdir(exist_ok=True)
        
        # Generate PCA projection (3 components for 3D viz)
        logging.info(f"Generating PCA projection (3 components for 3D)...")
        pca_3d = PCA(n_components=3, random_state=42)
        z_pca_3d = pca_3d.fit_transform(train_latents)
        logging.info(f"PCA explained variance: {pca_3d.explained_variance_ratio_[0]:.3f}, "
                    f"{pca_3d.explained_variance_ratio_[1]:.3f}, {pca_3d.explained_variance_ratio_[2]:.3f}")
        
        # Also keep 2D for matplotlib
        z_pca_2d = z_pca_3d[:, :2]
        
        # Calculer t-SNE seulement si assez de samples
        if n_samples >= 50:
            # t-SNE 3D
            logging.info(f"Generating t-SNE 3D projection with perplexity={perplexity:.1f} (n_samples={n_samples})")
            tsne_3d = TSNE(n_components=3, random_state=42, perplexity=perplexity, 
                          init="pca", learning_rate="auto")
            z_tsne_3d = tsne_3d.fit_transform(train_latents)
            z_tsne_2d = z_tsne_3d[:, :2]
            
            visualizations_2d = [("PCA", z_pca_2d, pca_3d), ("t-SNE", z_tsne_2d, None)]
            visualizations_3d = [("PCA", z_pca_3d, pca_3d), ("t-SNE", z_tsne_3d, None)]
        else:
            logging.info(f"Skipping t-SNE (n_samples={n_samples} < 50)")
            visualizations_2d = [("PCA", z_pca_2d, pca_3d)]
            visualizations_3d = [("PCA", z_pca_3d, pca_3d)]
        
        # Générer une figure pour chaque visualisation
        colors = cm.tab20(np.linspace(0, 1, k))
        
        # Process both 2D and 3D visualizations
        for idx, ((viz_method_2d, z_embedded_2d, projection_model_2d), 
                  (viz_method_3d, z_embedded_3d, projection_model_3d)) in enumerate(zip(visualizations_2d, visualizations_3d)):
            
            viz_method = viz_method_2d  # Same for both
            
            # Use cluster labels from FULL latent space k-means (not 2D/3D)
            logging.info(f"Visualizing {viz_method} projection with clusters from full {latent_dim}D k-means...")
            
            # Project archetypes to 2D and 3D space for visualization
            arch_embedded_2d = None
            arch_embedded_3d = None
            
            if archetype_latents is not None:
                logging.info(f"Projecting archetypes to {viz_method} space...")
                
                # Projeter les archetypes sur le plan 2D et 3D
                if projection_model_2d is not None:  # PCA
                    arch_embedded_2d = projection_model_2d.transform(archetype_latents)[:, :2]
                    arch_embedded_3d = projection_model_3d.transform(archetype_latents)
                else:  # t-SNE - utiliser KNN pour approximation
                    from sklearn.neighbors import NearestNeighbors
                    nbrs = NearestNeighbors(n_neighbors=1).fit(train_latents)
                    _, indices = nbrs.kneighbors(archetype_latents)
                    arch_embedded_2d = z_embedded_2d[indices.flatten()]
                    arch_embedded_3d = z_embedded_3d[indices.flatten()]
            
            # ===== 1. Create Interactive 3D Plotly Visualization =====
            output_path_3d = interactive_dir / f"epoch_{epoch:04d}_{viz_method.lower()}_3d.html"
            create_interactive_3d_visualization(
                z_embedded_3d, 
                train_cluster_labels_full,
                arch_embedded_3d, 
                archetype_names, 
                archetype_cluster_assignments_full,
                train_images,
                colors, 
                k, 
                viz_method, 
                epoch, 
                n_samples, 
                latent_dim,
                output_path_3d
            )
            
            # ===== 2. Create Static 2D Matplotlib Visualization for TensorBoard =====
            fig, ax = plt.subplots(figsize=(14, 10))
            
            # Colorier par cluster k-means (calculé sur l'espace latent complet)
            for cluster_id in range(k):
                mask = train_cluster_labels_full == cluster_id
                if np.sum(mask) > 0:
                    # Nom du cluster (archetype correspondant si disponible)
                    if archetype_latents is not None and cluster_id in cluster_to_archetypes_full:
                        archs = cluster_to_archetypes_full[cluster_id]
                        label = f"C{cluster_id}: {archs[0]}" if len(archs) == 1 else f"C{cluster_id}: {len(archs)} archs"
                    else:
                        label = f"Cluster {cluster_id}"
                    
                    ax.scatter(z_embedded_2d[mask, 0], z_embedded_2d[mask, 1], 
                              c=[colors[cluster_id]], label=label, s=30, alpha=0.6, edgecolors='none')
            
            # Superposer les archetypes (étoiles colorées par cluster)
            if archetype_latents is not None and arch_embedded_2d is not None:
                # Colorier chaque archetype selon son cluster k-means (sur l'espace latent complet)
                for i, (name, cluster_id) in enumerate(zip(archetype_names, archetype_cluster_assignments_full)):
                    ax.scatter(arch_embedded_2d[i, 0], arch_embedded_2d[i, 1], 
                              c=[colors[cluster_id]], marker='*', s=500, 
                              edgecolors='white', linewidths=2, zorder=10)
                    
                    # Annoter l'archetype
                    ax.annotate(name, (arch_embedded_2d[i, 0], arch_embedded_2d[i, 1]),
                               fontsize=8, ha='center', va='bottom', 
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
            
            ax.set_title(f"Latent Space {viz_method} - K-means on Full {latent_dim}D (Epoch {epoch}, n={n_samples}, k={k})", fontsize=14)
            ax.set_xlabel(f"{viz_method} Component 1")
            ax.set_ylabel(f"{viz_method} Component 2")
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9, ncol=2)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            # Save to TensorBoard with method-specific tag
            tag = "latent/pca_train_kmeans" if viz_method == "PCA" else "latent/tsne_train_kmeans"
            writer.add_figure(tag, fig, epoch)
            plt.close(fig)
            logging.info(f"[OK] {viz_method} visualization saved to TensorBoard")
        
    except Exception as e:
        logging.warning(f"Visualization failed: {e}")
        import traceback
        logging.debug(traceback.format_exc())

