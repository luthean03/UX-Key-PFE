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
    with torch.no_grad():
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


def slerp(z1: np.ndarray, z2: np.ndarray, alpha: float) -> np.ndarray:
    """Spherical Linear Interpolation (SLERP) between two latent codes.
    
    SLERP preserves the norm of latent vectors, providing smoother
    interpolations than linear interpolation.
    
    Args:
        z1: (latent_dim,) first latent code
        z2: (latent_dim,) second latent code  
        alpha: interpolation factor in [0, 1]
        
    Returns:
        z_interp: interpolated latent code
    """
    # Normalize to unit vectors
    z1_norm = z1 / (np.linalg.norm(z1) + 1e-8)
    z2_norm = z2 / (np.linalg.norm(z2) + 1e-8)
    
    # Compute angle between vectors
    dot = np.clip(np.dot(z1_norm, z2_norm), -1.0 + 1e-6, 1.0 - 1e-6)
    omega = np.arccos(dot)
    sin_omega = np.sin(omega)
    
    # Handle nearly parallel vectors (use lerp instead)
    if np.abs(sin_omega) < 1e-6:
        return (1 - alpha) * z1 + alpha * z2
    
    # Scale back to original magnitudes (average of both)
    scale1 = np.linalg.norm(z1)
    scale2 = np.linalg.norm(z2)
    scale = (1 - alpha) * scale1 + alpha * scale2
    
    z_interp = (np.sin((1 - alpha) * omega) / sin_omega) * z1_norm + \
               (np.sin(alpha * omega) / sin_omega) * z2_norm
    
    return z_interp * scale


def generate_interpolation_video(model, latent1, latent2, archetype_img1, archetype_img2, device, 
                                  num_steps=10, include_endpoints=True, target_height=512,
                                  method='slerp'):
    """Generate interpolation frames between two latents (for TensorBoard video).
    
    Interpolates both in latent space (using SLERP) AND geometrically (varying height/width)
    to handle archetypes with different dimensions.
    
    Args:
        model: VAE model
        latent1: (latent_dim,) first latent code (archetype 1)
        latent2: (latent_dim,) second latent code (archetype 2)
        archetype_img1: Tensor (1, C, H, W) original archetype 1 image (optional)
        archetype_img2: Tensor (1, C, H, W) original archetype 2 image (optional)
        device: torch.device
        num_steps: Number of interpolation steps (excluding endpoints if include_endpoints=True)
        include_endpoints: If True, add archetype reconstructions at start/end
        target_height: Resize all frames to this height for consistent visualization
        method: 'slerp' (spherical, smoother) or 'lerp' (linear)
    
    Returns:
        torch.Tensor: (num_frames, C, H, W) interpolated frames or None if failed
    """
    model.eval()
    reconstructions = []
    
    # Get original dimensions for geometric interpolation
    # archetype_imgX can be either None or a tuple returned by load_archetypes: (padded_img, mask, orig_h, orig_w)
    if archetype_img1 is not None and isinstance(archetype_img1, tuple):
        orig_h1, orig_w1 = int(archetype_img1[2]), int(archetype_img1[3])
    elif archetype_img1 is not None:
        orig_h1, orig_w1 = archetype_img1.shape[2:]
    else:
        orig_h1, orig_w1 = target_height, target_height

    if archetype_img2 is not None and isinstance(archetype_img2, tuple):
        orig_h2, orig_w2 = int(archetype_img2[2]), int(archetype_img2[3])
    elif archetype_img2 is not None:
        orig_h2, orig_w2 = archetype_img2.shape[2:]
    else:
        orig_h2, orig_w2 = target_height, target_height
    
    with torch.no_grad():
        # 1. Reconstruct archetype 1 (beginning) - use forward pass like validation
        if include_endpoints:
            try:
                if archetype_img1 is not None:
                    # Forward complet (encode + decode) comme pendant validation
                    if isinstance(archetype_img1, tuple):
                        img_t, mask_t, _, _ = archetype_img1
                        recon1, _, _ = model(img_t.to(device), mask=mask_t.to(device))
                        # Crop reconstruction to original size (remove padding)
                        recon1 = recon1[:, :, :orig_h1, :orig_w1]
                        # Crop reconstruction to original size (remove padding)
                        recon1 = recon1[:, :, :orig_h1, :orig_w1]
                    else:
                        recon1, _, _ = model(archetype_img1.to(device))
                else:
                    # Fallback: decode depuis latent with original size
                    # Fallback: decode depuis latent with original size
                    z1_torch = torch.from_numpy(latent1).unsqueeze(0).float().to(device)
                    try:
                        recon1 = model.decode(z1_torch, output_size=(orig_h1, orig_w1))
                    except TypeError:
                        recon1 = model.decode(z1_torch)
                        recon1 = F.interpolate(recon1, size=(orig_h1, orig_w1), mode='bilinear', align_corners=False)
                    try:
                        recon1 = model.decode(z1_torch, output_size=(orig_h1, orig_w1))
                    except TypeError:
                        recon1 = model.decode(z1_torch)
                        recon1 = F.interpolate(recon1, size=(orig_h1, orig_w1), mode='bilinear', align_corners=False)
                reconstructions.append(recon1)
            except Exception as e:
                logging.warning(f"Endpoint 1 decode failed: {e}")
                return None
        
        # 2. Interpolation in latent space (SLERP or LERP) + geometric interpolation
        # If endpoints should be included separately, exclude alpha=0 and alpha=1
        if include_endpoints:
            alphas = np.linspace(0, 1, num_steps + 2)[1:-1]
        else:
            alphas = np.linspace(0, 1, num_steps)

        for alpha in alphas:
            # Use SLERP for smoother interpolation
            if method == 'slerp':
                z_interp = slerp(latent1, latent2, alpha)
            else:  # lerp
                z_interp = (1 - alpha) * latent1 + alpha * latent2

            z_interp_torch = torch.from_numpy(z_interp).unsqueeze(0).float().to(device)

            # Compute desired output geometry BEFORE decoding (avoid deforming after decode)
            h_interp = int((1 - alpha) * orig_h1 + alpha * orig_h2)
            w_interp = int((1 - alpha) * orig_w1 + alpha * orig_w2)

            try:
                # Prefer decoder API that accepts an output_size argument
                try:
                    recon = model.decode(z_interp_torch, output_size=(h_interp, w_interp))
                except TypeError:
                    # Fallback: decode then resize (for decoders without output_size param)
                    recon = model.decode(z_interp_torch)
                    recon = F.interpolate(recon, size=(h_interp, w_interp), 
                                           mode='bilinear', align_corners=False)

                reconstructions.append(recon)
            except Exception as e:
                logging.warning(f"Interpolation decode failed ({e})")
                return None
        
        # 3. Reconstruct archetype 2 (end) - use forward pass like validation
        if include_endpoints:
            try:
                if archetype_img2 is not None:
                    # Forward complet (encode + decode) comme pendant validation
                    if isinstance(archetype_img2, tuple):
                        img_t, mask_t, _, _ = archetype_img2
                        recon2, _, _ = model(img_t.to(device), mask=mask_t.to(device))
                        # Crop reconstruction to original size (remove padding)
                        recon2 = recon2[:, :, :orig_h2, :orig_w2]
                        # Crop reconstruction to original size (remove padding)
                        recon2 = recon2[:, :, :orig_h2, :orig_w2]
                    else:
                        recon2, _, _ = model(archetype_img2.to(device))
                else:
                    # Fallback: decode depuis latent with original size
                    # Fallback: decode depuis latent with original size
                    z2_torch = torch.from_numpy(latent2).unsqueeze(0).float().to(device)
                    try:
                        recon2 = model.decode(z2_torch, output_size=(orig_h2, orig_w2))
                    except TypeError:
                        recon2 = model.decode(z2_torch)
                        recon2 = F.interpolate(recon2, size=(orig_h2, orig_w2), mode='bilinear', align_corners=False)
                    try:
                        recon2 = model.decode(z2_torch, output_size=(orig_h2, orig_w2))
                    except TypeError:
                        recon2 = model.decode(z2_torch)
                        recon2 = F.interpolate(recon2, size=(orig_h2, orig_w2), mode='bilinear', align_corners=False)
                reconstructions.append(recon2)
            except Exception as e:
                logging.warning(f"Endpoint 2 decode failed: {e}")
                return None
    
    if len(reconstructions) == 0:
        return None

    # Return frames as a list of native-size tensors (C, H, W) without extra padding
    frames = []
    # Return frames as a list of native-size tensors (C, H, W) without extra padding
    frames = []
    for t in reconstructions:
        # Ensure shape (1, C, H, W)
        # Ensure shape (1, C, H, W)
        if t.dim() == 3:
            t = t.unsqueeze(0)
        # squeeze batch dim -> (C, H, W)
        frames.append(t[0].cpu())

        # squeeze batch dim -> (C, H, W)
        frames.append(t[0].cpu())

    return frames


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
    
    with torch.no_grad():
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
    
    # 6. Interpolation Image Grid (entre 2 archetypes aléatoires)
    if archetype_latents is not None and len(archetype_latents) >= 2:
        try:
            import torchvision
            # Choisir 2 archetypes au hasard
            idx1, idx2 = np.random.choice(len(archetype_latents), 2, replace=False)
            logging.info(f"Generating interpolation sequence between {archetype_names[idx1]} and {archetype_names[idx2]}")
            
            # Générer les frames (avec endpoints = archetypes reconstruits via forward complet)
            img1 = archetype_images[idx1] if archetype_images is not None else None
            img2 = archetype_images[idx2] if archetype_images is not None else None
            interp_frames = generate_interpolation_video(
                model, archetype_latents[idx1], archetype_latents[idx2], 
                img1, img2, device, 
                num_steps=8, include_endpoints=True, target_height=512
            )
            
            if interp_frames is not None:
                # interp_frames is a list of tensors (C,H,W) with variable sizes.
                tag = f"interpolations/{archetype_names[idx1]}_to_{archetype_names[idx2]}"

                # 0) Log preprocessed original archetype images (cropped to original size) as step 0 and final step
                def _crop_original(ar_img_tuple):
                    # ar_img_tuple: (padded_cpu, mask_cpu, orig_h, orig_w)
                    padded_cpu, mask_cpu, oh, ow = ar_img_tuple
                    # padded_cpu shape: (1, C, H, W)
                    img = padded_cpu[0, :, :int(oh), :int(ow)].clone()
                    return img

                # If archetype images are tuples (padded, mask, orig_h, orig_w) we crop, else use as-is
                if isinstance(img1, tuple):
                    start_img = _crop_original(img1)
                elif img1 is not None:
                    start_img = img1[0].cpu() if img1.dim() == 4 else img1.cpu()
                else:
                    start_img = None

                if isinstance(img2, tuple):
                    end_img = _crop_original(img2)
                elif img2 is not None:
                    end_img = img2[0].cpu() if img2.dim() == 4 else img2.cpu()
                else:
                    end_img = None

                step_base = epoch * 1000
                current_step = step_base
                
                # 1) Create grid: [preprocessed_1, reconstructed_1] at start
                if start_img is not None and len(interp_frames) > 0:
                    # Resize to same height for grid (use max height)
                    recon1 = interp_frames[0]  # First frame is recon1
                    max_h = max(start_img.shape[1], recon1.shape[1])
                    
                    # Resize both to same height keeping aspect ratio
                    start_resized = F.interpolate(start_img.unsqueeze(0), size=(max_h, start_img.shape[2]), mode='bilinear', align_corners=False)[0]
                    recon1_resized = F.interpolate(recon1.unsqueeze(0), size=(max_h, recon1.shape[2]), mode='bilinear', align_corners=False)[0]
                    
                    # Normalize each
                    start_norm = (start_resized - start_resized.min()) / (start_resized.max() - start_resized.min() + 1e-8)
                    recon1_norm = (recon1_resized - recon1_resized.min()) / (recon1_resized.max() - recon1_resized.min() + 1e-8)
                    
                    # Create grid side by side
                    grid = torch.cat([start_norm, recon1_norm], dim=2)  # Concatenate horizontally
                    writer.add_image(f"{tag}_grid_start", grid, global_step=current_step)
                    current_step += 1

                # 2) Log each interpolated frame at increasing steps
                for frame_idx, frame in enumerate(interp_frames):
                    # frame is (C,H,W)
                    fnorm = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)
                    writer.add_image(tag, fnorm, global_step=current_step)
                    current_step += 1

                # 3) Create grid: [reconstructed_2, preprocessed_2] at end
                if end_img is not None and len(interp_frames) > 0:
                    # Resize to same height for grid
                    recon2 = interp_frames[-1]  # Last frame is recon2
                    max_h = max(recon2.shape[1], end_img.shape[1])
                    
                    # Resize both to same height
                    recon2_resized = F.interpolate(recon2.unsqueeze(0), size=(max_h, recon2.shape[2]), mode='bilinear', align_corners=False)[0]
                    end_resized = F.interpolate(end_img.unsqueeze(0), size=(max_h, end_img.shape[2]), mode='bilinear', align_corners=False)[0]
                    
                    # Normalize each
                    recon2_norm = (recon2_resized - recon2_resized.min()) / (recon2_resized.max() - recon2_resized.min() + 1e-8)
                    end_norm = (end_resized - end_resized.min()) / (end_resized.max() - end_resized.min() + 1e-8)
                    
                    # Create grid side by side
                    grid = torch.cat([recon2_norm, end_norm], dim=2)  # Concatenate horizontally
                    writer.add_image(f"{tag}_grid_end", grid, global_step=current_step)

                logging.info(f"✅ Interpolation ({len(interp_frames)} frames + 2 grids) saved: {archetype_names[idx1]} → {archetype_names[idx2]}")
        except Exception as e:
            logging.warning(f"Interpolation failed: {e}")
    
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

