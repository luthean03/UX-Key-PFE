"""Latent space evaluation metrics and visualisation utilities."""

import colorsys
import json
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.cluster import KMeans
from PIL import Image
import pathlib
import logging


def build_ground_truth_label_map(json_dir):
    """Build a dict mapping PNG stem -> urlId from supervised session JSONs.

    Each session JSON contains a ``loms`` dict whose keys are LOM hashes.
    The PNG files are named ``{sessionId}_{i}.png`` where *i* is the 0-based
    position of the LOM in the dict, so we enumerate the dict values.

    Args:
        json_dir: path to a directory of per-session JSON files.

    Returns:
        dict[str, str]: e.g. ``{"abc123_0": "about-us", "abc123_1": "index", ...}``
    """
    label_map = {}
    json_path = pathlib.Path(json_dir)
    json_files = list(json_path.glob("*.json"))
    if not json_files:
        logging.warning(f"No JSON files found in {json_dir} — ground-truth labels unavailable.")
        return label_map

    for jf in json_files:
        session_id = jf.stem
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception as e:
            logging.warning(f"Could not parse {jf.name}: {e}")
            continue

        loms = data.get("loms", {})
        if not isinstance(loms, dict):
            continue
        for i, lom in enumerate(loms.values()):
            url_id = lom.get("urlId")
            if url_id is not None:
                stem = f"{session_id}_{i}"
                label_map[stem] = url_id

    logging.info(f"Built ground-truth label map: {len(label_map)} entries from {len(json_files)} sessions")
    return label_map


def compute_clustering_metrics(true_labels, pred_labels, method_name="K-Means"):
    """Compute external clustering metrics against ground-truth labels.

    Requires ``true_labels`` and ``pred_labels`` to have the same length.
    Both can be arrays of arbitrary (but consistent) label types.

    Returns:
        dict with keys: ari, ami, nmi, homogeneity, completeness, v_measure.
    """
    from sklearn.metrics import (
        adjusted_rand_score,
        adjusted_mutual_info_score,
        normalized_mutual_info_score,
        homogeneity_completeness_v_measure,
    )

    true = np.asarray(true_labels)
    pred = np.asarray(pred_labels)

    ari   = adjusted_rand_score(true, pred)
    ami   = adjusted_mutual_info_score(true, pred)
    nmi   = normalized_mutual_info_score(true, pred)
    h, c, v = homogeneity_completeness_v_measure(true, pred)

    metrics = dict(ari=ari, ami=ami, nmi=nmi, homogeneity=h, completeness=c, v_measure=v)

    logging.info(
        f"[{method_name}] Clustering metrics vs ground truth — "
        f"ARI={ari:.4f}  AMI={ami:.4f}  NMI={nmi:.4f}  "
        f"Hom={h:.4f}  Com={c:.4f}  V={v:.4f}"
    )
    return metrics


def _generate_distinct_colors(n):
    """Generate n maximally distinct RGB colors using golden-ratio hue spacing in HSV."""
    golden_ratio = 0.618033988749895
    hue = 0.0
    colors = []
    for _ in range(n):
        hue = (hue + golden_ratio) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.82, 0.92)
        colors.append([int(r * 255), int(g * 255), int(b * 255)])
    return colors


def load_archetypes(archetypes_dir, model, device, max_height=2048):
    """Load and encode archetype wireframes.

    Returns:
        (latents, labels, label_names, images) or (None, None, None, None).
    """
    import torchvision.transforms as T
    
    archetypes_path = pathlib.Path(archetypes_dir)
    if not archetypes_path.exists():
        logging.warning(f"Archetypes directory not found: {archetypes_dir}")
        return None, None, None, None
    
    # Find all archetype images (PNG files without subdirectories)
    archetype_files = sorted(list(archetypes_path.glob("*.png")))
    if len(archetype_files) == 0:
        logging.warning(f"No archetype images found in {archetypes_dir}")
        return None, None, None, None
    
    logging.info(f"Loading {len(archetype_files)} archetypes from {archetypes_dir}")
    
    latents = []
    labels = []
    label_names = []
    images = []
    
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

                # Adaptive Average Pooling on valid region only (crop before pool)
                if mu.dim() == 4:
                    mask_latent = F.interpolate(mask.to(device), size=mu.shape[2:], mode='nearest')
                    valid_h = int(mask_latent[0, 0, :, 0].sum().item())
                    valid_w = int(mask_latent[0, 0, 0, :].sum().item())
                    valid_h = max(valid_h, 1)
                    valid_w = max(valid_w, 1)
                    mu_valid = mu[:, :, :valid_h, :valid_w]
                    mu_pooled = F.adaptive_avg_pool2d(mu_valid, output_size=(8, 4)).flatten(start_dim=1)
                    latents.append(mu_pooled.cpu().numpy())
                else:
                    latents.append(mu.cpu().numpy())
                labels.append(idx)
                # Store padded image, mask, and original dimensions
                images.append((padded.cpu(), mask.cpu(), h, w))
                
            except Exception as e:
                logging.warning(f"Failed to load {img_path.name}: {e}")
                continue
    
    if len(latents) == 0:
        return None, None, None, None
    
    latents = np.concatenate(latents, axis=0)
    labels = np.array(labels)

    logging.info(f"Loaded {len(latents)} archetypes: {', '.join(label_names)}")
    return latents, labels, label_names, images


def create_interactive_3d_visualization(z_embedded_3d, clustering_results, archetype_embedded, archetype_names,
    train_images, viz_method, epoch, n_samples, latent_dim, output_path,
    train_image_names=None, archetype_images=None, gt_labels=None, metrics=None):
    """Create an interactive 3D Plotly scatter with multiple clustering methods and a dropdown switcher.

    Args:
        z_embedded_3d: (N, 3) ndarray of embedded coordinates.
        clustering_results: dict mapping method name to
            {"labels": ndarray(N,), "archetype_labels": ndarray(M,) or None, "n_clusters": int}.
        archetype_embedded: (M, 3) ndarray or None.
        archetype_names: list[str] of archetype names.
        train_images: list of image tensors.
        viz_method: str, e.g. "PCA" or "t-SNE".
        epoch: int.
        n_samples: int.
        latent_dim: int.
        output_path: pathlib.Path.
        train_image_names: optional list[str].
        archetype_images: optional list of (padded, mask, h, w) tuples.
        gt_labels: optional list[str|None] — ground-truth urlId per training point.
        metrics: optional dict method_name -> {ari, ami, nmi, homogeneity, completeness, v_measure}.
    """
    try:
        import json
        import base64
        from io import BytesIO
        from collections import Counter

        logging.info(f"Creating interactive 3D {viz_method} visualization with multi-clustering support...")

        def tensor_to_base64(img_tensor, max_size=200):
            img_np = (img_tensor.squeeze().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np, mode='L')
            w, h = pil_img.size
            if w > max_size or h > max_size:
                ratio = min(max_size / w, max_size / h)
                pil_img = pil_img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
            buf = BytesIO()
            pil_img.save(buf, format='PNG')
            return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()

        # Prepare point coordinates
        points_x = z_embedded_3d[:, 0].tolist()
        points_y = z_embedded_3d[:, 1].tolist()
        points_z = z_embedded_3d[:, 2].tolist()

        # Image names
        if train_image_names and len(train_image_names) == len(train_images):
            image_names = list(train_image_names)
        else:
            image_names = [f"Sample {i}" for i in range(len(train_images))]

        # Base64 encode images
        images_base64 = [tensor_to_base64(img) for img in train_images]
        n_train = len(train_images)

        # Archetype data
        arch_js = None
        if archetype_embedded is not None and len(archetype_embedded) > 0:
            arch_js = {
                "x": archetype_embedded[:, 0].tolist(),
                "y": archetype_embedded[:, 1].tolist(),
                "z": archetype_embedded[:, 2].tolist(),
                "names": list(archetype_names) if archetype_names else [],
            }
            # Add archetype images
            if archetype_images is not None and len(archetype_images) > 0:
                for i, (padded, mask_t, h, w) in enumerate(archetype_images):
                    arch_img = padded[0, :, :h, :w]
                    images_base64.append(tensor_to_base64(arch_img))
                if archetype_names:
                    image_names.extend([f"\u2605 {name}" for name in archetype_names])

        # Clustering results — build JS-friendly dict with optional GT correctness info
        method_names = list(clustering_results.keys())
        clustering_js = {}
        max_k = 0
        has_gt_in_viz = gt_labels is not None and any(lbl is not None for lbl in gt_labels)
        for method_name, result in clustering_results.items():
            k = int(result["n_clusters"])
            max_k = max(max_k, k)
            cluster_labels_arr = np.asarray(result["labels"])
            entry = {
                "labels": cluster_labels_arr.tolist(),
                "archLabels": result["archetype_labels"].tolist() if result.get("archetype_labels") is not None else None,
                "k": k,
            }
            if has_gt_in_viz:
                # Majority-vote: for each cluster, find dominant GT class
                cluster_dominant = {}
                for c in np.unique(cluster_labels_arr):
                    gt_in_c = [gt_labels[i] for i in range(len(gt_labels))
                               if cluster_labels_arr[i] == c and gt_labels[i] is not None]
                    cluster_dominant[int(c)] = Counter(gt_in_c).most_common(1)[0][0] if gt_in_c else None
                entry["gtLabels"] = list(gt_labels)
                entry["correct"] = [
                    (cluster_dominant.get(int(cluster_labels_arr[i])) == lbl if lbl is not None else None)
                    for i, lbl in enumerate(gt_labels)
                ]
                # Map cluster id -> dominant class name (string keys for JSON)
                entry["clusterDominant"] = {str(c): d for c, d in cluster_dominant.items()}
            clustering_js[method_name] = entry

        # Generate color palette (maximally distinct via golden-ratio HSV spacing)
        colors_rgb = _generate_distinct_colors(max(max_k, 1))

        # Build data JSON
        # urlId -> archetype_name mapping (strip prefix before first '_')
        # e.g. "Web_about-us" -> "about-us", "Mobile_index" -> "index"
        arch_id_map = {}
        if archetype_names:
            for aname in archetype_names:
                stripped = aname.split('_', 1)[-1] if '_' in aname else aname
                arch_id_map[stripped.lower()] = aname

        data_obj = {
            "x": points_x, "y": points_y, "z": points_z,
            "names": image_names,
            "images": images_base64,
            "archetypes": arch_js,
            "nTrain": n_train,
            "clustering": clustering_js,
            "methodNames": method_names,
            "colors": colors_rgb,
            "metrics": metrics,  # None when no ground truth; method→{ari,ami,...} otherwise
            "archIdMap": arch_id_map,  # urlId (lowercase) -> archetype_name
        }
        data_json = json.dumps(data_obj)

        # Layout
        layout_obj = {
            "scene": {
                "xaxis": {"title": f"{viz_method} Component 1"},
                "yaxis": {"title": f"{viz_method} Component 2"},
                "zaxis": {"title": f"{viz_method} Component 3"},
            },
            "width": 1200, "height": 800,
            "hovermode": "closest",
            "showlegend": True,
            "margin": {"t": 80},
        }
        layout_json = json.dumps(layout_obj)

        # Title template ('{method}' placeholder for JS)
        title_template = f"Interactive 3D {viz_method} - {{method}} on {latent_dim}D Latent Space | Epoch {epoch}, n={n_samples}"

        # Build dropdown options HTML
        options_html = "\n".join(
            f'        <option value="{name}">{name}</option>' for name in method_names
        )

        # JavaScript code (plain string — no f-string brace escaping needed)
        js_code = r"""
(function() {
  var DATA = __DATA__;
  var LAYOUT = __LAYOUT__;
  var TITLE_TPL = '__TITLE_TPL__';

  var gd = document.getElementById('plotly-div');
  var tooltip = document.getElementById('hover-tooltip');
  var tooltipImg = document.getElementById('tooltip-img');
  var tooltipText = document.getElementById('tooltip-text');
  var lastMouseX = 0, lastMouseY = 0;
  var currentMethod = DATA.methodNames[0];

  document.addEventListener('mousemove', function(e) {
    lastMouseX = e.clientX;
    lastMouseY = e.clientY;
  });

  function getArchMap(methodName) {
    // Returns {archetype_name -> cluster_id}
    var method = DATA.clustering[methodName];
    var m = {};
    if (DATA.archetypes && method && method.archLabels) {
      for (var j = 0; j < DATA.archetypes.names.length; j++) {
        m[DATA.archetypes.names[j]] = method.archLabels[j];
      }
    }
    return m;
  }

  function resolveArchName(urlId) {
    // urlId -> archetype_name using pre-built map (handles "Web_" prefix etc.)
    if (!DATA.archIdMap) return null;
    var key = urlId ? urlId.toLowerCase() : '';
    return DATA.archIdMap[key] || null;
  }

  function buildTraces(methodName) {
    var method = DATA.clustering[methodName];
    if (!method) return [];
    var traces = [];
    var k = method.k;
    // Compute archMap once for the whole call (not per point)
    var archMap = getArchMap(methodName);

    for (var c = 0; c < k; c++) {
      // Per-cluster info: same for all points in this cluster — compute once
      var dom = method.clusterDominant ? method.clusterDominant[String(c)] : null;
      var archStr = '';
      if (dom != null) {
        var archName = resolveArchName(dom);
        var archCluster = (archName !== null && archMap[archName] !== undefined) ? archMap[archName] : null;
        if (archCluster !== null) {
          archStr = '<br>\uD83C\uDFDB ' + (archCluster === c ? '\u2705' : '\u274c cluster\u00a0' + archCluster) + '\u00a0' + dom;
        }
      }

      var cx = [], cy = [], cz = [], texts = [], cdata = [];
      for (var i = 0; i < DATA.x.length; i++) {
        if (method.labels[i] === c) {
          cx.push(DATA.x[i]);
          cy.push(DATA.y[i]);
          cz.push(DATA.z[i]);
          var gtStr = '';
          if (method.gtLabels && method.gtLabels[i] !== null && method.gtLabels[i] !== undefined) {
            var isCorrect = method.correct[i];
            gtStr = '<br>' + (isCorrect === true ? '\u2705 ' : (isCorrect === false ? '\u274c ' : '\u2753 ')) + method.gtLabels[i];
          }
          texts.push('<b>' + DATA.names[i] + '</b><br>Cluster ' + c + gtStr + archStr);
          cdata.push(i);
        }
      }
      if (cx.length === 0) continue;
      var col = DATA.colors[c % DATA.colors.length];
      traces.push({
        type: 'scatter3d', mode: 'markers',
        x: cx, y: cy, z: cz,
        name: 'Cluster ' + c + ' (' + cx.length + ')',
        marker: {size: 6, color: 'rgb(' + col[0] + ',' + col[1] + ',' + col[2] + ')', opacity: 0.8,
                 line: {width: 0.5, color: 'white'}},
        text: texts,
        hovertemplate: '%{text}<extra></extra>',
        customdata: cdata
      });
    }

    // Add archetypes
    if (DATA.archetypes && method.archLabels) {
      var arch = DATA.archetypes;
      for (var i = 0; i < arch.x.length; i++) {
        var clusterId = method.archLabels[i];
        var col = DATA.colors[clusterId % DATA.colors.length];
        var archIdx = DATA.nTrain + i;
        traces.push({
          type: 'scatter3d', mode: 'markers+text',
          x: [arch.x[i]], y: [arch.y[i]], z: [arch.z[i]],
          name: '\u2605 ' + arch.names[i] + ' (Cluster ' + clusterId + ')',
          marker: {size: 15, color: 'rgb(' + col[0] + ',' + col[1] + ',' + col[2] + ')',
                   symbol: 'diamond', line: {color: 'white', width: 2}},
          text: [arch.names[i]],
          textposition: 'top center',
          textfont: {size: 10, color: 'black'},
          hovertemplate: '<b>' + arch.names[i] + '</b><br>Cluster ' + clusterId + ' (' + methodName + ')<extra></extra>',
          customdata: [archIdx]
        });
      }
    }
    return traces;
  }

  function updateMetricsPanel(methodName) {
    var panel = document.getElementById('metrics-panel');
    if (!DATA.metrics || !DATA.metrics[methodName]) {
      panel.style.display = 'none';
      return;
    }
    var m = DATA.metrics[methodName];
    function colorCls(v) { return v >= 0.5 ? 'metric-good' : (v >= 0.2 ? '' : 'metric-bad'); }
    function row(label, val) {
      return '<tr><td class="metric-name">' + label + '</td>' +
             '<td class="metric-val ' + colorCls(val) + '">' + val.toFixed(3) + '</td></tr>';
    }
    panel.innerHTML =
      '<h4>\uD83D\uDCCA Metrics vs Ground Truth</h4>' +
      '<table>' +
      row('ARI', m.ari) + row('AMI', m.ami) + row('NMI', m.nmi) +
      row('Homogeneity', m.homogeneity) + row('Completeness', m.completeness) +
      row('V-measure', m.v_measure) +
      '</table>';
    panel.style.display = 'block';
  }

  function updatePlot(methodName) {
    var traces = buildTraces(methodName);
    var updatedLayout = JSON.parse(JSON.stringify(LAYOUT));
    updatedLayout.title = TITLE_TPL.replace('{method}', methodName) + ', k=' + DATA.clustering[methodName].k;
    Plotly.react(gd, traces, updatedLayout, {responsive: true});
    updateMetricsPanel(methodName);
  }

  // Initial plot
  updatePlot(currentMethod);

  // Method switch handler
  document.getElementById('method-selector').addEventListener('change', function() {
    currentMethod = this.value;
    updatePlot(currentMethod);
  });

  // Hover tooltip
  gd.on('plotly_hover', function(eventData) {
    try {
      var pt = eventData.points[0];
      var idx = pt.customdata;
      if (Array.isArray(idx)) idx = idx[0];
      if (idx == null || idx >= DATA.images.length) return;
      var img = DATA.images[idx];
      var name = DATA.names[idx] || 'Sample ' + idx;
      var cluster = pt.fullData.name || 'Unknown';
      var gtInfo = '';
      var method = DATA.clustering[currentMethod];
      if (method && method.gtLabels && idx < method.gtLabels.length && method.gtLabels[idx] !== null) {
        var isOk = method.correct[idx];
        gtInfo = '<br>' + (isOk === true ? '\u2705 ' : (isOk === false ? '\u274c ' : '\u2753 ')) + method.gtLabels[idx];
      }
      var archInfo = '';
      if (method && method.clusterDominant && idx < method.labels.length) {
        var clOfPt = method.labels[idx];
        var dom = method.clusterDominant[String(clOfPt)];
        if (dom != null) {
          var archName = resolveArchName(dom);
          var archMap = getArchMap(currentMethod);
          var archCl = (archName !== null && archMap[archName] !== undefined) ? archMap[archName] : null;
          if (archCl !== null) {
            archInfo = '<br>\uD83C\uDFDB\u00a0' + dom + '\u00a0' + (archCl === clOfPt ? '\u2705' : '\u274c\u00a0cluster\u00a0' + archCl);
          }
        }
      }
      if (img) {
        tooltipImg.src = img;
        tooltipText.innerHTML = '<b>' + name + '</b><br>' + cluster + gtInfo + archInfo;
        tooltip.style.display = 'block';
        var x = lastMouseX + 20;
        var y = lastMouseY - 120;
        if (x + 300 > window.innerWidth) x = lastMouseX - 300;
        if (y < 0) y = 10;
        if (y + 400 > window.innerHeight) y = window.innerHeight - 420;
        tooltip.style.left = x + 'px';
        tooltip.style.top = y + 'px';
      }
    } catch (e) {
      console.error('Hover error', e);
    }
  });

  gd.on('plotly_unhover', function() {
    tooltip.style.display = 'none';
  });
})();
"""
        # Replace JS placeholders with serialized data
        js_code = js_code.replace("__DATA__", data_json)
        js_code = js_code.replace("__LAYOUT__", layout_json)
        js_code = js_code.replace("__TITLE_TPL__", title_template)

        # Final HTML
        html = f"""<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Interactive 3D {viz_method} - Epoch {epoch}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
      body {{ margin: 0; font-family: 'Segoe UI', Arial, sans-serif; }}
      #toolbar {{
        position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
        z-index: 100; background: rgba(255,255,255,0.95); padding: 8px 20px;
        border-radius: 8px; box-shadow: 0 2px 12px rgba(0,0,0,0.15);
        display: flex; align-items: center; gap: 12px;
      }}
      #toolbar label {{ font-weight: 600; font-size: 14px; color: #2c3e50; }}
      #method-selector {{
        padding: 6px 12px; font-size: 14px; border: 2px solid #3498db;
        border-radius: 6px; background: white; cursor: pointer; font-weight: 500;
      }}
      #method-selector:hover {{ border-color: #2980b9; }}
      #metrics-panel {{
        position: fixed; bottom: 20px; right: 20px; z-index: 100;
        background: rgba(255,255,255,0.95); padding: 14px 18px;
        border-radius: 8px; box-shadow: 0 2px 12px rgba(0,0,0,0.15);
        font-size: 13px; display: none; min-width: 260px;
      }}
      #metrics-panel h4 {{
        margin: 0 0 10px 0; font-size: 14px; color: #2c3e50;
        border-bottom: 1px solid #eee; padding-bottom: 6px;
      }}
      #metrics-panel table {{ border-collapse: collapse; width: 100%; }}
      #metrics-panel td {{ padding: 3px 8px; }}
      #metrics-panel .metric-name {{ color: #666; }}
      #metrics-panel .metric-val {{ font-weight: 600; color: #2c3e50; text-align: right; }}
      .metric-good {{ color: #27ae60 !important; }}
      .metric-bad {{ color: #e74c3c !important; }}
      #hover-tooltip {{
        position: fixed; background: white; border: 3px solid #2c3e50;
        border-radius: 10px; padding: 15px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.5);
        pointer-events: none; z-index: 9999; display: none;
        max-width: 280px; max-height: 400px;
      }}
      #hover-tooltip img {{
        display: block; width: 100%; max-height: 200px;
        object-fit: contain; border-radius: 6px; background: #f5f5f5;
      }}
      #hover-tooltip .info {{
        margin-top: 12px; font-size: 16px; font-weight: 600;
        color: #2c3e50; text-align: center; line-height: 1.4;
      }}
    </style>
  </head>
  <body>
    <div id="toolbar">
      <label>Clustering Method:</label>
      <select id="method-selector">
{options_html}
      </select>
    </div>
    <div id="plotly-div" style="width:100%;height:100vh;"></div>
    <div id="metrics-panel"></div>
    <div id="hover-tooltip">
      <img id="tooltip-img" src="" alt="Preview">
      <div class="info" id="tooltip-text"></div>
    </div>
    <script>
{js_code}
    </script>
  </body>
</html>"""

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding='utf-8')
        logging.info(f"[OK] Interactive 3D {viz_method} saved to {output_path} (methods: {', '.join(method_names)})")
        return True
    except ImportError:
        logging.warning("Required libraries not installed (plotly, matplotlib).")
        return None
    except Exception as e:
        logging.warning(f"Failed to create interactive 3D visualization: {e}")
        import traceback
        logging.debug(traceback.format_exc())
        return None


def log_latent_space_visualization(model, valid_loader, archetypes_dir, device, writer, epoch, max_height=2048, max_samples=1000):
    """Generate and log latent space visualisations to TensorBoard."""
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    import base64
    from io import BytesIO
    from tqdm import tqdm
    
    # 1. Encode the validation dataset
    # We encode ALL samples for clustering quality, but only store image tensors
    # for a random subset (max_samples) used in visualization, to save RAM and time.
    n_batches = len(valid_loader)
    total_dataset_size = len(valid_loader.dataset) if hasattr(valid_loader, 'dataset') else n_batches * valid_loader.batch_size
    logging.info(f"Encoding validation dataset ({total_dataset_size} images, {n_batches} batches)...")
    
    # Pre-select which global sample indices to keep images for
    rng = np.random.RandomState(42)
    if total_dataset_size > max_samples:
        keep_image_indices = set(rng.choice(total_dataset_size, size=max_samples, replace=False).tolist())
    else:
        keep_image_indices = None  # keep all
    
    model.eval()
    valid_latents = []
    valid_images_sparse = {}  # idx -> tensor, only for selected samples
    global_idx = 0

    use_amp = device.type == "cuda"
    with torch.inference_mode():
        for i, (inputs, targets, masks) in enumerate(tqdm(valid_loader, desc="Encoding validation", leave=False)):
            targets = targets.to(device)
            masks = masks.to(device)
            if use_amp:
                with torch.amp.autocast("cuda"):
                    _, mu, _ = model(targets, mask=masks)
            else:
                _, mu, _ = model(targets, mask=masks)
            # Adaptive Average Pooling to fixed 8x4 grid on valid region only
            if mu.dim() == 4:
                mask_latent = F.interpolate(masks, size=mu.shape[2:], mode='nearest')
                batch_size_mu = mu.size(0)
                for j in range(batch_size_mu):
                    # Find valid height/width for this specific latent
                    valid_h = int(mask_latent[j, 0, :, 0].sum().item())
                    valid_w = int(mask_latent[j, 0, 0, :].sum().item())
                    valid_h = max(valid_h, 1)
                    valid_w = max(valid_w, 1)
                    # Extract only the semantic part
                    mu_valid = mu[j:j+1, :, :valid_h, :valid_w]
                    # Apply pooling on the clean region
                    mu_pooled = F.adaptive_avg_pool2d(mu_valid, output_size=(8, 4)).flatten()
                    valid_latents.append(mu_pooled.cpu().numpy())
            else:
                valid_latents.append(mu.cpu().numpy())
            # Only store images we'll actually use for visualization
            batch_size = targets.size(0)
            for j in range(batch_size):
                if keep_image_indices is None or (global_idx + j) in keep_image_indices:
                    valid_images_sparse[global_idx + j] = targets[j].cpu()
            global_idx += batch_size
    
    if len(valid_latents) == 0:
        logging.warning("No validation samples encoded, skipping visualization")
        return
    
    valid_latents = np.concatenate(valid_latents, axis=0)
    latent_dim = valid_latents.shape[1]
    n_total = len(valid_latents)

    logging.info(f"Encoded {n_total} validation samples (latent_dim={latent_dim})")
    logging.info(f"Stored {len(valid_images_sparse)} images for visualization (max_samples={max_samples})")
    
    # 2. Load archetypes to determine k
    archetype_latents, archetype_labels, archetype_names, archetype_images = load_archetypes(archetypes_dir, model, device, max_height)
    
    if archetype_latents is None:
        logging.warning("No archetypes loaded, using k=15 by default")
        k = 15
        archetype_names = [f"Cluster_{i}" for i in range(k)]
    else:
        k = len(archetype_names)
        logging.info(f"Loaded {k} archetypes: {', '.join(archetype_names)}")
    
    # 3. K-means clustering on latent space before dimensionality reduction
    logging.info(f"Applying k-means (k={k}) on FULL {latent_dim}D latent space ({n_total} samples)...")
    kmeans_full = KMeans(n_clusters=k, random_state=42, n_init=10)
    valid_cluster_labels_full = kmeans_full.fit_predict(valid_latents)
    
    # 4. Assign archetypes to clusters (on full latent space)
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
    
    # 5. Subsample for visualization (clustering already done on full dataset)
    # Use the same indices we pre-selected for image storage
    if n_total > max_samples:
        viz_indices = sorted(valid_images_sparse.keys())
        # In case some images failed to load, limit to what we actually have
        viz_indices = [i for i in viz_indices if i < n_total][:max_samples]
        logging.info(f"Subsampling {len(viz_indices)} points out of {n_total} for visualization (clustering was on all {n_total})...")
        viz_indices = np.array(viz_indices)
        viz_latents = valid_latents[viz_indices]
        viz_cluster_labels = valid_cluster_labels_full[viz_indices]
        viz_images = [valid_images_sparse[i] for i in viz_indices]
        n_samples = len(viz_indices)
    else:
        viz_latents = valid_latents
        viz_cluster_labels = valid_cluster_labels_full
        # Build ordered list from sparse dict
        viz_images = [valid_images_sparse[i] for i in sorted(valid_images_sparse.keys()) if i < n_total]
        n_samples = n_total

    logging.info(f"Visualization will display {n_samples} points (clustered on {n_total})")

    # 6. PCA / t-SNE on the subsampled data
    try:
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        
        perplexity = min(30.0, max(5.0, n_samples / 3))
        
        # PCA on subsampled latents
        logging.info(f"Generating PCA projection (3 components) on {n_samples} samples...")
        pca_3d = PCA(n_components=3, random_state=42)
        z_pca_3d = pca_3d.fit_transform(viz_latents)
        logging.info(f"PCA explained variance: {pca_3d.explained_variance_ratio_[0]:.3f}, "
                    f"{pca_3d.explained_variance_ratio_[1]:.3f}, {pca_3d.explained_variance_ratio_[2]:.3f}")
        z_pca_2d = z_pca_3d[:, :2]
        
        # t-SNE on subsampled latents (if enough samples)
        if n_samples >= 50:
            logging.info(f"Generating t-SNE 3D projection with perplexity={perplexity:.1f} on {n_samples} samples...")
            tsne_3d = TSNE(n_components=3, random_state=42, perplexity=perplexity, 
                          init="pca", learning_rate="auto")
            z_tsne_3d = tsne_3d.fit_transform(viz_latents)
            z_tsne_2d = z_tsne_3d[:, :2]
            
            visualizations_2d = [("PCA", z_pca_2d, pca_3d), ("t-SNE", z_tsne_2d, None)]
            visualizations_3d = [("PCA", z_pca_3d, pca_3d), ("t-SNE", z_tsne_3d, None)]
        else:
            logging.info(f"Skipping t-SNE (n_samples={n_samples} < 50)")
            visualizations_2d = [("PCA", z_pca_2d, pca_3d)]
            visualizations_3d = [("PCA", z_pca_3d, pca_3d)]
        
        # Generate a figure for each visualisation
        colors_list = _generate_distinct_colors(k)
        colors = np.array([[r/255, g/255, b/255] for r, g, b in colors_list])
        
        # Process both 2D and 3D visualizations
        for idx, ((viz_method_2d, z_embedded_2d, projection_model_2d), 
                  (viz_method_3d, z_embedded_3d, projection_model_3d)) in enumerate(zip(visualizations_2d, visualizations_3d)):
            
            viz_method = viz_method_2d  # Same for both
            
            logging.info(f"Visualizing {viz_method} projection with clusters from full {latent_dim}D k-means ({n_samples} displayed / {n_total} total)...")
            
            # Project archetypes to 2D and 3D space for visualization
            arch_embedded_2d = None
            arch_embedded_3d = None
            
            if archetype_latents is not None:
                logging.info(f"Projecting archetypes to {viz_method} space...")
                
                # Project archetypes
                if projection_model_2d is not None:  # PCA
                    arch_embedded_2d = projection_model_2d.transform(archetype_latents)[:, :2]
                    arch_embedded_3d = projection_model_3d.transform(archetype_latents)
                else:  # t-SNE: use nearest-neighbour approximation
                    from sklearn.neighbors import NearestNeighbors
                    nbrs = NearestNeighbors(n_neighbors=1).fit(viz_latents)
                    _, indices = nbrs.kneighbors(archetype_latents)
                    arch_embedded_2d = z_embedded_2d[indices.flatten()]
                    arch_embedded_3d = z_embedded_3d[indices.flatten()]
                                
            # ===== Create Static 2D Matplotlib Visualization for TensorBoard =====
            fig, ax = plt.subplots(figsize=(14, 10))
            
            # Colour by k-means cluster (computed on full latent space)
            for cluster_id in range(k):
                mask = viz_cluster_labels == cluster_id
                if np.sum(mask) > 0:
                    # Cluster label (use archetype name if available)
                    if archetype_latents is not None and cluster_id in cluster_to_archetypes_full:
                        archs = cluster_to_archetypes_full[cluster_id]
                        label = f"C{cluster_id}: {archs[0]}" if len(archs) == 1 else f"C{cluster_id}: {len(archs)} archs"
                    else:
                        label = f"Cluster {cluster_id}"
                    
                    ax.scatter(z_embedded_2d[mask, 0], z_embedded_2d[mask, 1], 
                              c=[colors[cluster_id]], label=label, s=30, alpha=0.6, edgecolors='none')
            
            # Overlay archetype markers (stars coloured by cluster)
            if archetype_latents is not None and arch_embedded_2d is not None:
                for i, (name, cluster_id) in enumerate(zip(archetype_names, archetype_cluster_assignments_full)):
                    ax.scatter(arch_embedded_2d[i, 0], arch_embedded_2d[i, 1], 
                              c=[colors[cluster_id]], marker='*', s=500, 
                              edgecolors='white', linewidths=2, zorder=10)
                    
                    # Annotate
                    ax.annotate(name, (arch_embedded_2d[i, 0], arch_embedded_2d[i, 1]),
                               fontsize=8, ha='center', va='bottom', 
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
            
            ax.set_title(f"Latent Space {viz_method} - K-means on Full {latent_dim}D (Epoch {epoch}, displayed={n_samples}/{n_total}, k={k})", fontsize=14)
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

