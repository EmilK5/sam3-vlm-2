"""Portable final-experiment tables and plain rectangle image exports.

All measurements come from the completed run records. Rendering is observational:
no probability threshold is introduced to make soft counts look like box counts.
"""

import csv
import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from PIL import ImageDraw

TABLE_FILES = ('aggregate_results.csv', 'per_image_results.csv', 'counts_by_image.csv',
               'results_summary.md', 'bbox_manifest.json')


def overlay_path(output_dir, variant, sample_id):
    def component(value):
        value = str(value)
        if re.fullmatch(r'[A-Za-z0-9_-]+', value):
            return value
        prefix = re.sub(r'[^A-Za-z0-9_-]', '_', value)[:80]
        return prefix + '_' + hashlib.sha256(value.encode()).hexdigest()[:12]
    return Path(output_dir) / 'bbox_images' / component(variant) / (component(sample_id) + '.png')


def render_candidate_boxes(image, graph, path):
    """Draw ONLY red rectangle outlines on an original-resolution RGB copy."""
    canvas = image.convert('RGB').copy()
    width, height = canvas.size
    pen = ImageDraw.Draw(canvas)
    line_width = max(2, round(min(width, height) / 400))
    drawn, outside = 0, 0
    for node in graph.active_nodes():
        box = node.geometry.bbox()
        coords = box.as_tuple()
        if box.coordinate_space != 'image' or not all(math.isfinite(v) for v in coords):
            raise ValueError('Final boxes must have finite original-image coordinates')
        x1, y1, x2, y2 = coords
        if x2 <= x1 or y2 <= y1:
            raise ValueError('Final boxes must have positive area')
        if x2 <= 0 or y2 <= 0 or x1 >= width or y1 >= height:
            outside += 1
            continue
        clipped = (max(0, min(width - 1, round(x1))), max(0, min(height - 1, round(y1))),
                   max(0, min(width - 1, round(x2))), max(0, min(height - 1, round(y2))))
        pen.rectangle(clipped, outline=(255, 48, 48), width=line_width)
        drawn += 1
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format='PNG')
    return {'bbox_image': str(path), 'bbox_count': drawn, 'bbox_outside_image_count': outside}


def _write_csv(path, rows, fields):
    def cell(value):
        # Keep numeric values numeric, including negative signed errors. Protect
        # user-provided text identifiers when opened in spreadsheet applications.
        if isinstance(value, str) and value.lstrip().startswith(('=', '+', '-', '@')):
            return "'" + value
        return value
    with Path(path).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: cell(row.get(k)) for k in fields})


def write_final_outputs(report, output_dir):
    """Write all-image and aggregate tables; bundle only this report's overlays."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = report['metadata']['variants']
    samples = report['samples']
    expected = report['metadata']['sample_count']
    by_variant = {v: {} for v in variants}
    ground_truth = {}
    for row in samples:
        v, key = row['variant'], row['sample_id']
        if key in by_variant[v]:
            raise ValueError(f'Duplicate final result: {v}/{key}')
        if key in ground_truth and ground_truth[key] != row['gt_count']:
            raise ValueError(f'Inconsistent GT for {key}')
        by_variant[v][key] = row
        ground_truth[key] = row['gt_count']
    common = set.intersection(*(set(k for k, r in by_variant[v].items() if r.get('success'))
                               for v in variants)) if variants else set()
    aggregate_rows = []
    for v in variants:
        rows = list(by_variant[v].values())
        valid = [r for r in rows if r.get('success')]
        a = report['aggregates'].get(v, {})
        abs_errors = [r['absolute_error'] for r in valid]
        aggregate_rows.append({
            'variant': v, 'count_type': rows[0].get('count_type') if rows else None,
            'n_expected': expected, 'n_success': len(valid), 'n_failed_or_missing': expected - len(valid),
            'complete': len(valid) == expected,
            'GT_total_all_images': sum(ground_truth.values()),
            'GT_total_successful_images': sum(r['gt_count'] for r in valid),
            'predicted_total_successful_images': sum(r['predicted_count'] for r in valid) if valid else None,
            'MAE': a.get('MAE'), 'MSE': a.get('MSE'), 'RMSE': a.get('RMSE'),
            'MRE_percent': a['MRE'] * 100 if a.get('MRE') is not None else None,
            'mean_signed_error': a.get('mean_signed_error'),
            'median_absolute_error': statistics.median(abs_errors) if abs_errors else None,
            'max_absolute_error': max(abs_errors) if abs_errors else None,
            'n_common_success': len(common),
            'MAE_common_images': statistics.mean(by_variant[v][k]['absolute_error'] for k in common) if common else None,
            'mean_runtime_seconds': a['avg_runtime_ms'] / 1000 if a.get('avg_runtime_ms') is not None else None,
            'mean_qwen_calls': a.get('avg_qwen_calls'), 'mean_sam3_calls': a.get('avg_sam3_calls'),
            'mean_sam3_tiles': a.get('avg_sam3_tiles'),
        })
    _write_csv(output_dir / TABLE_FILES[0], aggregate_rows, list(aggregate_rows[0]) if aggregate_rows else ['variant'])

    per_image = [dict(r,
        runtime_seconds=r['runtime_ms'] / 1000 if r.get('runtime_ms') is not None else None,
        relative_error_percent=r['relative_error'] * 100 if r.get('relative_error') is not None else None,
    ) for r in samples]
    _write_csv(output_dir / TABLE_FILES[1], per_image, [
        'sample_id', 'image_path', 'target', 'variant', 'gt_count', 'predicted_count', 'count_type', 'candidate_count',
        'absolute_error', 'signed_error', 'squared_error', 'relative_error_percent', 'success',
        'runtime_seconds', 'qwen_calls', 'sam3_calls', 'sam3_tiles', 'replans', 'stop_reason',
        'bbox_count', 'bbox_outside_image_count', 'bbox_image', 'failure_message', 'run_id',
    ])
    wide = []
    for key, gt in ground_truth.items():
        row = {'sample_id': key, 'gt_count': gt}
        for v in variants:
            result = by_variant[v].get(key, {})
            row[v] = result.get('predicted_count') if result.get('success') else None
        wide.append(row)
    _write_csv(output_dir / TABLE_FILES[2], wide, ['sample_id', 'gt_count'] + variants)

    def number(value):
        return f'{value:.3f}' if value is not None else '—'
    lines = ['# Final A–E results', '',
        '| Variant | Complete runs | GT total (successful images) | Predicted total | MAE | RMSE | MRE (%) | Signed error | Mean seconds |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for a in aggregate_rows:
        lines.append(f"| {a['variant']} | {a['n_success']}/{expected} | {a['GT_total_successful_images']} | "
                     + ' | '.join(number(a[k]) for k in ['predicted_total_successful_images', 'MAE', 'RMSE',
                         'MRE_percent', 'mean_signed_error', 'mean_runtime_seconds']) + ' |')
    lines += ['',
        'A/B report hard candidate counts. C/D/E report the sum of target probabilities (soft counts).',
        'A/B target passes use 0.20. C/D/E preserve the selected D bootstrap at 0.25 and Qwen positives at 0.20.',
        'C/D/E use the same prompt and negative-evidence policy. Qwen caps are 1/2/100. E keeps its '
        '1000-SAM3 cap and saturation policy with no separate tile, iteration or total-runtime cap.',
        'Metrics use successful runs only. CSV tables expose failures and MAE on the common successful image set. '
        'Relative error excludes zero-GT images; an unavailable value is blank in CSV.',
        'Each PNG contains only red rectangle outlines for final active candidates, without text, scores, IDs, '
        'masks or confidence filtering. Rectangle counts need not equal soft counts.',
        'Ground truth counts fruit on trees. Counts alone do not provide detection precision/recall or box IoU metrics.',
        'This 34-image development set informed parameter selection. The final run is not an independent held-out evaluation.']
    (output_dir / TABLE_FILES[3]).write_text('\n'.join(lines) + '\n')

    manifest = []
    with ZipFile(output_dir / 'bbox_images.zip', 'w', ZIP_DEFLATED) as archive:
        for row in samples:
            if not row.get('bbox_image'):
                continue
            path = Path(row['bbox_image'])
            relative = path.resolve().relative_to(output_dir.resolve())
            archive.write(path, str(relative))
            manifest.append({k: row.get(k) for k in ['sample_id', 'variant', 'candidate_count', 'bbox_count',
                'bbox_outside_image_count', 'success']} | {'path': str(relative)})
    (output_dir / TABLE_FILES[4]).write_text(json.dumps(manifest, indent=2))
    return {'tables': list(TABLE_FILES), 'bbox_archive': 'bbox_images.zip',
            'bbox_images_exported': len(manifest), 'bbox_policy': 'all_active_candidates_rectangles_only'}
