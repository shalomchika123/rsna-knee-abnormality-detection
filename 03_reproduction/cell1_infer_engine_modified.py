%%writefile /kaggle/working/infer.py
# Inference engine of Team Kangaroo (1st place, RSNA Knee 2026; tascj / liushuzhi), from their public notebook
# "knee final 20261022". Modified by us: one extra run, "ours_256" = OUR box-pooling classifier (ConvNeXt-T,
# 384 px frame, 112 mm crop at 256 px, trained on fold 0), "ours_full" = the same recipe trained on all the
# data for 5 epochs, and a loading fix for their checkpoints.
import argparse
import contextlib
import io
import json
import math
import os
import re
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import pydicom
import timm
import torch
import torch.nn.functional as F
from torch import nn
LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
COMPETITION = 'rsna-knee-abnormality-detection'
RUNS = {'ours_256': 'convnext_t', 'ours_full': 'convnext_t', 'convnext_t_5fold': 'convnext_t', 'mambaout_full': 'mambaout_base_plus', 'maxxvit_full': 'maxxvitv2_base', 'convnext_t_6channel_full': 'convnext_t', 'mambaout_6channel_full': 'mambaout_base_plus'}
RUN_OPTIONS = {'ours_256': {'attention': True}, 'ours_full': {'attention': True}, 'convnext_t_5fold': {'attention': True}, 'mambaout_full': {'attention': True, 'grid_offset': -15.5, 'sources': 7, 'source_hidden': 768}, 'maxxvit_full': {'attention': True, 'grid_offset': -8.0, 'sources': 7, 'source_hidden': 1024}, 'convnext_t_6channel_full': {'attention': True, 'grid_offset': 0.0, 'slot_channels': True}, 'mambaout_6channel_full': {'attention': True, 'grid_offset': -15.5, 'slot_channels': True}}
RUN_MODELS = {'ours_256': ['fold0_ema.pth'], 'ours_full': ['fold5_ema.pth'], 'mambaout_full': ['seed1.pth', 'seed2.pth', 'seed3.pth'], 'maxxvit_full': ['seed1.pth', 'seed2.pth', 'seed3.pth'], 'convnext_t_6channel_full': ['seed1.pth', 'seed2.pth', 'seed3.pth'], 'mambaout_6channel_full': ['seed1.pth', 'seed2.pth', 'seed3.pth']}
RUN_CROPS = {'ours_256': {'labels': ['ACL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA'], 'size_mm': 112.0, 'image_mm': 140.0, 'pixels': 256}, 'ours_full': {'labels': ['ACL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA'], 'size_mm': 112.0, 'image_mm': 140.0, 'pixels': 256}, 'convnext_t_5fold': {'labels': ['ACL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA'], 'size_mm': 112.0, 'image_mm': 140.0, 'pixels': 384}, 'mambaout_full': {'labels': ['ACL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA'], 'size_mm': 112.0, 'image_mm': 140.0, 'pixels': 384}, 'maxxvit_full': {'labels': ['ACL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA'], 'size_mm': 112.0, 'image_mm': 140.0, 'pixels': 384}, 'convnext_t_6channel_full': {'labels': ['ACL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA'], 'size_mm': 112.0, 'image_mm': 140.0, 'pixels': 384}, 'mambaout_6channel_full': {'labels': ['ACL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA'], 'size_mm': 112.0, 'image_mm': 140.0, 'pixels': 384}}
RUN_SIZES = {'ours_256': 384, 'ours_full': 384, 'convnext_t_5fold': 512, 'mambaout_full': 512, 'maxxvit_full': 512, 'convnext_t_6channel_full': 512, 'mambaout_6channel_full': 512}
NAMES = {'ours_256': 'knee-ours-classifier', 'ours_full': 'knee-classifier-full', 'convnext_t_5fold': 'knee-models-convnext-t', 'mambaout_full': 'knee-models-mambaout', 'maxxvit_full': 'knee-models-maxxvit', 'convnext_t_6channel_full': 'knee-models-convnext-t-6-channel', 'mambaout_6channel_full': 'knee-models-mambaout-6-channel'}

def dataset_name(run):
    return NAMES[run]

def kaggle_input(name):
    for pattern in (name, f'*/{name}', f'*/*/{name}'):
        found = sorted((p for p in Path('/kaggle/input').glob(pattern) if p.is_dir()))
        if found:
            return found[0]
    return Path('/kaggle/input') / name
ARCHS = {'convnext_t': {'backbone': 'convnext_tiny.in12k_ft_in1k'}, 'mambaout_base_plus': {'backbone': 'mambaout_base_plus_rw.sw_e150_in12k_ft_in1k', 'feature_map': 'intermediates'}, 'maxxvitv2_base': {'backbone': 'maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k'}}
FOLDS = 5
FOV_MM, IMAGE_SIZE, MAX_DEPTH, WINDOW = (140, 256, 64, (0.5, 99.5))
SIDE_SIZE = 224
DETECTOR_SIZE = 160
PAD_DEPTH = [32, 40, 48, 64, 128, 192, 320]
PAD_MULTIPLE = 32
PAD_TAIL = 8
LEFT = '(?:^|[^A-Z])(?:L|LT|LEFT|LINKS)(?:[^A-Z]|$)'
RIGHT = '(?:^|[^A-Z])(?:R|RT|RIGHT|RECHTS)(?:[^A-Z]|$)'
LATERALITY = {'L': 'left', 'R': 'right', 'LEFT': 'left', 'RIGHT': 'right'}

def truncated(ds):
    if ds.file_meta.TransferSyntaxUID.is_compressed:
        return False
    expected = ds.Rows * ds.Columns * int(ds.get('SamplesPerPixel', 1)) * ds.BitsAllocated // 8
    return len(ds.PixelData) < expected
_READ_POOL = None

def read_files(paths, threads):
    global _READ_POOL
    if threads <= 1:
        return {p: Path(p).read_bytes() for p in paths}
    if _READ_POOL is None:
        _READ_POOL = ThreadPoolExecutor(threads)
    return dict(zip(paths, _READ_POOL.map(lambda p: Path(p).read_bytes(), paths), strict=True))

def series_files(directory):
    return [os.path.join(directory, f) for f in sorted(os.listdir(directory)) if f.endswith('.dcm')]

def read_dicom_series(directory, blobs=None):
    files = series_files(directory)
    if not files:
        raise ValueError(f'No .dcm files in {directory}')
    datasets = [pydicom.dcmread(io.BytesIO(blobs[f]) if blobs and f in blobs else f) for f in files]
    datasets = [ds for ds in datasets if not truncated(ds)]
    if not datasets:
        raise ValueError(f'All slices truncated in {directory}')
    first = datasets[0]
    orientation = first.get('ImageOrientationPatient')
    normal = None
    if orientation is not None and len(orientation) == 6:
        row, col = (np.asarray(orientation[:3], float), np.asarray(orientation[3:], float))
        n = np.cross(row, col)
        normal = n / np.linalg.norm(n) if np.linalg.norm(n) > 0 else None
    records = []
    for ds in datasets:
        if int(ds.get('NumberOfFrames', 1) or 1) != 1:
            raise ValueError('Multi-frame DICOM')
        position = ds.get('ImagePositionPatient')
        records.append({'ds': ds, 'instance_number': int(ds.get('InstanceNumber', 0) or 0), 'location': None if position is None or normal is None else float(np.dot(normal, np.asarray(position, float))), 'ipp': None if position is None else [float(v) for v in position], 'iop': [float(v) for v in ds.get('ImageOrientationPatient', [])], 'slope': float(ds.get('RescaleSlope', 1) or 1), 'intercept': float(ds.get('RescaleIntercept', 0) or 0), 'pixel_spacing': [float(v) for v in ds.get('PixelSpacing', [0, 0])], 'pixels': ds.pixel_array})
    by_position = all((r['location'] is not None for r in records))
    if by_position:
        records.sort(key=lambda r: (r['location'], r['instance_number']))
    else:
        records.sort(key=lambda r: r['instance_number'])
    if first.get('PhotometricInterpretation', 'MONOCHROME2') != 'MONOCHROME2':
        raise ValueError(f'Unsupported PhotometricInterpretation {first.PhotometricInterpretation}')
    low = min((int(r['pixels'].min()) for r in records))
    high = max((int(r['pixels'].max()) for r in records))
    offset = min(0, low)
    if high - offset > np.iinfo(np.uint16).max:
        raise ValueError(f'Value range {low}..{high} does not fit uint16')
    pixels = [(r['pixels'].astype(np.int32) - offset).astype(np.uint16) for r in records]
    steps = np.diff([r['location'] for r in records]) if by_position and len(records) > 1 else []
    head = records[0]['ds']
    return {'pixels': pixels, 'slope': [r['slope'] for r in records], 'intercept': [r['intercept'] for r in records], 'offset': offset, 'n_slices': len(pixels), 'rows': pixels[0].shape[0], 'cols': pixels[0].shape[1], 'ps_row': records[0]['pixel_spacing'][0], 'ps_col': records[0]['pixel_spacing'][1], 'slice_spacing': float(np.median(steps)) if len(steps) else float('nan'), 'laterality': str(head.get('Laterality') or head.get('ImageLaterality') or '').upper(), 'description': str(head.get('SeriesDescription') or '').upper(), 'ipp': None if any((r['ipp'] is None for r in records)) else np.array([r['ipp'] for r in records]), 'orientation': [float(f'{v:.6f}') for v in records[0]['iop']] if len(records[0]['iop']) == 6 else None}

def header_side(series):
    dicom = {LATERALITY.get(s['laterality']) for s in series} - {None}
    if len(dicom) == 1:
        return (dicom.pop(), 'dicom')
    description = set()
    for s in series:
        left, right = (re.search(LEFT, s['description']), re.search(RIGHT, s['description']))
        if left and (not right):
            description.add('left')
        elif right and (not left):
            description.add('right')
    if len(description) == 1:
        return (description.pop(), 'description')
    return (None, None)

def center_crop_pad(image, size):
    out = np.zeros((size, size), image.dtype)
    (h, w), src, dst = (image.shape, [], [])
    for n in (h, w):
        if n >= size:
            start = (n - size) // 2
            src.append(slice(start, start + size))
            dst.append(slice(0, size))
        else:
            start = (size - n) // 2
            src.append(slice(0, n))
            dst.append(slice(start, start + n))
    out[dst[0], dst[1]] = image[src[0], src[1]]
    return out

def window_uint8(volume, low, high):
    lo, hi = (float(v) for v in np.percentile(volume[:, ::4, ::4], [low, high]))
    out = volume - np.float32(lo)
    out *= np.float32(255.0 / max(hi - lo, 1e-06))
    np.clip(out, 0, 255, out=out)
    return out.astype(np.uint8)

def resample_depth(volume, depth):
    positions = np.linspace(0, len(volume) - 1, depth)
    lower = np.floor(positions).astype(int)
    upper = np.minimum(lower + 1, len(volume) - 1)
    weight = (positions - lower)[:, None, None].astype(np.float32)
    return volume[lower] * (1 - weight) + volume[upper] * weight

def cap_depth(volume):
    if len(volume) > MAX_DEPTH:
        volume = np.rint(resample_depth(volume.astype(np.float32), MAX_DEPTH)).astype(np.uint8)
    return volume

def classifier_volume(s, out_size=IMAGE_SIZE, cap=True):
    zoom_y, zoom_x = (out_size * v / FOV_MM for v in (s['ps_row'], s['ps_col']))
    planes = []
    for raw, slope, intercept in zip(s['pixels'], s['slope'], s['intercept'], strict=True):
        h, w = raw.shape
        size = (max(1, round(w * zoom_x)), max(1, round(h * zoom_y)))
        interpolation = cv2.INTER_AREA if min(zoom_x, zoom_y) < 1 else cv2.INTER_LINEAR
        small = cv2.resize(raw, size, interpolation=interpolation)
        values = (small.astype(np.float32) + s['offset']) * slope + intercept
        planes.append(center_crop_pad(values, out_size))
    volume = window_uint8(np.stack(planes), *WINDOW)
    return cap_depth(volume) if cap else volume

def detector_image(s):
    raw = np.stack(s['pixels']).astype(np.float32)
    slope = np.asarray(s['slope'], np.float32)[:, None, None]
    intercept = np.asarray(s['intercept'], np.float32)[:, None, None]
    values = (raw + np.float32(s['offset'])) * slope + intercept
    low, high = np.percentile(values, [0.5, 99.5])
    volume = np.clip((values - low) * (255.0 / max(high - low, 1e-06)), 0, 255)
    volume = np.rint(volume).astype(np.uint8)
    h, w = volume.shape[1:]
    if max(h, w) < 512:
        scale = 512 / max(h, w)
        size = (round(w * scale), round(h * scale))
        volume = np.stack([cv2.resize(v, size, interpolation=cv2.INTER_LINEAR) for v in volume])
    return F.interpolate(torch.from_numpy(volume).float().div(255).unsqueeze(1), (DETECTOR_SIZE, DETECTOR_SIZE), mode='bilinear', align_corners=False, antialias=False).transpose(0, 1)

def image_boxes(boxes, s, size=IMAGE_SIZE):
    out = boxes.astype(np.float64)
    height = max(1, np.round(s['rows'] * (size * s['ps_row'] / FOV_MM)))
    width = max(1, np.round(s['cols'] * (size * s['ps_col'] / FOV_MM)))

    def offset(n):
        return -((n - size) // 2) if n >= size else (size - n) // 2
    top, left = (offset(height), offset(width))
    for k in (0, 3):
        out[:, k] = 2 * (left + out[:, k] * width) / size - 1
    for k in (1, 4):
        out[:, k] = 2 * (top + out[:, k] * height) / size - 1
    return out.astype(np.float32)

def uniform_rescale(s):
    return len(set(s['slope'])) == 1 and len(set(s['intercept'])) == 1 and (s['slope'][0] > 0)

def straddles(s, size=IMAGE_SIZE):
    return (size * s['ps_row'] / FOV_MM < 1) != (size * s['ps_col'] / FOV_MM < 1)

class StudyDataset(torch.utils.data.Dataset):

    def __init__(self, root, studies, series_table, sizes=(IMAGE_SIZE,), full_sizes=(), io_threads=8):
        self.root, self.studies, self.sizes = (Path(root), list(studies), tuple(sizes))
        self.io_threads = io_threads
        self.full_sizes = tuple(full_sizes)
        self.series = {k: g for k, g in series_table.groupby('StudyInstanceUID')}

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, index):
        study = self.studies[index]
        rows = self.series.get(study)
        out, errors = ([], [])
        rows = [] if rows is None else rows.sort_values('SeriesInstanceUID').to_dict('records')
        blobs = None
        if self.io_threads > 1:
            files = []
            for row in rows:
                try:
                    files += series_files(self.root / study / row['SeriesInstanceUID'])
                except OSError:
                    pass
            try:
                blobs = read_files(files, self.io_threads)
            except OSError:
                blobs = None
        for row in rows:
            uid = row['SeriesInstanceUID']
            try:
                s = read_dicom_series(self.root / study / uid, blobs)
                s.update(series=uid, plane=row['Anatomical_Plane'], fat=int(row['Fat_Suppression']), fluid=int(row['Fluid_Sensitive']))
                if not uniform_rescale(s):
                    s['detector'] = detector_image(s)
                s['volume'] = {n: classifier_volume(s, n) for n in self.sizes if straddles(s, n)}
                s['volume_full'] = {n: classifier_volume(s, n, cap=False) for n in self.full_sizes if straddles(s, n)}
                s['raw'] = torch.from_numpy(np.stack(s.pop('pixels')).view(np.int16))
                out.append(s)
            except Exception:
                errors.append(f'{study}/{uid}: {traceback.format_exc(limit=2)}')
        side, source = header_side(out)
        return {'study': study, 'series': out, 'side': side, 'source': source, 'errors': errors}
_MATRICES = {}

def resize_matrix(kind, n_in, n_out, device):
    key = (kind, n_in, n_out, str(device))
    if key not in _MATRICES:
        m = np.zeros((n_out, n_in), np.float32)
        scale = 1.0 / (n_out / n_in)
        for dx in range(n_out):
            if kind == 'area':
                fsx1 = dx * scale
                fsx2 = fsx1 + scale
                cell = min(scale, n_in - fsx1)
                sx1, sx2 = (math.ceil(fsx1), math.floor(fsx2))
                sx2 = min(sx2, n_in - 1)
                sx1 = min(sx1, sx2)
                if sx1 - fsx1 > 0.001:
                    m[dx, sx1 - 1] += np.float32((sx1 - fsx1) / cell)
                for sx in range(sx1, sx2):
                    m[dx, sx] += np.float32(1.0 / cell)
                if fsx2 - sx2 > 0.001:
                    m[dx, sx2] += np.float32(min(min(fsx2 - sx2, 1.0), cell) / cell)
            else:
                f = np.float32((dx + 0.5) * scale - 0.5)
                sx = math.floor(f)
                f = np.float32(f - np.float32(sx))
                if sx < 0:
                    f, sx = (np.float32(0), 0)
                if sx >= n_in - 1:
                    f, sx = (np.float32(0), n_in - 1)
                m[dx, sx] += np.float32(1) - f
                m[dx, min(sx + 1, n_in - 1)] += f
        _MATRICES[key] = to_device(m, device)
    return _MATRICES[key]

def resize_gpu(x, size, kind):
    wx = resize_matrix(kind, x.shape[2], size[0], x.device)
    wy = resize_matrix(kind, x.shape[1], size[1], x.device)
    return wy @ (x @ wx.T)

def to_device(x, device, dtype=None):
    t = torch.as_tensor(x, dtype=dtype)
    if torch.device(device).type != 'cuda':
        return t.to(device)
    return t.pin_memory().to(device, non_blocking=True)

def percentiles_gpu(order_stats, n, device, qs=WINDOW):
    ks, ts = ([], [])
    for q in qs:
        virtual = (n - 1) * (q / 100)
        prev = math.floor(virtual)
        ks += [prev, min(prev + 1, n - 1)]
        ts.append(virtual - prev)
    v = order_stats(to_device(ks, device, torch.int64)).float()
    a, b = (v[0::2].double(), v[1::2].double())
    diff = (v[1::2] - v[0::2]).double()
    t = to_device(ts, device, torch.float64)
    return torch.where(t >= 0.5, b - diff * (1 - t), a + diff * t)

def window_uint8_gpu(volume):
    sub = volume[:, ::4, ::4].flatten().sort().values
    lo, hi = percentiles_gpu(lambda ks: sub[ks], sub.numel(), volume.device)
    out = (volume - lo.float()) * (lo.new_full((), 255.0) / (hi - lo).clamp(min=1e-06)).float()
    return out.clamp(0, 255).to(torch.uint8)

def cap_depth_gpu(volume):
    if len(volume) <= MAX_DEPTH:
        return volume
    positions = np.linspace(0, len(volume) - 1, MAX_DEPTH)
    lower = np.floor(positions).astype(int)
    upper = np.minimum(lower + 1, len(volume) - 1)
    weight = to_device((positions - lower)[:, None, None].astype(np.float32), volume.device)
    lower, upper = (to_device(lower, volume.device, torch.int64), to_device(upper, volume.device, torch.int64))
    v = volume.float()
    return (v[lower] * (1 - weight) + v[upper] * weight).round().to(torch.uint8)

def rescale_gpu(small, s):
    slope = to_device(s['slope'], small.device, torch.float32)[:, None, None]
    intercept = to_device(s['intercept'], small.device, torch.float32)
    return (small + np.float32(s['offset'])) * slope + intercept[:, None, None]

def classifier_volume_gpu(u, s, out_size=IMAGE_SIZE, cap=True):
    zoom_y, zoom_x = (out_size * v / FOV_MM for v in (s['ps_row'], s['ps_col']))
    d, h, w = u.shape
    size = (max(1, round(w * zoom_x)), max(1, round(h * zoom_y)))
    kind = 'area' if min(zoom_x, zoom_y) < 1 else 'linear'
    values = rescale_gpu(resize_gpu(u, size, kind).round().clamp(0, 65535), s)
    out = values.new_zeros(d, out_size, out_size)
    src, dst = ([], [])
    for n in values.shape[1:]:
        if n >= out_size:
            start = (n - out_size) // 2
            src.append(slice(start, start + out_size))
            dst.append(slice(0, out_size))
        else:
            start = (out_size - n) // 2
            src.append(slice(0, n))
            dst.append(slice(start, start + n))
    out[:, dst[0], dst[1]] = values[:, src[0], src[1]]
    volume = window_uint8_gpu(out)
    return cap_depth_gpu(volume) if cap else volume

def side_volume_gpu(u, s):
    d, h, w = u.shape
    scale = SIDE_SIZE / max(h, w)
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    kind = 'area' if max(h, w) > SIDE_SIZE else 'linear'
    values = rescale_gpu(resize_gpu(u, size, kind).round().clamp(0, 65535), s)
    hh, ww = values.shape[1:]
    if hh != ww:
        side = max(hh, ww)
        square = values.new_zeros(d, side, side)
        top, left = ((side - hh) // 2, (side - ww) // 2)
        square[:, top:top + hh, left:left + ww] = values
        values = square
    return cap_depth_gpu(window_uint8_gpu(values))
DETECTOR_WINDOW_FP64 = True

def detector_image_gpu(ui, u, s):
    off, slope, intercept = (np.float32(v) for v in (s['offset'], s['slope'][0], s['intercept'][0]))
    values = (u + off) * slope + intercept
    flat_ui = ui.flatten().long()
    counts = torch.zeros(65536, dtype=torch.int64, device=ui.device)
    cum = counts.scatter_add_(0, flat_ui, torch.ones(1, dtype=torch.int64, device=ui.device).expand(len(flat_ui))).cumsum(0)

    def order_stats(ks):
        return (torch.searchsorted(cum, ks, right=True).float() + off) * slope + intercept
    low, high = percentiles_gpu(order_stats, ui.numel(), ui.device)
    scale = low.new_full((), 255.0) / (high - low).clamp(min=1e-06)
    if DETECTOR_WINDOW_FP64:
        volume = ((values.double() - low) * scale).clamp(0, 255).round()
    else:
        volume = ((values - low.float()) * scale.float()).clamp(0, 255).round()
    h, w = volume.shape[1:]
    if max(h, w) < 512:
        scale = 512 / max(h, w)
        size = (round(w * scale), round(h * scale))
        volume = resize_gpu(volume.float(), size, 'linear').round().clamp(0, 255)
    return F.interpolate(volume.float().div(255).unsqueeze(1), (DETECTOR_SIZE, DETECTOR_SIZE), mode='bilinear', align_corners=False, antialias=False).transpose(0, 1)

def gpu_inputs(s, need_side, device, sizes=(IMAGE_SIZE,), full_sizes=()):
    ui = s.pop('raw').to(device, non_blocking=True).to(torch.int32) & 65535
    u = ui.float()
    det = s['detector'].to(device) if 'detector' in s else detector_image_gpu(ui, u, s)
    cpu, cpu_full = (s.get('volume', {}), s.get('volume_full', {}))
    volumes, fulls = ({}, {})
    for n in sizes:
        if n in full_sizes:
            full = torch.from_numpy(cpu_full[n]).to(device) if n in cpu_full else classifier_volume_gpu(u, s, n, cap=False)
            fulls[n] = full
            volumes[n] = torch.from_numpy(cpu[n]).to(device) if n in cpu else cap_depth_gpu(full)
        else:
            volumes[n] = torch.from_numpy(cpu[n]).to(device) if n in cpu else classifier_volume_gpu(u, s, n)
    side = side_volume_gpu(u, s) if need_side else None
    return (det, volumes, side, fulls)

def worker_init(_):
    cv2.setNumThreads(1)
    torch.set_num_threads(1)

def identity(item):
    return item

def depth_mask(depths, size):
    valid = torch.arange(size, device=depths.device) < depths[:, None]
    return valid[:, None, :, None, None]

def masked_group_norm(x, norm, mask):
    b, c, *spatial = x.shape
    groups = norm.num_groups
    dtype = torch.promote_types(x.dtype, torch.float32)
    grouped = x.to(dtype).reshape(b, groups, c // groups, *spatial)
    m = mask.to(dtype).reshape(b, 1, 1, *mask.shape[2:])
    dims = tuple(range(2, grouped.dim()))
    broadcast = 1
    for n, k in zip(spatial, mask.shape[2:], strict=True):
        if k == 1:
            broadcast *= n
    count = m.sum(dims, keepdim=True) * (c // groups) * broadcast
    mean = (grouped * m).sum(dims, keepdim=True) / count
    var = ((grouped - mean) * m).square().sum(dims, keepdim=True) / count
    y = ((grouped - mean) * torch.rsqrt(var + norm.eps)).reshape(x.shape)
    affine = (1, c) + (1,) * len(spatial)
    return y * norm.weight.to(dtype).view(affine) + norm.bias.to(dtype).view(affine)

def run_masked(layers, x, mask):
    for layer in layers:
        if isinstance(layer, nn.GroupNorm):
            x = masked_group_norm(x, layer, mask)
        else:
            x = layer(x)
        if isinstance(layer, nn.SiLU):
            x = x * mask
    return x

class Residual(nn.Module):

    def __init__(self, inc, out):
        super().__init__()
        self.skip = nn.Identity()
        self.net = nn.Sequential(nn.Conv3d(inc, out, 3, padding=1, bias=False), nn.GroupNorm(4, out), nn.SiLU(), nn.Conv3d(out, out, 3, padding=1, bias=False), nn.GroupNorm(4, out))

    def forward(self, x, mask):
        return F.silu(run_masked(self.net, x, mask) + self.skip(x)) * mask

class AxisHead(nn.Module):

    def __init__(self, channels, labels, margin):
        super().__init__()
        self.labels, self.margin = (labels, margin)
        self.net = nn.Sequential(nn.Conv1d(channels * 2 + 1, 48, 5, padding=2), nn.GroupNorm(4, 48), nn.SiLU(), nn.Conv1d(48, 48, 3, padding=1), nn.GroupNorm(4, 48), nn.SiLU(), nn.Conv1d(48, labels * 2, 1))

    def forward(self, x, axis, mask):
        reduce = tuple((d for d in (2, 3, 4) if d != axis))
        other = x.shape[3] if axis == 4 else x.shape[4]
        count = mask.sum(2).reshape(-1, 1, 1) * other
        mean = (x * mask).sum(reduce) / count
        peak = x.masked_fill(~mask, float('-inf')).amax(reduce)
        pooled = torch.cat([mean, peak], 1)
        n, m = (pooled.shape[-1], self.margin)
        extent = 1 + 2 * m / max(n - 1, 1)
        grid = torch.linspace(-extent, extent, n + 2 * m, device=x.device, dtype=x.dtype)
        if m:
            pooled = F.pad(pooled, (m, m), mode='replicate')
        pooled = torch.cat([pooled, grid[None, None].expand(x.shape[0], 1, -1)], 1)
        hidden = self.net[:-1](pooled)
        return self.net[-1](hidden).reshape(x.shape[0], self.labels, 2, -1).float()

    def forward_depths(self, x, depths):
        b, _, d = x.shape[:3]
        m = self.margin
        pooled = torch.cat([x.mean((3, 4)), x.amax((3, 4))], 1)
        position = torch.arange(d + 2 * m, device=x.device)
        last = (depths - 1)[:, None]
        source = torch.minimum((position[None] - m).clamp(min=0), last)
        valid = (position[None] < depths[:, None] + 2 * m)[:, None]
        pooled = pooled.gather(2, source[:, None].expand(-1, pooled.shape[1], -1))
        grid = -1 + 2 * (position[None] - m) / last.clamp(min=1)
        features = torch.cat([pooled, grid[:, None].to(pooled.dtype)], 1) * valid
        hidden = run_masked(self.net[:-1], features, valid)
        out = self.net[-1](hidden).reshape(b, self.labels, 2, -1).float()
        return out.masked_fill(~valid[:, None], -10000.0)

def decode_boundaries(logits, z_bins, margin):
    edges = []
    for axis, item in enumerate(logits):
        if axis == 2:
            index = torch.arange(item.shape[-1], device=item.device, dtype=torch.float32)
            positions = (index - margin) / (z_bins[:, None] - 1).clamp(min=1)
            positions = positions.masked_fill(index >= z_bins[:, None] + 2 * margin, 0)
            positions = positions[:, None, None]
        else:
            beyond = margin / (item.shape[-1] - 2 * margin - 1)
            positions = torch.linspace(-beyond, 1 + beyond, item.shape[-1], device=item.device)
        probs = item.softmax(-1)
        edges.append((probs * positions.expand_as(probs)).sum(-1))
    edges = torch.stack(edges, -1)
    lo, hi = (edges.amin(2), edges.amax(2))
    hi = torch.maximum(hi, lo + 0.0001).clamp(max=1)
    lo = torch.minimum(lo, hi - 0.0001).clamp(min=0)
    return torch.cat([lo, hi], -1)

class Detector(nn.Module):

    def __init__(self, fpn_channels=64, labels=12, margin=4, divisor=4):
        super().__init__()
        self.encoder = timm.create_model('convnext_tiny.in12k_ft_in1k', pretrained=False, features_only=True, drop_path_rate=0.1)
        channels = self.encoder.feature_info.channels()
        self.projections = nn.ModuleList([nn.Conv2d(c, fpn_channels, 1) for c in channels])
        self.fusion = nn.ModuleList([Residual(fpn_channels, fpn_channels) for _ in channels])
        self.refine = nn.Sequential(nn.Conv3d(fpn_channels + 3, fpn_channels, 3, padding=1, bias=False), nn.GroupNorm(4, fpn_channels), nn.SiLU(), nn.Dropout3d(0.0))
        self.metadata = nn.Sequential(nn.Linear(10, 48), nn.SiLU(), nn.Linear(48, 2 * fpn_channels))
        self.heads = nn.ModuleList([AxisHead(fpn_channels, labels, margin) for _ in range(3)])
        self.margin, self.divisor = (margin, divisor)
        config = self.encoder.pretrained_cfg
        self.register_buffer('mean', torch.tensor(config['mean']).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(config['std']).view(1, 3, 1, 1))

    def forward(self, image, meta, depths):
        b, _, d, h, w = image.shape
        mask = depth_mask(depths, d)
        slices = (image * mask).transpose(1, 2).reshape(b * d, 1, h, w)
        levels = self.encoder((slices.expand(-1, 3, -1, -1) - self.mean) / self.std)
        if str(getattr(self.encoder, 'output_fmt', 'NCHW')).endswith('NHWC'):
            levels = [f.permute(0, 3, 1, 2) for f in levels]
        features = [projection(level).reshape(b, d, -1, *level.shape[-2:]).transpose(1, 2) * mask for projection, level in zip(self.projections, levels, strict=True)]
        fused = self.fusion[-1](features[-1], mask)
        for index in range(len(features) - 2, -1, -1):
            fused = F.interpolate(fused, features[index].shape[2:], mode='trilinear', align_corners=False)
            fused = self.fusion[index](fused + features[index], mask)
        lattice = (d, h // self.divisor, w // self.divisor)
        if tuple(fused.shape[2:]) != lattice:
            fused = F.interpolate(fused, lattice, mode='trilinear', align_corners=False) * mask
        index = torch.arange(d, device=image.device)
        z = -1 + 2 * index[None] / (depths[:, None] - 1).clamp(min=1)
        z = (z * (index[None] < depths[:, None])).to(fused.dtype)
        ly, lx = lattice[1:]
        y, x = (torch.linspace(-1, 1, n, device=image.device, dtype=fused.dtype) for n in (ly, lx))
        coords = torch.stack([x.view(1, 1, 1, lx).expand(b, d, ly, lx), y.view(1, 1, ly, 1).expand(b, d, ly, lx), z.view(b, d, 1, 1).expand(b, d, ly, lx)], 1)
        fused = run_masked(self.refine, torch.cat([fused, coords], 1) * mask, mask)
        gamma, beta = self.metadata(meta).chunk(2, 1)
        fused = (fused * (1 + 0.1 * gamma[:, :, None, None, None]) + 0.1 * beta[:, :, None, None, None]) * mask
        logits = [self.heads[0](fused, 4, mask), self.heads[1](fused, 3, mask), self.heads[2].forward_depths(fused, depths)]
        return decode_boundaries(logits, depths, self.margin)

def detector_meta(s, side):
    plane = s['plane'].lower()
    extent = [s['cols'] * s['ps_col'] / 200, s['rows'] * s['ps_row'] / 200, s['n_slices'] * s['slice_spacing'] / 200]
    return [float(plane == p) for p in ('sagittal', 'coronal', 'axial')] + [float(side == v) for v in ('left', 'right')] + ([1.0, 0.0] if s['fat'] else [0.0, 1.0]) + extent

def detect(model, image, s, side, device, amp):
    image = image[None].to(device)
    d = image.shape[2]
    size = next((b for b in PAD_DEPTH if b >= d), -(-d // 64) * 64)
    image = F.pad(image, (0, 0, 0, 0, 0, size - d))
    meta = torch.tensor([detector_meta(s, side)], dtype=torch.float32, device=device)
    depths = torch.tensor([d], device=device)
    with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
        boxes = model(image, meta, depths).float()
        sagittal = s['plane'] == 'Sagittal'
        if sagittal:
            index = torch.arange(size, device=device)
            flipped = image[:, :, torch.where(index < d, d - 1 - index, index)]
        else:
            flipped = image.flip(-1)
        flipped_meta = meta.clone()
        flipped_meta[:, [3, 4]] = meta[:, [4, 3]]
        mirrored = model(flipped, flipped_meta, depths).float()
    axis = 2 if sagittal else 0
    mirrored = mirrored.clone()
    mirrored[:, :, axis], mirrored[:, :, axis + 3] = (1 - mirrored[:, :, axis + 3], 1 - mirrored[:, :, axis])
    return ((boxes + mirrored) / 2)[0].cpu().numpy()

class SideClassifier(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model('resnet18d.ra2_in1k', pretrained=False, num_classes=0, drop_path_rate=0.1, in_chans=3)
        config = self.encoder.pretrained_cfg
        self.register_buffer('mean', (torch.tensor(config['mean']) * 255).view(1, 3, 1, 1))
        self.register_buffer('std', (torch.tensor(config['std']) * 255).view(1, 3, 1, 1))
        self.head = nn.Linear(self.encoder.num_features, len(LABELS))

    def forward(self, image, slice_series, series_depth):
        x = image.unsqueeze(1).float().expand(-1, 3, -1, -1)
        x = ((x - self.mean) / self.std).contiguous(memory_format=torch.channels_last)
        features = self.encoder(x)
        pooled = features.new_zeros(series_depth.shape[0] + 1, features.shape[1])
        pooled = pooled.index_add(0, slice_series, features)[:-1]
        return self.head(pooled / series_depth.unsqueeze(1).to(pooled.dtype))

class BoxPoolClassifier(nn.Module):

    def __init__(self, backbone, feature_map='forward_features', encoder_args=None, subsample=4, attention=False, grid_offset=0.0, sources=1, source_hidden=None, slot_channels=False):
        super().__init__()
        self.feature_map, self.subsample = (feature_map, subsample)
        self.compiled = None
        self.grid_offset = float(grid_offset)
        self.encoder = timm.create_model(backbone, pretrained=False, num_classes=0, drop_path_rate=0.1, **encoder_args or {})
        self.slot_channels = bool(slot_channels)
        if self.slot_channels:
            name, conv = next(((n, m) for n, m in self.encoder.named_modules() if isinstance(m, nn.Conv2d)))
            if conv.in_channels != 3:
                raise ValueError(f'slot_channels: the first convolution {name} takes {conv.in_channels} channels')
            wider = nn.Conv2d(len(SLOTS), conv.out_channels, conv.kernel_size, conv.stride, conv.padding, conv.dilation, conv.groups, conv.bias is not None)
            parent, _, attribute = name.rpartition('.')
            setattr(self.encoder.get_submodule(parent) if parent else self.encoder, attribute, wider)
        config = self.encoder.pretrained_cfg
        mean, std = (torch.tensor(config['mean']) * 255, torch.tensor(config['std']) * 255)
        if self.slot_channels:
            mean, std = (mean[torch.arange(len(SLOTS)) % 3], std[torch.arange(len(SLOTS)) % 3])
        self.register_buffer('mean', mean.view(1, -1, 1, 1))
        self.register_buffer('std', std.view(1, -1, 1, 1))
        width, self.sources = (self.encoder.num_features, int(sources))
        self.source_hidden = nn.ParameterList()
        if source_hidden:
            self.source_hidden.extend([nn.Parameter(torch.zeros(self.sources, int(source_hidden), width)), nn.Parameter(torch.zeros(self.sources, int(source_hidden)))])
            width = int(source_hidden)
        self.head = nn.Linear(width, len(LABELS) * self.sources)
        self.attention = nn.Conv2d(self.encoder.num_features, len(LABELS), 1) if attention else None

    def box_weights(self, slice_series, series_depth, series_boxes, size, theta=None, image_size=None):
        batch, (h, w), k = (series_depth.shape[0], size, self.subsample)
        valid = slice_series < batch
        series = slice_series.clamp(max=batch - 1)
        start = torch.cumsum(series_depth, 0) - series_depth
        index = torch.arange(len(slice_series), device=slice_series.device) - start[series]
        z = (index.float() + 0.5) / series_depth[series].float()
        if theta is None:
            theta = torch.eye(2, 3, device=series_boxes.device).expand(batch, 2, 3)
        theta = theta.float()
        if self.grid_offset:
            d = theta.new_full((2, 1), 2 * self.grid_offset / image_size)
            theta = torch.cat([theta[:, :, :2], theta[:, :, 2:] + theta[:, :, :2] @ d], 2)
        box = series_boxes.float()[series][..., None, None]
        grid = F.affine_grid(theta[series], (len(series), 1, h * k, w * k), False)
        gx, gy = (grid[..., 0].unsqueeze(1), grid[..., 1].unsqueeze(1))
        in_z = (z[:, None] >= box[:, :, 2, 0, 0]) & (z[:, None] <= box[:, :, 5, 0, 0])
        in_z = in_z[..., None, None]
        inside = (gx >= box[:, :, 0]) & (gx <= box[:, :, 3])
        inside &= (gy >= box[:, :, 1]) & (gy <= box[:, :, 4])
        inside &= in_z & valid[:, None, None, None]
        return F.avg_pool2d(inside.float(), k)

    def feature_maps(self, x):
        if self.compiled is not None:
            full = len(x) // PAD_MULTIPLE * PAD_MULTIPLE
            chunks = [c for c in (*x[:full].split(PAD_MULTIPLE), *x[full:].split(PAD_TAIL)) if len(c)]
            return torch.cat([self.compiled(chunk) for chunk in chunks])
        return self.encoder_maps(x)

    def encoder_maps(self, x):
        if self.feature_map == 'intermediates':
            return self.encoder.forward_intermediates(x, indices=1, norm=True, output_fmt='NCHW', intermediates_only=True)[-1]
        return self.encoder.forward_features(x)

    def forward(self, image, slice_series, series_depth, series_boxes, theta=None, slots=None):
        x = slots.float() if self.slot_channels else image.unsqueeze(1).float().expand(-1, 3, -1, -1)
        x = ((x - self.mean) / self.std).contiguous(memory_format=torch.channels_last)
        maps = self.feature_maps(x).float()
        n, c, h, w = maps.shape
        batch, labels = (series_depth.shape[0], len(LABELS))
        weight = self.box_weights(slice_series, series_depth, series_boxes, (h, w), theta, image.shape[-1])
        if self.attention is not None:
            score = self.attention(maps).float().masked_fill(weight <= 0, float('-inf'))
            index = slice_series[:, None].expand(-1, labels)
            top = maps.new_full((batch + 1, labels), float('-inf'))
            top = top.scatter_reduce(0, index, score.amax((2, 3)), 'amax')
            top = torch.where(torch.isfinite(top), top, torch.zeros_like(top))
            weight = weight * torch.exp(score - top[slice_series][..., None, None])
        per_slice = torch.einsum('nkhw,nchw->nkc', weight, maps).float()
        sums = maps.new_zeros(batch + 1, labels, c).index_add(0, slice_series, per_slice)[:-1]
        totals = maps.new_zeros(batch + 1, labels).index_add(0, slice_series, weight.sum((2, 3)))
        totals = totals[:-1]
        means = maps.new_zeros(batch + 1, c).index_add(0, slice_series, maps.mean((2, 3)))[:-1]
        means = means / series_depth.unsqueeze(1).float()
        pooled = torch.where(totals[..., None] > 0, sums / totals.clamp(min=1e-06)[..., None], means[:, None].expand(-1, labels, -1))
        weight = self.head.weight.float().view(self.sources, labels, -1)[:1]
        bias = self.head.bias.float().view(self.sources, labels)[:1]
        if len(self.source_hidden):
            w, b = (self.source_hidden[0][:1].float(), self.source_hidden[1][:1].float())
            hidden = F.gelu(torch.einsum('bkc,shc->bskh', pooled.float(), w) + b[:, None])
            return (torch.einsum('bskh,skh->bsk', hidden, weight) + bias)[:, 0]
        return (torch.einsum('bkc,skc->bsk', pooled, weight) + bias)[:, 0]
TRAINING_ONLY = ('report_head.', 'lesion_head.', 'findings_head.', 'joint_head.')

class WeightReader:

    def __init__(self, paths, shard, num_shards, cache):
        self.paths = list(dict.fromkeys((Path(p) for p in paths)))
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        mine = self.paths[shard::num_shards]
        self.order = mine + [p for p in reversed(self.paths) if p not in mine]
        self.data, self.lock = ({}, threading.Lock())
        threading.Thread(target=self._read, daemon=True).start()

    def cached(self, path):
        import hashlib
        stat = path.stat()
        key = hashlib.sha1(f'{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}'.encode()).hexdigest()[:16]
        return self.cache / f'{key}_{path.name}'

    def _read(self):
        for p in self.order:
            if self.cached(p).exists() or not p.exists():
                continue
            with open(p, 'rb') as f:
                data = f.read()
            with self.lock:
                self.data[p] = data
            tmp = self.cached(p).with_name(f'{self.cached(p).name}.{os.getpid()}.tmp')
            tmp.write_bytes(data)
            os.replace(tmp, self.cached(p))

    def get(self, path):
        path = Path(path)
        if path not in self.paths or not path.exists():
            return path
        while True:
            with self.lock:
                data = self.data.pop(path, None)
            if data is not None:
                return io.BytesIO(data)
            if self.cached(path).exists():
                return io.BytesIO(self.cached(path).read_bytes())
            time.sleep(0.05)
READER = None

def load(model, path, device, channels_last=False):
    state = torch.load(READER.get(path) if READER else path, map_location='cpu', weights_only=False)
    state = state.get('ema', state.get('model', state))
    state = {k.removeprefix('_orig_mod.'): v for k, v in state.items()}
    state = {k: v for k, v in state.items() if not k.startswith(TRAINING_ONLY)}
    keys = model.load_state_dict(state, strict=False)
    # Our training notebook replaced the encoder's unused classifier head (its norm) with Identity, so those keys
    # are absent from our checkpoint; the classifier never uses them (it reads forward_features).
    missing = {k for k in keys.missing_keys if not k.startswith('encoder.head.')}
    if keys.unexpected_keys or missing - {'mean', 'std'}:
        raise RuntimeError(f'{path}: missing {keys.missing_keys}, unexpected {keys.unexpected_keys}')
    model = model.to(device).eval()
    return model.to(memory_format=torch.channels_last) if channels_last else model

def pack(volumes, multiple=PAD_MULTIPLE):
    image = torch.cat([torch.as_tensor(v) for v in volumes])
    depths = torch.tensor([len(v) for v in volumes])
    slice_series = torch.repeat_interleave(torch.arange(len(volumes)), depths)
    pad = -len(image) % multiple
    image = torch.cat([image, image[torch.arange(pad) % len(image)]])
    slice_series = torch.cat([slice_series, torch.full((pad,), len(volumes))])
    return (image, slice_series, depths)

def union_crop_theta(series_boxes, labels, scale):
    boxes = series_boxes.index_select(1, to_device(labels, series_boxes.device, torch.int64)).float()
    x0, y0 = (boxes[..., 0].amin(1), boxes[..., 1].amin(1))
    x1, y1 = (boxes[..., 3].amax(1), boxes[..., 4].amax(1))
    limit = 1.0 - scale
    cx = ((x0 + x1) / 2).clamp(-limit, limit)
    cy = ((y0 + y1) / 2).clamp(-limit, limit)
    zero = torch.zeros_like(cx)
    s = torch.full_like(cx, scale)
    return torch.stack([torch.stack([s, zero, cx], 1), torch.stack([zero, s, cy], 1)], 1)

def affine_grid_2d(theta, h, w):
    xs = (torch.arange(w, device=theta.device, dtype=theta.dtype) * 2 + 1) / w - 1
    ys = (torch.arange(h, device=theta.device, dtype=theta.dtype) * 2 + 1) / h - 1
    t = theta[:, :, :, None, None]
    gx = t[:, 0, 0] * xs + t[:, 0, 1] * ys[:, None] + t[:, 0, 2]
    gy = t[:, 1, 0] * xs + t[:, 1, 1] * ys[:, None] + t[:, 1, 2]
    return torch.stack([gx, gy], dim=-1)

def crop_series(image, slice_series, crop_theta, size):
    identity = torch.eye(2, 3, device=image.device)[None]
    theta = torch.cat([crop_theta.float(), identity])[slice_series]
    grid = affine_grid_2d(theta, size, size)
    return F.grid_sample(image.float().unsqueeze(1), grid, mode='bilinear', padding_mode='zeros', align_corners=False).squeeze(1)
SLOTS = (('Sagittal', 1), ('Sagittal', 0), ('Coronal', 1), ('Coronal', 0), ('Axial', 1), ('Axial', 0))

def slot_of(s):
    return SLOTS.index((s['plane'], int(s['fluid'])))

def slot_pairs(series):
    sources = {}
    for j, b in enumerate(series):
        k = slot_of(b)
        if b['n_slices'] >= 2 and (k not in sources or b['slice_spacing'] < series[sources[k]]['slice_spacing']):
            sources[k] = j
    return [[(k, j) for k, j in sorted(sources.items()) if k != slot_of(a)] for a in series]

def patient_frame(s, size):
    if s.get('ipp') is None or s.get('orientation') is None or len(s['ipp']) != s['n_slices']:
        return None
    o = np.asarray(s['orientation'], np.float64)
    row_dir, col_dir = (o[:3], o[3:])
    height = max(1.0, float(np.round(s['rows'] * (size * s['ps_row'] / FOV_MM))))
    width = max(1.0, float(np.round(s['cols'] * (size * s['ps_col'] / FOV_MM))))

    def offset(n):
        return -((n - size) // 2) if n >= size else (size - n) // 2
    top, left, half = (offset(height), offset(width), size / 2)
    a_c, a_r = (half * s['cols'] / width, half * s['rows'] / height)
    b_c = (half - left) * s['cols'] / width - 0.5
    b_r = (half - top) * s['rows'] / height - 0.5
    sc, sr = (float(s['ps_col']), float(s['ps_row']))
    return (np.asarray(s['ipp'], np.float64), a_c * sc * row_dir, a_r * sr * col_dir, b_c * sc * row_dir + b_r * sr * col_dir)

def slice_affines(frame, positions):
    ipp, x_axis, y_axis, offset = frame
    lower = np.floor(positions).astype(int)
    upper = np.minimum(lower + 1, len(ipp) - 1)
    weight = (positions - lower)[:, None]
    origin = ipp[lower] * (1 - weight) + ipp[upper] * weight + offset
    columns = [np.broadcast_to(x_axis, origin.shape), np.broadcast_to(y_axis, origin.shape), origin]
    return np.ascontiguousarray(np.stack(columns, 2), np.float64)

def slot_source_geometry(frame):
    ipp, x_axis, y_axis, offset = frame
    if len(ipp) < 2:
        return None
    normal = np.cross(x_axis, y_axis)
    normal /= np.linalg.norm(normal)
    along = ipp @ normal
    ordered = np.sort(along)
    low = ordered[0] - (ordered[1] - ordered[0]) / 2
    high = ordered[-1] + (ordered[-1] - ordered[-2]) / 2
    track = np.stack([along, ipp @ x_axis / np.linalg.norm(x_axis), ipp @ y_axis / np.linalg.norm(y_axis)], 1)
    index = np.arange(len(ipp))
    slope, intercept = np.polyfit(index, track, 1)
    if abs(slope[0]) <= 0.001 or np.abs(track - (np.outer(index, slope) + intercept)).max() > 0.001:
        return None
    return np.concatenate([normal, x_axis, y_axis, offset, slope, intercept, [low, high, len(ipp)]]).astype(np.float64)

def sample_slot_sources(volumes, start, geometry, affine, theta, sign, size, chunk=128):
    device, rows = (volumes.device, len(affine))
    out = torch.zeros(rows, size, size, dtype=torch.float32, device=device)
    if rows == 0:
        return out
    a, t, geo = (affine.double(), theta.double(), geometry.double())
    x_col, y_col = (a[:, :, 0] * sign.double()[:, None], a[:, :, 1])
    du = t[:, 0, 0, None] * x_col + t[:, 1, 0, None] * y_col
    dv = t[:, 0, 1, None] * x_col + t[:, 1, 1, None] * y_col
    normal, x_b, y_b, offset = (geo[:, 0:3], geo[:, 3:6], geo[:, 6:9], geo[:, 9:12])
    slope, intercept, low, high, depth = (geo[:, 12:15], geo[:, 15:18], geo[:, 18], geo[:, 19], geo[:, 20])
    base = a[:, :, 2] + t[:, 0, 2, None] * x_col + t[:, 1, 2, None] * y_col - offset

    def dot(p, q):
        return (p * q).sum(-1)

    def affine_in_uv(c0, cu, cv):
        return torch.stack([c0, cu, cv], 1)
    h = affine_in_uv(dot(base, normal), dot(du, normal), dot(dv, normal))
    z = affine_in_uv((h[:, 0] - intercept[:, 0]) / slope[:, 0], h[:, 1] / slope[:, 0], h[:, 2] / slope[:, 0])
    planar, moves = ([], [])
    for axis, k in ((x_b, 1), (y_b, 2)):
        norm2 = dot(axis, axis)
        length = norm2.sqrt()
        planar.append(affine_in_uv((dot(base, axis) - length * (intercept[:, k] + z[:, 0] * slope[:, k])) / norm2, (dot(du, axis) - length * z[:, 1] * slope[:, k]) / norm2, (dot(dv, axis) - length * z[:, 2] * slope[:, k]) / norm2))
        moves.append(slope[:, k] / length)
    h_low = h.clone()
    h_low[:, 0] -= low
    h_high = h.clone()
    h_high[:, 0] -= high
    grid1 = (torch.arange(size, dtype=torch.float32, device=device) * 2 + 1) / size - 1
    total = float(len(volumes))
    source = volumes.float()[None, None]

    def plane(c, r):
        c = c[r].float()
        return c[:, 0, None, None] + c[:, 1, None, None] * grid1[None, None, :] + c[:, 2, None, None] * grid1[None, :, None]
    for s in range(0, rows, chunk):
        r = slice(s, s + chunk)
        zs = plane(z, r)
        last = (depth[r].float() - 1)[:, None, None]
        clamped = torch.minimum(zs.clamp(min=0.0), last)
        excess = zs - clamped
        xs = plane(planar[0], r) + excess * moves[0][r].float()[:, None, None]
        ys = plane(planar[1], r) + excess * moves[1][r].float()[:, None, None]
        zn = (2 * (start[r].float()[:, None, None] + clamped) + 1) / total - 1
        grid = torch.stack([xs, ys, zn], -1)
        values = F.grid_sample(source, grid[None], mode='bilinear', padding_mode='zeros', align_corners=False)[0, 0]
        inside = (plane(h_low, r) >= 0) & (plane(h_high, r) <= 0)
        out[r] = values.masked_fill_(~inside, 0.0)
    return out

def sample_slot_general(volume, frame, affine, theta, sign, size):
    ipp, x_b, y_b, off = frame
    d64 = dict(dtype=torch.float64, device=volume.device)
    normal = np.cross(x_b, y_b)
    normal /= np.linalg.norm(normal)
    along = ipp @ normal
    flip = along[-1] < along[0]
    ordered = along[::-1] if flip else along
    low = ordered[0] - (ordered[1] - ordered[0]) / 2
    high = ordered[-1] + (ordered[-1] - ordered[-2]) / 2
    positions = torch.tensor(ordered.copy(), **d64)
    slopes = 1.0 / (positions[1:] - positions[:-1])
    depth = len(ipp)
    g = (torch.arange(size, **d64) * 2 + 1) / size - 1
    a, t = (affine.double(), theta.double())
    x_col, y_col = (a[:, :, 0] * sign.double()[:, None], a[:, :, 1])
    du = t[:, 0, 0, None] * x_col + t[:, 1, 0, None] * y_col
    dv = t[:, 0, 1, None] * x_col + t[:, 1, 1, None] * y_col
    base = a[:, :, 2] + t[:, 0, 2, None] * x_col + t[:, 1, 2, None] * y_col - torch.tensor(off, **d64)
    n_t = torch.tensor(normal, **d64)
    axes = [torch.tensor(v, **d64) for v in (x_b, y_b)]
    origins = [torch.tensor(ipp @ v, **d64) for v in (x_b, y_b)]
    source = volume.float()[None, None]
    out = torch.zeros(len(a), size, size, dtype=torch.float32, device=volume.device)
    for r in range(len(a)):
        points = base[r] + g[None, :, None] * du[r] + g[:, None, None] * dv[r]
        h = points @ n_t
        inside = (h >= low) & (h <= high)
        j = (torch.searchsorted(positions, h.contiguous(), right=True) - 1).clamp(0, depth - 2)
        z = slopes[j] * (h - positions[j]) + j
        z = torch.where(h < positions[0], 0.0, torch.where(h >= positions[-1], depth - 1.0, z))
        if flip:
            z = depth - 1 - z
        lower = z.floor().long()
        upper = (lower + 1).clamp(max=depth - 1)
        w = z - lower
        xy = [(points @ ax - (q[lower] * (1 - w) + q[upper] * w)) / float(ax @ ax) for ax, q in zip(axes, origins, strict=True)]
        grid = torch.stack([*xy, (2 * z + 1) / depth - 1], -1).float()
        values = F.grid_sample(source, grid[None, None], mode='bilinear', padding_mode='zeros', align_corners=False)
        out[r] = values[0, 0, 0].masked_fill_(~inside, 0.0)
    return out

def slot_inputs(own, slice_series, depths, theta, info, start, stop):
    n, pixels = (len(own), own.shape[-1])
    out = own.new_zeros(n, len(SLOTS), pixels, pixels)
    own_slot = torch.repeat_interleave(torch.tensor(info['own'][start:stop]), depths.cpu())
    count = len(own_slot)
    own_slot = torch.cat([own_slot, own_slot[torch.arange(n - count) % count]]).to(own.device)
    out[torch.arange(n, device=own.device), own_slot] = own
    if theta is None:
        theta = torch.eye(2, 3, device=own.device).expand(stop - start, 2, 3)
    first = 0
    for b, i in enumerate(range(start, stop)):
        d = int(depths[b])
        rows = slice(first, first + d)
        first += d
        frame_i = info['frames'][i]
        if frame_i is None:
            continue
        affine = torch.from_numpy(slice_affines(frame_i, info['positions'][i])).to(own.device)
        th = theta[b].float().expand(d, 2, 3)
        sign = torch.full((d,), -1.0 if info['mirror'][i] else 1.0, device=own.device)
        for k, j in info['pairs'][i]:
            try:
                if info['frames'][j] is None:
                    continue
                volume = info['full'][j]
                geometry = info['geometry'][j]
                if geometry is not None:
                    values = sample_slot_sources(volume, torch.zeros(d, dtype=torch.long, device=own.device), torch.from_numpy(geometry).to(own.device).expand(d, -1), affine, th, sign, pixels)
                else:
                    values = sample_slot_general(volume, info['frames'][j], affine, th, sign, pixels)
                out[rows, k] = values
            except Exception:
                print(f"slot {SLOTS[k]} of series {info['uids'][i]} from {info['uids'][j]} left zeros: {traceback.format_exc(limit=2)}", flush=True)
    return out
PROFILE = None
RUN_WEIGHTS = None

@contextlib.contextmanager
def stage(name):
    if PROFILE is None:
        yield
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    PROFILE[name] = PROFILE.get(name, 0.0) + time.perf_counter() - start

def predict_study(item, models, device, amp, max_slices=1024):
    detector, sides, runs = models
    sizes = sorted({size for size, _, _, _ in runs})
    slot_sizes = sorted({size for size, _, _, slots in runs if slots})
    series = item['series']
    side, source, p_right = (item['side'], item['source'], float('nan'))
    with stage('gpu preprocessing'):
        inputs = [gpu_inputs(s, side is None and s['plane'] in ('Coronal', 'Axial'), device, sizes, slot_sizes) for s in series]
    if side is None:
        views = [x[2] for x in inputs if x[2] is not None]
        if views:
            with stage('side classifier'):
                image, slice_series, depths = pack(views)
                image, slice_series, depths = (image.to(device), slice_series.to(device), depths.to(device))
                with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                    p = [torch.sigmoid(m(image, slice_series, depths)[:, 0].float()) for m in sides]
                p_right = float(torch.stack(p).mean(0).mean())
            side, source = ('right' if p_right >= 0.5 else 'left', 'classifier')
        else:
            side, source = ('right', 'default')
    volumes, boxes, rows = ({n: [] for n in sizes}, {n: [] for n in sizes}, [])
    for s, (det, series_volumes, _, _) in zip(series, inputs, strict=True):
        with stage('detector'):
            raw = detect(detector, det, s, side, device, amp)
        mirror = side == 'left' and s['plane'] in ('Coronal', 'Axial')
        for n in sizes:
            volume, b = (series_volumes[n], image_boxes(raw, s, n))
            if mirror:
                volume = volume.flip(2)
                b[:, [0, 3]] = -b[:, [3, 0]]
            volumes[n].append(volume)
            boxes[n].append(b)
        rows.append({'series': s['series'], 'plane': s['plane'], 'mirror': mirror, 'boxes': raw})
    slot_info = {}
    if slot_sizes:
        pairs = slot_pairs(series)
        native = [np.linspace(0, s['n_slices'] - 1, MAX_DEPTH) if s['n_slices'] > MAX_DEPTH else np.arange(s['n_slices'], dtype=np.float64) for s in series]
        used = {j for p in pairs for _, j in p}
        for n in slot_sizes:
            frames = [patient_frame(s, n) for s in series]
            slot_info[n] = dict(pairs=pairs, own=[slot_of(s) for s in series], mirror=[r['mirror'] for r in rows], frames=frames, positions=native, uids=[s['series'] for s in series], geometry={j: slot_source_geometry(frames[j]) for j in used if frames[j] is not None}, full={j: inputs[j][3][n] for j in used})
    run_probs = []
    for n in sizes:
        group = [(crop, folds, slots) for size, crop, folds, slots in runs if size == n]
        budget = max(1, int(max_slices * (IMAGE_SIZE / n) ** 2))
        chunks, start = ([], 0)
        while start < len(volumes[n]):
            stop, total = (start + 1, len(volumes[n][start]))
            while stop < len(volumes[n]) and total + len(volumes[n][stop]) <= budget:
                total += len(volumes[n][stop])
                stop += 1
            chunks.append((start, stop))
            start = stop
        per_run = [[] for _ in group]
        compiled = all((m.compiled is not None for _, folds, _ in group for m in folds))
        for start, stop in chunks:
            with stage('pack'):
                image, slice_series, depths = pack(volumes[n][start:stop], PAD_TAIL if compiled else PAD_MULTIPLE)
                image, slice_series = (image.to(device), slice_series.to(device))
                depths = depths.to(device)
                series_boxes = torch.from_numpy(np.stack(boxes[n][start:stop])).to(device)
            for g, (crop, folds, slots) in enumerate(group):
                inputs, theta, slot_x = (image, None, None)
                if crop is not None:
                    with stage('input crop'):
                        labels = [LABELS.index(name) for name in crop['labels']]
                        theta = union_crop_theta(series_boxes, labels, crop['size_mm'] / crop['image_mm'])
                        inputs = crop_series(image, slice_series, theta, int(crop['pixels']))
                if slots:
                    with stage('slots'):
                        slot_x = slot_inputs(inputs.float(), slice_series, depths, theta, slot_info[n], start, stop)
                with stage(f'classifier x{len(folds)}'), torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                    p = [torch.sigmoid(m(inputs, slice_series, depths, series_boxes, theta, slot_x).float()) for m in folds]
                per_run[g].append(torch.stack(p).mean(0).cpu().numpy())
        run_probs += [np.concatenate(p) for p in per_run]
    per_series = np.average(run_probs, 0, weights=RUN_WEIGHTS)
    for row, p in zip(rows, per_series, strict=True):
        row.update(side=side, source=source, p_right=p_right, probs=p)
    return (per_series.mean(0), rows)

def run(args):
    if args.run is None:
        raise SystemExit(f'--run: one or more of {sorted(RUNS)}')
    if args.weights is not None and len(args.weights) != len(args.run):
        raise SystemExit('--weights: one directory per --run')
    if args.run_weights is not None and len(args.run_weights) != len(args.run):
        raise SystemExit('--run-weights: one weight per --run')
    global RUN_WEIGHTS, READER
    RUN_WEIGHTS = args.run_weights
    if args.stage != 'read' and args.weight_cache != 'none':
        detector, sides, runs = model_paths(args)
        READER = WeightReader([detector, *sides, *sum(runs, [])], args.shard, args.num_shards, args.weight_cache)
    root = Path(args.competition or kaggle_input(COMPETITION))
    series_table = pd.read_csv(root / f'{args.split}_series.csv')
    studies = study_list(args, root)[args.shard::args.num_shards]
    (Path(args.out) / f'predictions_shard{args.shard}.csv').unlink(missing_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    amp = device.type == 'cuda'
    loaded = time.time()
    if args.stage == 'read':
        models = None
        sizes = sorted({RUN_SIZES.get(r, IMAGE_SIZE) for r in args.run})
        slot_sizes = sorted({RUN_SIZES.get(r, IMAGE_SIZE) for r in args.run if RUN_OPTIONS.get(r, {}).get('slot_channels')})
    else:
        models = load_models(args, device)
        if args.compile:
            for key in ('cache_size_limit', 'recompile_limit'):
                if hasattr(torch._dynamo.config, key):
                    setattr(torch._dynamo.config, key, 64)
            for _, _, run_models, _ in models[2]:
                for m in run_models:
                    m.compiled = torch.compile(m.encoder_maps, dynamic=False)
        sizes = sorted({size for size, _, _, _ in models[2]})
        slot_sizes = sorted({size for size, _, _, slots in models[2] if slots})
    print(f'shard {args.shard}: models loaded in {time.time() - loaded:.1f} s', flush=True)
    print(f"shard {args.shard}/{args.num_shards}: {len(studies)} {args.split} studies{(' (debug mode)' if args.debug else '')}, runs {' + '.join(args.run)}, stage {args.stage}, device {device}, torch {torch.__version__}, timm {timm.__version__}", flush=True)
    loader = torch.utils.data.DataLoader(StudyDataset(root / f'{args.split}_series', studies, series_table, sizes, slot_sizes, args.io_threads), batch_size=None, collate_fn=identity, pin_memory=device.type == 'cuda', num_workers=args.workers, worker_init_fn=worker_init, prefetch_factor=4 if args.workers else None)
    global PROFILE
    PROFILE = {} if args.profile else None
    predictions, debug, failures, skipped = ([], [], [], 0)
    start = last = reported = time.time()
    wait = 0.0
    with torch.no_grad():
        for i, item in enumerate(loader):
            wait += time.time() - last
            for error in item['errors']:
                print(f'series skipped: {error}', flush=True)
            skipped += len(item['errors'])
            try:
                if models is None:
                    pass
                elif item['series']:
                    if args.trace and i == 10:
                        profiler = torch.profiler.profile(record_shapes=False).__enter__()
                    probs, rows = predict_study(item, models, device, amp, args.max_slices)
                    if args.trace and i == 9 + args.trace:
                        profiler.__exit__(None, None, None)
                        table = profiler.key_averages().table(sort_by='self_device_time_total', row_limit=50)
                        Path(args.out).mkdir(parents=True, exist_ok=True)
                        (Path(args.out) / f'ops_shard{args.shard}.txt').write_text(table)
                        profiler.export_chrome_trace(str(Path(args.out) / f'trace_shard{args.shard}.json'))
                    predictions.append([item['study'], *probs.tolist()])
                    debug += [{'study': item['study'], **r} for r in rows]
                else:
                    failures.append(item['study'])
            except Exception:
                print(f"study failed: {item['study']}: {traceback.format_exc(limit=3)}", flush=True)
                failures.append(item['study'])
            last = time.time()
            if last - reported >= args.log_every or i + 1 == len(studies):
                reported = last
                elapsed = last - start
                eta = elapsed / (i + 1) * (len(studies) - i - 1)
                print(f'[shard {args.shard}] {i + 1}/{len(studies)} studies, {elapsed / 60:.1f} min, {elapsed / (i + 1):.2f} s per study, ETA {eta / 60:.1f} min, waiting for data {wait:.0f} s', flush=True)
    if PROFILE is not None:
        report = {'studies': len(studies), 'wall': time.time() - start, 'waiting for data': wait, **PROFILE}
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / f'profile_shard{args.shard}.json').write_text(json.dumps(report, indent=1))
        print('profile ' + json.dumps({k: round(v, 1) for k, v in report.items()}), flush=True)
    if models is None:
        print(f'shard {args.shard} done (stage read): {len(studies)} studies, {time.time() - start:.0f} s', flush=True)
        return
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(predictions, columns=['StudyInstanceUID', *LABELS]).to_csv(out / f'predictions_shard{args.shard}.csv', index=False)
    if args.series_rows:
        rows = []
        for r in debug:
            boxes, probs = (r.pop('boxes'), r.pop('probs'))
            rows.append({**r, **{f'box_{k}': v for k, v in enumerate(boxes.ravel())}, **dict(zip(LABELS, probs, strict=True))})
        pd.DataFrame(rows).to_csv(out / f'series_shard{args.shard}.csv', index=False)
    print(f'shard {args.shard} done: {len(predictions)} studies, {len(failures)} without a prediction {failures[:5]}, {skipped} series skipped, {time.time() - start:.0f} s', flush=True)

def model_paths(args):
    common = Path(args.common or kaggle_input('knee-models-common'))
    weights = [Path(w) for w in args.weights or [kaggle_input(dataset_name(r)) for r in args.run]]
    runs = [[w / name for name in RUN_MODELS.get(r, [f'fold{k}.pth' for k in range(FOLDS)])[:args.models]] for r, w in zip(args.run, weights, strict=True)]
    return (common / 'detector.pth', [common / f'knee_side_fold{k}.pth' for k in range(FOLDS)], runs)

def load_models(args, device):
    detector, sides, runs = model_paths(args)
    return (load(Detector(), detector, device), [load(SideClassifier(), path, device, True) for path in sides], [(RUN_SIZES.get(r, IMAGE_SIZE), RUN_CROPS.get(r), [load(BoxPoolClassifier(**ARCHS[RUNS[r]], **RUN_OPTIONS.get(r, {})), path, device, True) for path in paths], bool(RUN_OPTIONS.get(r, {}).get('slot_channels'))) for r, paths in zip(args.run, runs, strict=True)])

def study_list(args, root):
    if args.debug:
        studies = sorted(set(pd.read_csv(root / 'train.csv').StudyInstanceUID))
        sample = np.random.default_rng(0).choice(len(studies), args.debug, replace=False)
        return sorted((studies[i] for i in sample))
    return sorted(set(pd.read_csv(args.studies or root / f'{args.split}.csv').StudyInstanceUID))

def merge(args):
    out = Path(args.out)
    parts = [out / f'predictions_shard{k}.csv' for k in range(args.num_shards)]
    missing = [p.name for p in parts if not p.exists()]
    if missing:
        raise SystemExit(f'Shard outputs missing: {missing}')
    predictions = pd.concat([pd.read_csv(p) for p in parts]).set_index('StudyInstanceUID')
    if predictions.index.duplicated().any():
        raise SystemExit('A study predicted by more than one shard')
    root = Path(args.competition or kaggle_input(COMPETITION))
    if args.split == 'test' and (not args.studies):
        template = pd.read_csv(root / 'sample_submission.csv')
    else:
        template = pd.DataFrame({'StudyInstanceUID': study_list(args, root)})
    submission = predictions.reindex(template.StudyInstanceUID)[LABELS]
    missing = submission.isna().any(axis=1)
    submission = submission.fillna(0.5).reset_index()
    submission.to_csv(out / 'submission.csv', index=False)
    print(f"{len(parts)} shards, {len(submission)} studies, {int(missing.sum())} filled with 0.5 -> {out / 'submission.csv'}")

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--competition', help=f'default: {COMPETITION} under /kaggle/input')
    parser.add_argument('--split', default='test', help='<split>_series/, <split>_series.csv')
    parser.add_argument('--studies', help='CSV with StudyInstanceUID: only these studies')
    parser.add_argument('--common', help='shared models (default: knee-models-common)')
    parser.add_argument('--weights', nargs='+', help="each run's model directory (default: from --run)")
    parser.add_argument('--run', nargs='+', choices=sorted(RUNS), help='one or more runs: the mean of their probabilities')
    parser.add_argument('--run-weights', nargs='+', type=float, help="the runs' weights in the mean (default: equal)")
    parser.add_argument('--out', default='/kaggle/working')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=2)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--io-threads', type=int, default=8, help="threads reading a study's files (per loader worker)")
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max-slices', type=int, default=1024, help='slices per classifier pack at 256 px (fewer at a larger size: 256 at 512 px, a whole study as a rule)')
    parser.add_argument('--log-every', type=float, default=30, help='seconds between progress lines')
    parser.add_argument('--series-rows', action='store_true', help='also write per-series rows')
    parser.add_argument('--stage', choices=['full', 'read'], default='full', help='timing: read = the loader only (DICOM reading and CPU preprocessing), no model')
    parser.add_argument('--models', type=int, help="timing: each run's first N models only")
    parser.add_argument('--compile', action='store_true', help="torch.compile each classifier's encoder for two static shapes (chunks of PAD_MULTIPLE and PAD_TAIL slices)")
    parser.add_argument('--weight-cache', default='/tmp/knee_weights', help='local directory the shards share the read weight files through (WeightReader); none: torch.load(path)')
    parser.add_argument('--profile', action='store_true', help='timing: CUDA-synchronized time of each study stage')
    parser.add_argument('--trace', type=int, default=0, help='timing: torch.profiler on N studies')
    parser.add_argument('--debug', type=int, nargs='?', const=1200, help='debug mode: N (default 1200) training studies in place of the test set')
    parser.add_argument('--merge', action='store_true', help='merge the shards into submission.csv')
    args = parser.parse_args()
    if args.debug:
        args.split = 'train'
    merge(args) if args.merge else run(args)
if __name__ == '__main__':
    main()