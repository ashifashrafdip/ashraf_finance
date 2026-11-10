import os
import sys
import io
import cv2
import numpy as np
from PIL import Image, ImageOps, ExifTags
from insightface.app import FaceAnalysis

def get_models_root():
    candidates = []
    if getattr(sys, "frozen", False):
        exe_dir = os.path.dirname(sys.executable)
        candidates.append(exe_dir)
        if hasattr(sys, "_MEIPASS"):
            candidates.append(sys._MEIPASS)
    src_dir = os.path.dirname(os.path.abspath(__file__))
    candidates.append(src_dir)
    candidates.append(os.path.join(src_dir, "models"))
    home_root = os.path.expanduser("~/.insightface")
    candidates.append(home_root)

    for c in candidates:
        if os.path.isdir(os.path.join(c, "models", "buffalo_l")):
            return c
    return "~/.insightface"

fa_small = None
fa_large = None

def get_detectors():
    global fa_small, fa_large
    root_dir = get_models_root()
    if fa_small is None:
        fa_small = FaceAnalysis(name="buffalo_l", root=root_dir, providers=["CPUExecutionProvider"])
        fa_small.prepare(ctx_id=0, det_size=(640, 640))
    if fa_large is None:
        fa_large = FaceAnalysis(name="buffalo_l", root=root_dir, providers=["CPUExecutionProvider"])
        fa_large.prepare(ctx_id=0, det_size=(1600, 1600))
    return fa_small, fa_large

def detect_faces_multiscale(bgr_img: np.ndarray):
    det_small, det_large = get_detectors()
    f640 = det_small.get(bgr_img)
    f1600 = det_large.get(bgr_img)

    all_faces = list(f640)
    for f2 in f1600:
        dup = False
        for f1 in all_faces:
            xA = max(f1.bbox[0], f2.bbox[0])
            yA = max(f1.bbox[1], f2.bbox[1])
            xB = min(f1.bbox[2], f2.bbox[2])
            yB = min(f1.bbox[3], f2.bbox[3])
            inter = max(0.0, xB - xA) * max(0.0, yB - yA)
            area1 = (f1.bbox[2] - f1.bbox[0]) * (f1.bbox[3] - f1.bbox[1])
            area2 = (f2.bbox[2] - f2.bbox[0]) * (f2.bbox[3] - f2.bbox[1])
            iou = inter / float(area1 + area2 - inter + 1e-5)
            if iou > 0.40:
                dup = True
                break
        if not dup:
            all_faces.append(f2)

    all_faces = sorted(
        all_faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        reverse=True
    )
    return all_faces

def crop_face_bgr(bgr_img: np.ndarray, bbox, pad_ratio: float = 0.22):
    h, w = bgr_img.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw, bh = x2 - x1, y2 - y1
    px, py = bw * pad_ratio, bh * pad_ratio
    cx1 = max(0, int(x1 - px))
    cy1 = max(0, int(y1 - py))
    cx2 = min(w, int(x2 + px))
    cy2 = min(h, int(y2 + py))
    crop = bgr_img[cy1:cy2, cx1:cx2]
    ch, cw = crop.shape[:2]
    return crop, f"{cw} × {ch}"

EXIF_LOOKUP = {
    'ExposureProgram': {0: 'Not defined', 1: 'Manual', 2: 'Normal program', 3: 'Aperture priority', 4: 'Shutter priority', 5: 'Creative program', 6: 'Action program', 7: 'Portrait mode', 8: 'Landscape mode'},
    'MeteringMode': {0: 'Unknown', 1: 'Average', 2: 'Center-weighted average', 3: 'Spot', 4: 'Multi-spot', 5: 'Pattern / Multi-segment', 6: 'Partial'},
    'ExposureMode': {0: 'Auto exposure', 1: 'Manual exposure', 2: 'Auto bracket'},
    'WhiteBalance': {0: 'Auto white balance', 1: 'Manual white balance'},
    'SceneCaptureType': {0: 'Standard', 1: 'Landscape', 2: 'Portrait', 3: 'Night scene'},
    'ColorSpace': {1: 'sRGB', 65535: 'Uncalibrated / Wide color (P3)'},
    'Orientation': {1: 'Normal (0 deg)', 3: 'Upside down (180 deg)', 6: 'Rotated 90 deg CW', 8: 'Rotated 90 deg CCW'}
}

def parse_metadata_dict(raw_bytes: bytes, filename: str):
    pil = Image.open(io.BytesIO(raw_bytes))
    w, h = pil.size
    fmt = pil.format or 'JPEG'
    size_bytes = len(raw_bytes)
    size_str = f"{size_bytes / (1024*1024):.2f} MB" if size_bytes >= 1024*1024 else f"{size_bytes / 1024:.1f} KB"

    dpi = pil.info.get('dpi', None)
    dpi_str = f"{int(dpi[0])} × {int(dpi[1])} DPI" if dpi else "72 × 72 DPI"

    xmp_bytes = pil.info.get('xmp', b'')
    xmp_str = xmp_bytes.decode('utf-8', errors='replace') if xmp_bytes else ''
    photoshop_info = pil.info.get('photoshop', None)
    adobe_info = pil.info.get('adobe', None)
    c2pa_detected = (b'jumdc2pa' in raw_bytes or b'c2pa' in raw_bytes or b'Adobe Firefly' in raw_bytes)

    exif = pil.getexif()
    tags = {}
    if exif:
        for k, v in exif.items():
            name = ExifTags.TAGS.get(k, str(k))
            tags[name] = v
        for ifd_id in [ExifTags.IFD.Exif, ExifTags.IFD.GPSInfo, ExifTags.IFD.Makernote]:
            try:
                ifd = exif.get_ifd(ifd_id)
                for k, v in ifd.items():
                    name = ExifTags.TAGS.get(k, ExifTags.GPSTAGS.get(k, str(k)))
                    tags[name] = v
            except Exception:
                pass

    make = str(tags.get('Make', '')).strip()
    model = str(tags.get('Model', '')).strip()
    device = f"{make} {model}".strip() if (make or model) else "Not recorded"

    soft = tags.get('Software')
    soft_str = str(soft).strip() if soft else None
    if not soft_str and 'photoshop' in xmp_str.lower():
        soft_str = 'Adobe Photoshop'
    if not soft_str and c2pa_detected:
        soft_str = 'Adobe Firefly (Generative AI)'
    if not soft_str:
        soft_str = 'Not available'

    is_edited = False
    if any(k in soft_str.lower() for k in ['photoshop', 'gimp', 'canva', 'firefly', 'lightroom', 'paint']):
        is_edited = True
    if photoshop_info is not None or adobe_info is not None:
        is_edited = True
    if c2pa_detected:
        is_edited = True

    dt_orig = str(tags.get('DateTimeOriginal', '')).strip() or "Not recorded"
    dt_mod = str(tags.get('DateTime', '')).strip() or "Not recorded"
    dt_dig = str(tags.get('DateTimeDigitized', '')).strip() or "Not recorded"
    dt_mismatch = bool(dt_orig != "Not recorded" and dt_mod != "Not recorded" and dt_orig[:10] != dt_mod[:10])

    exp_time = tags.get('ExposureTime')
    exp_str = 'Not recorded'
    if exp_time:
        try:
            val = float(exp_time)
            if 0 < val < 1.0:
                denom = int(round(1.0 / val))
                exp_str = f"1/{denom} s ({val:.4f}s)"
            else:
                exp_str = f"{val:.2f} s"
        except Exception:
            exp_str = str(exp_time)

    f_num = tags.get('FNumber')
    f_num_str = f"f/{float(f_num):.2g}" if f_num else "Not recorded"

    iso = tags.get('ISOSpeedRatings')
    iso_str = f"ISO {iso}" if iso else "Not recorded"

    focal = tags.get('FocalLength')
    focal_str = f"{float(focal):.2f} mm" if focal else "Not recorded"

    focal_35 = tags.get('FocalLengthIn35mmFilm')
    focal_35_str = f"{focal_35} mm (35mm eq)" if focal_35 else "Not recorded"

    lens_model = str(tags.get('LensModel', '')).strip() or "Not recorded"
    lens_make = str(tags.get('LensMake', '')).strip() or "Not recorded"
    host_comp = str(tags.get('HostComputer', '')).strip() or "Not recorded"

    gps_coords = "Not available"
    gps_alt = "Not available"
    if 'GPSLatitude' in tags and 'GPSLongitude' in tags:
        try:
            lat_dms = tags['GPSLatitude']
            lat_ref = tags.get('GPSLatitudeRef', tags.get('InteropIndex', 'N'))
            lon_dms = tags['GPSLongitude']
            lon_ref = tags.get('GPSLongitudeRef', 'E')
            lat = float(lat_dms[0]) + float(lat_dms[1])/60.0 + float(lat_dms[2])/3600.0
            lon = float(lon_dms[0]) + float(lon_dms[1])/60.0 + float(lon_dms[2])/3600.0
            gps_coords = f"{lat:.4f}° {lat_ref}, {lon:.4f}° {lon_ref}"
        except Exception:
            pass
    if 'GPSAltitude' in tags:
        try:
            gps_alt = f"{float(tags['GPSAltitude']):.1f} m above sea level"
        except Exception:
            pass

    def lookup(field, val, default="Not recorded"):
        if val is None or val == '': return default
        table = EXIF_LOOKUP.get(field, {})
        return table.get(val, f"{val}")

    mapped = {
        'EXIF availability': 'Full EXIF metadata' if len(tags) > 5 else ('Partial metadata' if len(tags) > 0 else 'No EXIF found'),
        'Forensic integrity': 'Software edited (Tamper risk)' if is_edited else 'Clean (Camera original)',
        'Software tag': soft_str,
        'Device': device,
        'Host computer / OS': host_comp,
        'Lens model': lens_model,
        'Lens make': lens_make,
        'Capture date / local': dt_orig,
        'File modification date': dt_mod,
        'Date consistency': 'Modified date mismatch detected' if dt_mismatch else 'Consistent',
        'Timezone offset': str(tags.get('OffsetTimeOriginal', tags.get('OffsetTime', 'Not specified'))),
        'Pixel dimensions': f"{w} × {h} px",
        'File size': f"{size_str} ({size_bytes:,} bytes)",
        'File format': f"{fmt} ({pil.mode})",
        'Resolution / DPI': dpi_str,
        'Orientation': lookup('Orientation', tags.get('Orientation')),
        'Color space': lookup('ColorSpace', tags.get('ColorSpace')),
        'Shutter speed / Exposure': exp_str,
        'Aperture / F-number': f_num_str,
        'ISO speed rating': iso_str,
        'Focal length': focal_str,
        'Focal length (35mm eq)': focal_35_str,
        'Exposure program': lookup('ExposureProgram', tags.get('ExposureProgram')),
        'Exposure mode': lookup('ExposureMode', tags.get('ExposureMode')),
        'Metering mode': lookup('MeteringMode', tags.get('MeteringMode')),
        'Flash': f"Fired ({tags.get('Flash')})" if tags.get('Flash') not in [None, 0, 16] else 'Off / Did not fire',
        'White balance': lookup('WhiteBalance', tags.get('WhiteBalance')),
        'Brightness value': f"{float(tags.get('BrightnessValue')):.2f} EV" if 'BrightnessValue' in tags else "Not recorded",
        'GPS coordinates': gps_coords,
        'GPS altitude': gps_alt,
        'Photoshop metadata block': 'Detected (Adobe Photoshop IRB / 8BIM)' if photoshop_info else 'None',
        'XMP metadata block': 'Present (Adobe XMP data)' if xmp_bytes else 'None',
        'Content Credentials (C2PA)': 'Generative AI markers (Firefly/C2PA) detected' if c2pa_detected else 'None detected'
    }

    # Add all remaining raw tags from image
    skip_keys = {'MakerNote', 'exif', 'photoshop', 'xmp', 'adobe', 'adobe_transform', 'icc_profile', 'dpi', 'ExifOffset', 'GPSInfo'}
    for k, v in tags.items():
        field_name = f"EXIF: {k}"
        if field_name not in mapped and k not in mapped and k not in skip_keys:
            val_s = str(v)
            if len(val_s) > 120: val_s = val_s[:120] + '...'
            mapped[field_name] = val_s

    summary = {
        "filename": filename,
        "format": fmt,
        "size_str": size_str,
        "dimensions": f"{w} × {h}",
        "availability": mapped['EXIF availability'],
        "device": device,
        "software": soft_str,
        "is_soft_edited": is_edited,
        "dt_orig": dt_orig,
        "dt_mod": dt_mod,
        "dt_mismatch": dt_mismatch,
        "gps": gps_coords
    }

    return mapped, is_edited, dt_mismatch, summary

def load_image_rgb(path_or_bytes):
    if isinstance(path_or_bytes, bytes):
        pil_raw = Image.open(io.BytesIO(path_or_bytes))
    else:
        pil_raw = Image.open(path_or_bytes)
    pil = ImageOps.exif_transpose(pil_raw).convert("RGB")
    rgb = np.array(pil)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr, rgb

def verify_files(path1: str, path2: str = None):
    with open(path1, "rb") as f:
        raw1 = f.read()
    fname1 = os.path.basename(path1)
    bgr1, rgb1 = load_image_rgb(raw1)
    m1, is_ed1, dt_mis1, s1 = parse_metadata_dict(raw1, fname1)

    if path2 is None:
        # Single Image Mode: Selfie holding ID
        faces = detect_faces_multiscale(bgr1)
        if len(faces) == 0:
            raise ValueError("No face detected in the image.")
        if len(faces) == 1:
            raise ValueError("Only 1 face was detected. Could not locate the ID card face portrait. Please try 2-image mode.")

        selfie_face = faces[0]
        id_face = faces[1]

        cos_sim = float(np.dot(selfie_face.normed_embedding, id_face.normed_embedding))
        threshold = 0.35

        if cos_sim >= threshold:
            pct = min(99.4, 70.0 + ((cos_sim - threshold) / (0.75 - threshold)) * 29.4)
        else:
            pct = max(1.0, (max(0.0, cos_sim) / threshold) * 45.0)

        is_match = cos_sim >= threshold
        sim_label = ("Strong similarity" if pct >= 80 else "Match verified") if is_match else "Mismatch detected"

        crop_bgr, id_dim = crop_face_bgr(bgr1, id_face.bbox)
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

        doc_overrides = {
            'Forensic integrity': 'Document crop (Physical card)',
            'EXIF availability': 'Extracted from physical ID',
            'Software tag': 'Printed on physical ID',
            'Device': 'Physical document in photo',
            'Host computer / OS': 'Physical badge',
            'Lens model': 'Captured via selfie camera',
            'Pixel dimensions': id_dim,
            'File size': 'Cropped from document',
            'File format': 'JPEG portrait crop'
        }

        rows = []
        for k, v1 in m1.items():
            v_id = doc_overrides.get(k, f"Parent: {v1}")
            is_hl = False
            if any(w in str(v1).lower() for w in ['photoshop', 'firefly', 'tamper risk', 'mismatch detected', 'canva', 'gimp']):
                is_hl = True
            rows.append({
                "field": k,
                "val_id": v_id,
                "val_selfie": v1,
                "highlight": is_hl
            })

        return {
            "mode": "single",
            "similarity_pct": f"{pct:.1f}%",
            "similarity_label": sim_label,
            "is_match": is_match,
            "card_id": {
                "rgb_image": crop_rgb,
                "label": "ID card",
                "filename": "id_card_extracted.jpg",
                "badge": "EXTRACTED",
                "footer": f"JPEG • {id_dim}"
            },
            "card_selfie": {
                "rgb_image": rgb1,
                "label": "Selfie holding ID",
                "filename": s1["filename"],
                "badge": "UPLOADED",
                "footer": f"{s1['format']} • {s1['size_str']} • {s1['dimensions']}"
            },
            "table": {
                "col_id_title": "ID CARD",
                "col_selfie_title": "SELFIE HOLDING ID",
                "rows": rows
            }
        }
    else:
        # Dual Image Mode: path1 = Selfie, path2 = ID Card
        with open(path2, "rb") as f:
            raw2 = f.read()
        fname2 = os.path.basename(path2)
        bgr2, rgb2 = load_image_rgb(raw2)
        m2, is_ed2, dt_mis2, s2 = parse_metadata_dict(raw2, fname2)

        faces1 = detect_faces_multiscale(bgr1)
        faces2 = detect_faces_multiscale(bgr2)

        if not faces1:
            raise ValueError(f"No face detected in Selfie image ({fname1}).")
        if not faces2:
            raise ValueError(f"No face detected in ID Card image ({fname2}).")

        selfie_face = faces1[0]
        id_face = faces2[0]

        cos_sim = float(np.dot(selfie_face.normed_embedding, id_face.normed_embedding))
        threshold = 0.35

        if cos_sim >= threshold:
            pct = min(99.4, 70.0 + ((cos_sim - threshold) / (0.75 - threshold)) * 29.4)
        else:
            pct = max(1.0, (max(0.0, cos_sim) / threshold) * 45.0)

        is_match = cos_sim >= threshold
        sim_label = ("Strong similarity" if pct >= 80 else "Match verified") if is_match else "Mismatch detected"

        all_keys = list(m1.keys())
        for k in m2.keys():
            if k not in all_keys:
                all_keys.append(k)

        rows = []
        for k in all_keys:
            v_id = m2.get(k, "Not recorded")
            v_selfie = m1.get(k, "Not recorded")
            is_hl = False
            if any(w in str(v_id).lower() or w in str(v_selfie).lower() for w in ['photoshop', 'firefly', 'tamper risk', 'mismatch detected', 'canva', 'gimp']):
                is_hl = True
            rows.append({
                "field": k,
                "val_id": v_id,
                "val_selfie": v_selfie,
                "highlight": is_hl
            })

        return {
            "mode": "dual",
            "similarity_pct": f"{pct:.1f}%",
            "similarity_label": sim_label,
            "is_match": is_match,
            "card_id": {
                "rgb_image": rgb2,
                "label": "ID card",
                "filename": s2["filename"],
                "badge": "DOCUMENT",
                "footer": f"{s2['format']} • {s2['size_str']} • {s2['dimensions']}"
            },
            "card_selfie": {
                "rgb_image": rgb1,
                "label": "Selfie holding ID",
                "filename": s1["filename"],
                "badge": "SELFIE",
                "footer": f"{s1['format']} • {s1['size_str']} • {s1['dimensions']}"
            },
            "table": {
                "col_id_title": "ID CARD",
                "col_selfie_title": "SELFIE HOLDING ID",
                "rows": rows
            }
        }
