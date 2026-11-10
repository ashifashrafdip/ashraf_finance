# Ashraf Finance — Biometric ID Verification & Image Forensics

An automated identity verification and digital image forensics platform built with **InsightFace (ArcFace)**, **OpenCV**, and **PySide6 (Qt 6 Desktop GUI)**.

It automatically compares identity cards and selfie photos to determine biometric identity matching, detects image manipulation (Photoshop, Generative AI, face-swapping), and extracts over 75+ camera, optical, GPS, and EXIF metadata parameters.

---

## ✨ Features

- **Native Python Desktop GUI (PySide6 / Qt 6):**
  - Standalone desktop application window with zero browser or localhost dependencies.
  - Senior-designer grade Fintech / Stripe-inspired visual styling.
  - Unified drag & drop zone directly on the window.
  - Live thumbnail previews before verification.
  - Multi-threaded asynchronous processing so the UI never lags or freezes.

- **Biometric Face Verification (ArcFace):**
  - High-precision facial representation matching using deep embedding vectors (ArcFace `buffalo_l`).
  - Calibrated similarity scoring with clear match / mismatch classification.

- **Dual Mode Biometric Detection:**
  - **Single-Image Mode (Selfie Holding ID):** Multi-scale face detection (640px and 1600px) locates both the user's face and tiny ID card portrait in a single photo, crops the ID badge photo, and verifies identity.
  - **Dual-Image Mode (ID Card + Selfie):** Compares an ID card photo and selfie photo side-by-side.

- **Digital Forensics & Tamper Detection:**
  - Detects editing software tags (e.g. `Adobe Photoshop 27.1`, GIMP, Canva, Adobe Firefly).
  - Inspects Adobe Photoshop IRB (8BIM) blocks and XMP Core history.
  - Generative AI & Content Credentials (C2PA) marker detection.
  - Date consistency audit (flags discrepancy between camera capture date and file modification date).

- **Complete EXIF & File Metadata Inspection (75+ Fields):**
  - Camera & Device: Make, Model, Host Computer/OS, Lens Model & Specification.
  - Optical & Shot Parameters: Exposure Time, Aperture (F-number), ISO, Focal Length, 35mm Equivalent, Metering Mode, Flash, White Balance, Brightness.
  - Geolocation: GPS Latitude, Longitude, Altitude, UTC Timestamp.
  - Technical: Dimensions, Color Space (sRGB / Display P3), DPI, Orientation, Format.
  - Live metadata search box to instantly filter any field or value.

---

## 🚀 Quick Start

### 1. Run the Native Desktop App
Double-click **`KYC_Verifier.exe`** directly from the project folder!

Or launch from Python:
```bash
python gui_verifier.py
```
Or double-click **`Run_KYC_GUI.bat`**.

### 2. Installation (from source)
Clone the repository:
```bash
git clone https://github.com/ashifashrafdip/ashraf_finance.git
cd ashraf_finance
```

Install dependencies:
```bash
python -m pip install -r requirements.txt
```

---

## 📄 License
MIT License.
