import os
import sys
import io

if sys.stdout is None:
    sys.stdout = io.StringIO()
if sys.stderr is None:
    sys.stderr = io.StringIO()

import time
import numpy as np
import cv2
from PIL import Image

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QScrollArea, QFrame,
    QLineEdit, QTableWidget, QTableWidgetItem, QHeaderView,
    QProgressBar, QGraphicsDropShadowEffect, QSizePolicy, QMessageBox
)
from PySide6.QtCore import Qt, QThread, Signal, QSize, QTimer
from PySide6.QtGui import (
    QIcon, QPixmap, QImage, QColor, QFont, QPainter, QPen, QBrush
)

import biometrics_engine

def cv_to_qpixmap(rgb_img: np.ndarray, max_w: int = 400, max_h: int = 240) -> QPixmap:
    h, w, ch = rgb_img.shape
    bytes_per_line = ch * w
    qimg = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    pix = QPixmap.fromImage(qimg)
    return pix.scaled(max_w, max_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

def file_to_thumb(path: str, size: int = 44) -> QPixmap:
    try:
        pil = Image.open(path)
        pil.thumbnail((size * 2, size * 2))
        pil = pil.convert("RGB")
        rgb = np.array(pil)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        return pix.scaled(size, size, Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation)
    except Exception:
        pix = QPixmap(size, size)
        pix.fill(QColor("#E2E8F0"))
        return pix

class VerifyWorker(QThread):
    sig_finished = Signal(dict)
    sig_error = Signal(str)

    def __init__(self, path1: str, path2: str = None):
        super().__init__()
        self.path1 = path1
        self.path2 = path2

    def run(self):
        try:
            res = biometrics_engine.verify_files(self.path1, self.path2)
            self.sig_finished.emit(res)
        except Exception as e:
            self.sig_error.emit(str(e))

class DropZoneWidget(QFrame):
    sig_files_dropped = Signal(list)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setObjectName("dropZone")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.is_drag_active = False

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setContentsMargins(24, 28, 24, 28)
        layout.setSpacing(6)

        # Upload icon circle
        self.icon_lbl = QLabel()
        self.icon_lbl.setFixedSize(40, 40)
        self.icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.icon_lbl.setObjectName("dropIcon")
        self.icon_lbl.setText("↑")
        layout.addWidget(self.icon_lbl, alignment=Qt.AlignmentFlag.AlignCenter)

        # Main heading
        self.heading_lbl = QLabel("Drag & drop 1 or 2 images here, or browse files")
        self.heading_lbl.setObjectName("dropHeading")
        self.heading_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.heading_lbl)

        # Subheading
        self.subheading_lbl = QLabel("Upload 1 image (Selfie holding ID) or 2 images (ID card + Selfie) • Automatic detection")
        self.subheading_lbl.setObjectName("dropSubheading")
        self.subheading_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.subheading_lbl)

        self.update_style()

    def update_style(self):
        if self.is_drag_active:
            self.setStyleSheet("""
                #dropZone {
                    background-color: #E6F4F1;
                    border: 2px solid #007A64;
                    border-radius: 8px;
                }
                #dropIcon {
                    background-color: #007A64;
                    color: #FFFFFF;
                    border-radius: 20px;
                    font-size: 18px;
                    font-weight: bold;
                }
                #dropHeading {
                    color: #007A64;
                    font-size: 13.5px;
                    font-weight: 600;
                }
                #dropSubheading {
                    color: #005F4E;
                    font-size: 11.5px;
                }
            """)
        else:
            self.setStyleSheet("""
                #dropZone {
                    background-color: #F8FAFC;
                    border: 1px dashed #CBD5E1;
                    border-radius: 8px;
                }
                #dropZone:hover {
                    background-color: #F0FDF4;
                    border-color: #007A64;
                }
                #dropIcon {
                    background-color: #FFFFFF;
                    color: #007A64;
                    border: 1px solid #E2E8F0;
                    border-radius: 20px;
                    font-size: 18px;
                    font-weight: bold;
                }
                #dropHeading {
                    color: #111827;
                    font-size: 13px;
                    font-weight: 600;
                }
                #dropSubheading {
                    color: #64748B;
                    font-size: 11.5px;
                }
            """)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            self.is_drag_active = True
            self.update_style()
            event.acceptProposedAction()

    def dragLeaveEvent(self, event):
        self.is_drag_active = False
        self.update_style()

    def dropEvent(self, event):
        self.is_drag_active = False
        self.update_style()
        urls = event.mimeData().urls()
        files = []
        for u in urls:
            p = u.toLocalFile()
            if p.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                files.append(p)
        if files:
            self.sig_files_dropped.emit(files)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            files, _ = QFileDialog.getOpenFileNames(
                self,
                "Select KYC Images (1 or 2 files)",
                "",
                "Image Files (*.jpg *.jpeg *.png *.webp)"
            )
            if files:
                self.sig_files_dropped.emit(files)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ashraf Finance — Verification results")
        self.resize(1140, 880)
        self.setMinimumSize(960, 720)
        
        # Load app icon if exists
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kyc_verifier.ico")
        if os.path.isfile(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self.current_files = []
        self.current_table_rows = []
        self.worker = None

        self.init_ui()
        self.apply_theme()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ── 1. Header Bar ──
        header_bar = QFrame()
        header_bar.setObjectName("headerBar")
        header_bar.setFixedHeight(58)
        header_layout = QHBoxLayout(header_bar)
        header_layout.setContentsMargins(32, 0, 32, 0)

        # Left brand logo
        brand_wrap = QWidget()
        brand_layout = QHBoxLayout(brand_wrap)
        brand_layout.setContentsMargins(0, 0, 0, 0)
        brand_layout.setSpacing(10)

        logo_icon = QLabel()
        logo_icon.setFixedSize(28, 28)
        logo_icon.setObjectName("brandLogoIcon")
        logo_icon.setText("AF")
        logo_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        brand_layout.addWidget(logo_icon)

        brand_name = QLabel("Ashraf Finance")
        brand_name.setObjectName("brandName")
        brand_layout.addWidget(brand_name)
        header_layout.addWidget(brand_wrap)

        header_layout.addStretch()

        # Right status pill
        pill = QFrame()
        pill.setObjectName("enginePill")
        pill_layout = QHBoxLayout(pill)
        pill_layout.setContentsMargins(10, 4, 10, 4)
        pill_layout.setSpacing(6)

        dot = QLabel()
        dot.setFixedSize(6, 6)
        dot.setObjectName("engineDot")
        pill_layout.addWidget(dot)

        pill_text = QLabel("Automated Biometric Verifier")
        pill_text.setObjectName("enginePillText")
        pill_layout.addWidget(pill_text)

        header_layout.addWidget(pill)
        main_layout.addWidget(header_bar)

        # ── 2. Scrollable Body Content ──
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setObjectName("contentScroll")

        scroll_content = QWidget()
        self.content_layout = QVBoxLayout(scroll_content)
        self.content_layout.setContentsMargins(40, 24, 40, 40)
        self.content_layout.setSpacing(20)

        # Drop Zone
        self.drop_zone = DropZoneWidget()
        self.drop_zone.sig_files_dropped.connect(self.handle_files)
        self.content_layout.addWidget(self.drop_zone)

        # Loading / Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(4)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setRange(0, 0) # Indeterminate
        self.progress_bar.setObjectName("topProgressBar")
        self.progress_bar.hide()
        self.content_layout.addWidget(self.progress_bar)

        self.status_lbl = QLabel("Analyzing facial biometrics and forensic metadata...")
        self.status_lbl.setObjectName("statusLabel")
        self.status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_lbl.hide()
        self.content_layout.addWidget(self.status_lbl)

        # Previews Bar
        self.previews_frame = QFrame()
        self.previews_frame.setObjectName("previewsFrame")
        self.previews_frame.hide()
        previews_layout = QVBoxLayout(self.previews_frame)
        previews_layout.setContentsMargins(16, 12, 16, 12)
        previews_layout.setSpacing(10)

        previews_header = QHBoxLayout()
        self.previews_mode_lbl = QLabel("UPLOADED INPUTS")
        self.previews_mode_lbl.setObjectName("previewsModeLabel")
        previews_header.addWidget(self.previews_mode_lbl)

        previews_header.addStretch()

        btn_clear = QPushButton("✕ Clear inputs")
        btn_clear.setObjectName("btnClear")
        btn_clear.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_clear.clicked.connect(self.reset_all)
        previews_header.addWidget(btn_clear)
        previews_layout.addLayout(previews_header)

        self.previews_grid = QHBoxLayout()
        self.previews_grid.setSpacing(12)
        previews_layout.addLayout(self.previews_grid)
        self.content_layout.addWidget(self.previews_frame)

        # Empty State Guide (shown when no files are loaded)
        self.empty_guide = QFrame()
        self.empty_guide.setObjectName("emptyGuide")
        empty_layout = QVBoxLayout(self.empty_guide)
        empty_layout.setContentsMargins(24, 40, 24, 40)
        empty_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_layout.setSpacing(8)

        empty_icon = QLabel("✦")
        empty_icon.setObjectName("emptyIcon")
        empty_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_layout.addWidget(empty_icon, alignment=Qt.AlignmentFlag.AlignCenter)

        empty_title = QLabel("Awaiting KYC Verification Upload")
        empty_title.setObjectName("emptyTitle")
        empty_layout.addWidget(empty_title, alignment=Qt.AlignmentFlag.AlignCenter)

        empty_desc = QLabel("Drag and drop 1 image (Selfie holding ID) or 2 images (ID Card + Selfie) above to start automated facial biometric verification, tamper analysis, and complete EXIF forensic metadata extraction.")
        empty_desc.setObjectName("emptyDesc")
        empty_desc.setWordWrap(True)
        empty_desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_desc.setMaximumWidth(520)
        empty_layout.addWidget(empty_desc, alignment=Qt.AlignmentFlag.AlignCenter)

        self.content_layout.addWidget(self.empty_guide)

        # ── 3. Verification Results Container ──
        self.results_container = QWidget()
        self.results_container.setObjectName("resultsContainer")
        self.results_container.hide()
        results_layout = QVBoxLayout(self.results_container)
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_layout.setSpacing(18)

        results_title = QLabel("Verification results")
        results_title.setObjectName("resultsTitle")
        results_layout.addWidget(results_title)

        # Block 1: Side-by-side cards
        cards_row = QHBoxLayout()
        cards_row.setSpacing(16)

        # Card 1: ID Card
        self.card_id = self.create_evidence_card("ID card", "EXTRACTED")
        cards_row.addWidget(self.card_id)

        # Card 2: Selfie holding ID
        self.card_selfie = self.create_evidence_card("Selfie holding ID", "UPLOADED")
        cards_row.addWidget(self.card_selfie)

        results_layout.addLayout(cards_row)

        # Block 2: Similarity Metric Strip
        self.metric_frame = QFrame()
        self.metric_frame.setObjectName("metricFrame")
        metric_layout = QHBoxLayout(self.metric_frame)
        metric_layout.setContentsMargins(20, 14, 20, 14)

        metric_left = QHBoxLayout()
        metric_left.setSpacing(10)
        sim_icon = QLabel("☵")
        sim_icon.setFixedSize(28, 28)
        sim_icon.setObjectName("simIcon")
        sim_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        metric_left.addWidget(sim_icon)

        sim_title = QLabel("ID card ↔ Selfie holding ID")
        sim_title.setObjectName("simTitle")
        metric_left.addWidget(sim_title)
        metric_layout.addLayout(metric_left)

        metric_layout.addStretch()

        metric_right = QHBoxLayout()
        metric_right.setSpacing(14)
        
        score_wrap = QVBoxLayout()
        score_wrap.setSpacing(2)
        score_wrap.setAlignment(Qt.AlignmentFlag.AlignRight)
        
        self.sim_score_lbl = QLabel("80.6%")
        self.sim_score_lbl.setObjectName("simScore")
        score_wrap.addWidget(self.sim_score_lbl, alignment=Qt.AlignmentFlag.AlignRight)

        self.sim_bar = QFrame()
        self.sim_bar.setFixedHeight(3)
        self.sim_bar.setObjectName("simUnderline")
        score_wrap.addWidget(self.sim_bar)
        metric_right.addLayout(score_wrap)

        self.sim_label_lbl = QLabel("Strong similarity")
        self.sim_label_lbl.setObjectName("simStatusLabel")
        metric_right.addWidget(self.sim_label_lbl)

        metric_layout.addLayout(metric_right)
        results_layout.addWidget(self.metric_frame)

        # Block 3: EXIF & File Metadata Table
        table_section = QFrame()
        table_section.setObjectName("tableSection")
        table_sec_layout = QVBoxLayout(table_section)
        table_sec_layout.setContentsMargins(0, 0, 0, 0)
        table_sec_layout.setSpacing(0)

        # Table header toolbar
        tb_bar = QFrame()
        tb_bar.setObjectName("tableHeaderBar")
        tb_bar.setFixedHeight(50)
        tb_bar_layout = QHBoxLayout(tb_bar)
        tb_bar_layout.setContentsMargins(18, 0, 18, 0)
        tb_bar_layout.setSpacing(12)

        tb_title = QLabel("EXIF & file metadata")
        tb_title.setObjectName("tableTitle")
        tb_bar_layout.addWidget(tb_title)

        self.badge_count = QLabel("Showing all fields")
        self.badge_count.setObjectName("fieldsBadge")
        tb_bar_layout.addWidget(self.badge_count)

        tb_bar_layout.addStretch()

        self.search_input = QLineEdit()
        self.search_input.setObjectName("metaSearch")
        self.search_input.setPlaceholderText("Search metadata fields (e.g. GPS, ISO, Date, Photoshop)...")
        self.search_input.setFixedWidth(280)
        self.search_input.textChanged.connect(self.filter_table)
        tb_bar_layout.addWidget(self.search_input)

        table_sec_layout.addWidget(tb_bar)

        # Table widget
        self.table_widget = QTableWidget(0, 3)
        self.table_widget.setObjectName("metaTable")
        self.table_widget.setHorizontalHeaderLabels(["METADATA FIELD", "ID CARD", "SELFIE HOLDING ID"])
        self.table_widget.horizontalHeader().setStretchLastSection(True)
        self.table_widget.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        self.table_widget.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        self.table_widget.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table_widget.setColumnWidth(0, 240)
        self.table_widget.setColumnWidth(1, 380)
        self.table_widget.verticalHeader().setVisible(False)
        self.table_widget.setShowGrid(False)
        self.table_widget.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table_widget.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table_widget.setMinimumHeight(380)
        table_sec_layout.addWidget(self.table_widget)

        results_layout.addWidget(table_section)
        self.content_layout.addWidget(self.results_container)

        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)

    def create_evidence_card(self, title_text: str, badge_text: str) -> QFrame:
        card = QFrame()
        card.setObjectName("evidenceCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Card top bar
        top = QFrame()
        top.setObjectName("cardTopBar")
        top.setFixedHeight(38)
        top_l = QHBoxLayout(top)
        top_l.setContentsMargins(14, 0, 14, 0)
        top_l.setSpacing(8)

        icon = QLabel("🪪" if "ID" in title_text else "👤")
        icon.setObjectName("cardHeaderIcon")
        top_l.addWidget(icon)

        title = QLabel(title_text)
        title.setObjectName("cardHeaderTitle")
        top_l.addWidget(title)

        top_l.addStretch()
        layout.addWidget(top)

        # Viewport with image
        viewport = QFrame()
        viewport.setObjectName("cardViewport")
        viewport.setFixedHeight(240)
        vp_l = QVBoxLayout(viewport)
        vp_l.setContentsMargins(10, 10, 10, 10)

        badge_row = QHBoxLayout()
        badge = QLabel(badge_text)
        badge.setObjectName("cardBadge")
        badge_row.addWidget(badge, alignment=Qt.AlignmentFlag.AlignLeft)
        badge_row.addStretch()
        vp_l.addLayout(badge_row)

        img_lbl = QLabel()
        img_lbl.setObjectName("cardImage")
        img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vp_l.addWidget(img_lbl, 1, Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(viewport)

        # Bottom bar
        bot = QFrame()
        bot.setObjectName("cardBotBar")
        bot.setFixedHeight(48)
        bot_l = QVBoxLayout(bot)
        bot_l.setContentsMargins(14, 6, 14, 6)
        bot_l.setSpacing(1)

        fname = QLabel("filename.jpg")
        fname.setObjectName("cardFilename")
        bot_l.addWidget(fname)

        fmeta = QLabel("JPEG • 0 × 0")
        fmeta.setObjectName("cardFilemeta")
        bot_l.addWidget(fmeta)

        layout.addWidget(bot)

        # Store references on card object
        card.lbl_badge = badge
        card.lbl_title = title
        card.lbl_image = img_lbl
        card.lbl_fname = fname
        card.lbl_fmeta = fmeta

        return card

    def apply_theme(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #FFFFFF;
                font-family: 'Segoe UI', Inter, -apple-system, sans-serif;
            }
            #headerBar {
                background-color: #FFFFFF;
                border-bottom: 1px solid #E5E7EB;
            }
            #brandLogoIcon {
                background-color: #007A64;
                color: #FFFFFF;
                border-radius: 6px;
                font-weight: bold;
                font-size: 13px;
            }
            #brandName {
                color: #111827;
                font-size: 16px;
                font-weight: 700;
                letter-spacing: -0.3px;
            }
            #enginePill {
                background-color: #FFFFFF;
                border: 1px solid #E5E7EB;
                border-radius: 14px;
            }
            #engineDot {
                background-color: #007A64;
                border-radius: 3px;
            }
            #enginePillText {
                color: #64748B;
                font-size: 11.5px;
                font-weight: 500;
            }
            #contentScroll {
                background-color: #FFFFFF;
            }
            #topProgressBar {
                border: none;
                background-color: #E6F4F1;
                border-radius: 2px;
            }
            #topProgressBar::chunk {
                background-color: #007A64;
                border-radius: 2px;
            }
            #statusLabel {
                color: #007A64;
                font-size: 12.5px;
                font-weight: 600;
            }
            #previewsFrame {
                background-color: #F8FAFC;
                border: 1px solid #E2E8F0;
                border-radius: 8px;
            }
            #previewsModeLabel {
                color: #64748B;
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 0.5px;
            }
            #btnClear {
                background: transparent;
                border: none;
                color: #64748B;
                font-size: 11.5px;
                font-weight: 600;
            }
            #btnClear:hover {
                color: #DC2626;
            }
            #emptyGuide {
                background-color: #FAFAFA;
                border: 1px dashed #E2E8F0;
                border-radius: 8px;
            }
            #emptyIcon {
                color: #007A64;
                font-size: 24px;
            }
            #emptyTitle {
                color: #111827;
                font-size: 14px;
                font-weight: 700;
            }
            #emptyDesc {
                color: #64748B;
                font-size: 12px;
                line-height: 1.5;
            }
            #resultsTitle {
                color: #111827;
                font-size: 20px;
                font-weight: 700;
                letter-spacing: -0.4px;
            }
            #evidenceCard {
                background-color: #FFFFFF;
                border: 1px solid #E2E8F0;
                border-radius: 8px;
            }
            #cardTopBar {
                background-color: #FFFFFF;
                border-bottom: 1px solid #F1F5F9;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
            }
            #cardHeaderTitle {
                color: #111827;
                font-size: 12.5px;
                font-weight: 600;
            }
            #cardViewport {
                background-color: #F8FAFC;
            }
            #cardBadge {
                background-color: rgba(255, 255, 255, 0.95);
                border: 1px solid #E2E8F0;
                border-radius: 4px;
                color: #475569;
                font-size: 9.5px;
                font-weight: 700;
                padding: 2px 6px;
                letter-spacing: 0.5px;
            }
            #cardBotBar {
                background-color: #FFFFFF;
                border-top: 1px solid #F1F5F9;
                border-bottom-left-radius: 8px;
                border-bottom-right-radius: 8px;
            }
            #cardFilename {
                color: #111827;
                font-size: 12px;
                font-weight: 600;
            }
            #cardFilemeta {
                color: #64748B;
                font-size: 11px;
            }
            #metricFrame {
                background-color: #FFFFFF;
                border: 1px solid #E2E8F0;
                border-radius: 8px;
            }
            #simIcon {
                background-color: #E6F4F1;
                color: #007A64;
                border-radius: 6px;
                font-size: 14px;
                font-weight: bold;
            }
            #simTitle {
                color: #111827;
                font-size: 13px;
                font-weight: 600;
            }
            #simScore {
                color: #111827;
                font-size: 26px;
                font-weight: 800;
                letter-spacing: -0.5px;
            }
            #simUnderline {
                background-color: #007A64;
                border-radius: 1px;
            }
            #simStatusLabel {
                color: #007A64;
                font-size: 13.5px;
                font-weight: 600;
            }
            #tableSection {
                background-color: #FFFFFF;
                border: 1px solid #E2E8F0;
                border-radius: 8px;
            }
            #tableHeaderBar {
                background-color: #FFFFFF;
                border-bottom: 1px solid #E2E8F0;
            }
            #tableTitle {
                color: #111827;
                font-size: 14px;
                font-weight: 700;
            }
            #fieldsBadge {
                background-color: #E6F4F1;
                color: #007A64;
                border-radius: 10px;
                font-size: 11px;
                font-weight: 600;
                padding: 2px 8px;
            }
            #metaSearch {
                background-color: #F8FAFC;
                border: 1px solid #CBD5E1;
                border-radius: 6px;
                padding: 4px 10px;
                font-size: 11.5px;
                color: #111827;
            }
            #metaSearch:focus {
                border-color: #007A64;
                background-color: #FFFFFF;
            }
            #metaTable {
                background-color: #FFFFFF;
                border: none;
                font-size: 12px;
            }
            #metaTable QHeaderView::section {
                background-color: #FAFAFA;
                color: #64748B;
                border: none;
                border-bottom: 1px solid #E2E8F0;
                font-size: 10.5px;
                font-weight: 600;
                letter-spacing: 0.5px;
                padding: 8px 16px;
            }
            #metaTable::item {
                padding: 8px 16px;
                border-bottom: 1px solid #F1F5F9;
            }
        """)

    def handle_files(self, files: list):
        if not files:
            return
        self.current_files = files[:2]
        self.render_input_previews(self.current_files)

        # Start background worker
        self.progress_bar.show()
        self.status_lbl.show()
        self.empty_guide.hide()

        path1 = self.current_files[0]
        path2 = self.current_files[1] if len(self.current_files) > 1 else None

        self.worker = VerifyWorker(path1, path2)
        self.worker.sig_finished.connect(self.on_verification_done)
        self.worker.sig_error.connect(self.on_verification_error)
        self.worker.start()

    def render_input_previews(self, files: list):
        # Clear existing preview items
        while self.previews_grid.count():
            item = self.previews_grid.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        if len(files) == 1:
            self.previews_mode_lbl.setText("1 INPUT IMAGE (AUTO-DETECTED: SELFIE HOLDING ID)")
        else:
            self.previews_mode_lbl.setText("2 INPUT IMAGES (AUTO-DETECTED: DUAL-IMAGE COMPARISON)")

        for idx, fpath in enumerate(files):
            card = QFrame()
            card.setStyleSheet("""
                QFrame {
                    background-color: #FFFFFF;
                    border: 1px solid #E2E8F0;
                    border-radius: 6px;
                    padding: 4px;
                }
            """)
            c_l = QHBoxLayout(card)
            c_l.setContentsMargins(6, 6, 8, 6)
            c_l.setSpacing(8)

            thumb_lbl = QLabel()
            thumb_lbl.setFixedSize(44, 44)
            thumb_lbl.setPixmap(file_to_thumb(fpath, 44))
            thumb_lbl.setStyleSheet("border-radius: 4px; background: #E2E8F0;")
            c_l.addWidget(thumb_lbl)

            info_wrap = QVBoxLayout()
            info_wrap.setSpacing(1)

            role_str = "SOURCE PHOTO" if len(files) == 1 else ("INPUT 1: SELFIE" if idx == 0 else "INPUT 2: ID CARD")
            role_lbl = QLabel(role_str)
            role_lbl.setStyleSheet("color: #007A64; font-size: 10px; font-weight: 700; letter-spacing: 0.5px;")
            info_wrap.addWidget(role_lbl)

            name_lbl = QLabel(os.path.basename(fpath))
            name_lbl.setStyleSheet("color: #111827; font-size: 12px; font-weight: 600;")
            info_wrap.addWidget(name_lbl)

            sz = os.path.getsize(fpath)
            sz_str = f"{sz / (1024*1024):.1f} MB" if sz >= 1024*1024 else f"{sz / 1024:.0f} KB"
            size_lbl = QLabel(sz_str)
            size_lbl.setStyleSheet("color: #64748B; font-size: 11px;")
            info_wrap.addWidget(size_lbl)

            c_l.addLayout(info_wrap)
            self.previews_grid.addWidget(card)

        self.previews_grid.addStretch()
        self.previews_frame.show()

    def on_verification_done(self, res: dict):
        self.progress_bar.hide()
        self.status_lbl.hide()

        # Update Card 1 (ID Card)
        c_id = res["card_id"]
        self.card_id.lbl_title.setText(c_id["label"])
        self.card_id.lbl_badge.setText(c_id["badge"])
        self.card_id.lbl_fname.setText(c_id["filename"])
        self.card_id.lbl_fmeta.setText(c_id["footer"])
        self.card_id.lbl_image.setPixmap(cv_to_qpixmap(c_id["rgb_image"], 340, 210))

        # Update Card 2 (Selfie)
        c_sf = res["card_selfie"]
        self.card_selfie.lbl_title.setText(c_sf["label"])
        self.card_selfie.lbl_badge.setText(c_sf["badge"])
        self.card_selfie.lbl_fname.setText(c_sf["filename"])
        self.card_selfie.lbl_fmeta.setText(c_sf["footer"])
        self.card_selfie.lbl_image.setPixmap(cv_to_qpixmap(c_sf["rgb_image"], 340, 210))

        # Update Similarity Metric
        self.sim_score_lbl.setText(res["similarity_pct"])
        self.sim_label_lbl.setText(res["similarity_label"])
        if res["is_match"]:
            self.sim_bar.setStyleSheet("background-color: #007A64; border-radius: 1px;")
            self.sim_label_lbl.setStyleSheet("color: #007A64; font-size: 13.5px; font-weight: 600;")
        else:
            self.sim_bar.setStyleSheet("background-color: #DC2626; border-radius: 1px;")
            self.sim_label_lbl.setStyleSheet("color: #DC2626; font-size: 13.5px; font-weight: 600;")

        # Update Table Headers & Rows
        tb_data = res["table"]
        self.table_widget.setHorizontalHeaderLabels([
            "METADATA FIELD",
            tb_data["col_id_title"],
            tb_data["col_selfie_title"]
        ])
        self.current_table_rows = tb_data["rows"]
        self.search_input.clear()
        self.render_table(self.current_table_rows)

        self.empty_guide.hide()
        self.results_container.show()

    def on_verification_error(self, err_msg: str):
        self.progress_bar.hide()
        self.status_lbl.hide()
        self.empty_guide.show()
        QMessageBox.warning(self, "Verification Alert", f"Could not complete verification:\n\n{err_msg}")

    def render_table(self, rows: list):
        self.table_widget.setRowCount(len(rows))
        self.badge_count.setText(f"Showing all {len(rows)} metadata fields")

        for r_idx, r in enumerate(rows):
            # Column 0: Field Name
            item0 = QTableWidgetItem(r["field"])
            item0.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
            item0.setForeground(QColor("#111827"))
            self.table_widget.setItem(r_idx, 0, item0)

            # Column 1: ID Card Value
            self.set_table_cell(r_idx, 1, r["val_id"], r["highlight"] and ("photoshop" in str(r["val_id"]).lower() or "tamper" in str(r["val_id"]).lower()))

            # Column 2: Selfie Value
            self.set_table_cell(r_idx, 2, r["val_selfie"], r["highlight"])

            self.table_widget.setRowHeight(r_idx, 36)

    def set_table_cell(self, row: int, col: int, val, is_highlight: bool):
        val_str = str(val) if val is not None else "Not available"
        item = QTableWidgetItem(val_str)
        item.setFont(QFont("Consolas", 9))

        if is_highlight or any(k in val_str.lower() for k in ["photoshop", "tamper risk", "mismatch detected", "firefly"]):
            item.setBackground(QColor("#FFF7ED"))
            item.setForeground(QColor("#C2410C"))
            item.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
        elif val_str in ["Not recorded", "Not available", "None"]:
            item.setForeground(QColor("#94A3B8"))
        else:
            item.setForeground(QColor("#334155"))

        self.table_widget.setItem(row, col, item)

    def filter_table(self, query: str):
        q = query.lower().strip()
        matched = 0
        for r_idx in range(self.table_widget.rowCount()):
            if not q:
                self.table_widget.setRowHidden(r_idx, False)
                matched += 1
            else:
                f_text = self.table_widget.item(r_idx, 0).text().lower()
                c1_text = self.table_widget.item(r_idx, 1).text().lower()
                c2_text = self.table_widget.item(r_idx, 2).text().lower()
                if q in f_text or q in c1_text or q in c2_text:
                    self.table_widget.setRowHidden(r_idx, False)
                    matched += 1
                else:
                    self.table_widget.setRowHidden(r_idx, True)

        total = self.table_widget.rowCount()
        if not q:
            self.badge_count.setText(f"Showing all {total} metadata fields")
        else:
            self.badge_count.setText(f"Found {matched} of {total} fields")

    def reset_all(self):
        self.current_files = []
        self.current_table_rows = []
        self.previews_frame.hide()
        self.results_container.hide()
        self.empty_guide.show()
        self.progress_bar.hide()
        self.status_lbl.hide()

def main():
    import multiprocessing
    multiprocessing.freeze_support()

    try:
        app = QApplication.instance() or QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        log_path = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "kyc_gui_error.log")
        with open(log_path, "w", encoding="utf-8") as f:
            import traceback
            traceback.print_exc(file=f)
        raise

if __name__ == "__main__":
    main()
