#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import date
import calendar
from urllib.parse import urljoin, unquote
import subprocess
import numpy as np
from astropy.io import fits
from PIL import Image

import requests
from html.parser import HTMLParser

DOWNLOAD_ROOT = Path.cwd() / "downloads"  # change if you want somewhere else


def write_concat_list(files_in_order: list[Path], list_path: Path) -> None:
    """
    Create an ffmpeg concat list file preserving the exact order.
    """
    list_path.parent.mkdir(parents=True, exist_ok=True)
    with open(list_path, "w", encoding="utf-8") as f:
        for p in files_in_order:
            # ffmpeg concat expects: file '<path>'
            # Use forward slashes to avoid Windows escaping issues.
            f.write(f"file '{p.as_posix()}'\n")


def build_url(camera_type: str, image_size: str, image_format: str, yyyy: int, mm: int, dd: int) -> str:
    """
    Build the base URL for SOHO LASCO images.

    jpg:
      https://soho.nascom.nasa.gov/data/REPROCESSING/Completed/{YEAR}/{camera}/{YYYYMMDD}/

    realtime fits:
      https://umbra.nascom.nasa.gov/pub/lasco/lastimage/level_05/{YYMMDD}/{camera}/

    archive fits:
      https://umbra.nascom.nasa.gov/pub/lasco_level05/{YYMMDD}/{camera}/
    """
    cam = camera_type.lower()  # server expects lowercase

    if image_format == "jpg":
        return (
            "https://soho.nascom.nasa.gov/data/REPROCESSING/Completed/"
            f"{yyyy:04d}/{cam}/{yyyy:04d}{mm:02d}{dd:02d}/"
        )

    # Common logic for FITS (both realtime and archive use YYMMDD)
    if "fits" in image_format:
        yy = yyyy % 100
        subdir = f"{yy:02d}{mm:02d}{dd:02d}/{cam}/"

        if image_format == "realtime fits":
            return f"https://umbra.nascom.nasa.gov/pub/lasco/lastimage/level_05/{subdir}"

        if image_format == "archive fits":
            return f"https://umbra.nascom.nasa.gov/pub/lasco_level05/{subdir}"

    raise ValueError(f"Unsupported image format: {image_format}")


def build_download_subfolder(camera_type: str, image_format: str, yyyy: int, mm: int, dd: int) -> str:
    """
    Option B folder naming:
      YYYYMMDD_LASCO_{camera}_{format}
    Example:
      20260111_LASCO_c3_archive_fits
    """
    cam = camera_type.lower()
    yyyymmdd = f"{yyyy:04d}{mm:02d}{dd:02d}"
    # Replace spaces with underscores for cleaner folder names (e.g. "archive fits" -> "archive_fits")
    fmt = image_format.lower().replace(" ", "_")
    return f"{yyyymmdd}_LASCO_{cam}_{fmt}"


class HrefParser(HTMLParser):
    """Minimal parser to extract href values from anchor tags."""

    def __init__(self):
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() != "a":
            return
        for k, v in attrs:
            if k.lower() == "href" and v:
                self.hrefs.append(v)


def filename_matches_size(name: str, size: str) -> bool:
    """
    Return True if the filename appears to be for the requested size (512/1024).

    Common patterns:
      *_512.jpg, *_1024.jpg
      *_512.png, *_1024.png
      *_512.fts, *_1024.fts
      ..._512... etc
    """
    stem = Path(name).stem.lower()  # filename without extension
    size = str(size)

    # Most common: token separated by underscores
    tokens = stem.replace("-", "_").split("_")
    if size in tokens:
        return True

    # Fallback: endswith digits (some datasets do this)
    if stem.endswith(size):
        return True

    return False


def fetch_directory_listing(url: str, timeout_s: int = 20) -> list[str]:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; CameraUrlTool/1.0)"}
    r = requests.get(url, headers=headers, timeout=timeout_s)
    r.raise_for_status()

    parser = HrefParser()
    parser.feed(r.text)

    names: set[str] = set()
    for href in parser.hrefs:
        href = href.strip()

        if href in ("../", "./", "/"):
            continue
        if href.lower().startswith("?"):
            continue

        last = href.split("/")[-1]
        last = unquote(last)

        if href.endswith("/"):
            continue
        if not last:
            continue

        names.add(last)

    return sorted(names, key=str.lower)


def download_file(url: str, dest_path: Path, timeout_s: int = 30) -> None:
    """
    Download a URL to dest_path using streaming.
    """
    headers = {"User-Agent": "Mozilla/5.0 (compatible; CameraUrlTool/1.0)"}
    with requests.get(url, headers=headers, stream=True, timeout=timeout_s) as r:
        r.raise_for_status()
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)


def run_ffmpeg_concat(list_txt: Path, output_mp4: Path, fps: int = 30) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-r", str(fps),
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_txt),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(output_mp4),
    ]
    subprocess.run(cmd, check=True)


def fits_to_png(fits_path: Path, png_path: Path) -> None:
    with fits.open(fits_path, memmap=False) as hdul:
        data = hdul[0].data

    data = np.asarray(data)
    while data.ndim > 2:
        data = data[0]

    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    lo, hi = np.percentile(data, (1, 99))
    if hi <= lo:
        lo, hi = float(data.min()), float(data.max())
        if hi <= lo:
            hi = lo + 1.0

    scaled = np.clip((data - lo) / (hi - lo), 0, 1)
    img8 = (scaled * 255).astype(np.uint8)

    png_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img8).save(png_path)


def make_video_from_download(dest_dir: Path, filenames_in_order: list[str],
                             camera_type: str, fmt: str, yyyy: int, mm: int, dd: int,
                             fps: int = 30) -> Path:
    fmt = fmt.lower().strip()
    yyyymmdd = f"{yyyy:04d}{mm:02d}{dd:02d}"
    cam = camera_type.lower()

    safe_fmt = fmt.replace(" ", "_")
    video_name = f"{yyyymmdd}_LASCO_{cam}_{safe_fmt}_{fps}fps.mp4"
    video_path = dest_dir / video_name
    list_txt = dest_dir / "ffmpeg_list.txt"

    if "jpg" in fmt:
        frames = [dest_dir / name for name in filenames_in_order]
        frames = [p for p in frames if p.exists()]
        if not frames:
            raise RuntimeError("No JPG frames exist in the download folder to build video.")
        write_concat_list(frames, list_txt)
        run_ffmpeg_concat(list_txt, video_path, fps=fps)
        return video_path

    if "fits" in fmt:
        conv_dir = dest_dir / "_png_frames"
        conv_dir.mkdir(parents=True, exist_ok=True)

        png_frames: list[Path] = []
        for name in filenames_in_order:
            fits_path = dest_dir / name
            if not fits_path.exists():
                continue
            png_name = Path(name).with_suffix(".png").name
            png_path = conv_dir / png_name
            fits_to_png(fits_path, png_path)
            png_frames.append(png_path)

        if not png_frames:
            raise RuntimeError("No FITS frames could be converted to PNG.")
        write_concat_list(png_frames, list_txt)
        run_ffmpeg_concat(list_txt, video_path, fps=fps)
        return video_path

    raise ValueError(f"Unsupported format for video: {fmt}")


class CameraUrlTool(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SOHO LASCO URL Builder")
        self.resizable(False, False)
        self._build_menu()

        today = date.today()
        self.camera_options = ["C2", "C3"]
        self.size_options = ["512", "1024"]

        self.format_options = ["jpg", "realtime fits", "archive fits"]

        self.fps_options = ["60", "30", "24", "15", "10", "5", "2", "1"]
        self.var_fps = tk.StringVar(value="10")  # default slow-ish, good for LASCO

        this_year = today.year
        self.year_options = [str(y) for y in range(this_year, this_year - 21, -1)]

        self.var_camera = tk.StringVar(value=self.camera_options[0])
        self.var_size = tk.StringVar(value=self.size_options[0])
        self.var_format = tk.StringVar(value=self.format_options[0])
        self.var_year = tk.StringVar(value=str(today.year))
        self.var_month = tk.StringVar(value=f"{today.month:02d}")
        self.var_day = tk.StringVar(value=f"{today.day:02d}")

        self._downloading = False

        self._build_ui()
        self._refresh_day_options()
        self._set_url_preview()

    def on_about(self):
        messagebox.showinfo(
            "About",
            "SOHO LASCO Image Downloader\n\n"
            "A tool for downloading and creating videos from SOHO LASCO data.\n\n"
            "Author: Steven Sagerian\n"
            "Email: steven.sagerian@gmail.com\n\n"
            "Version: 1.1.0\n\n"
            "GitHub: https://github.com/ssagerian/soho-lasco-downloader\n"

        )

    def _build_menu(self):
        menubar = tk.Menu(self)

        about_menu = tk.Menu(menubar, tearoff=0)
        about_menu.add_command(label="About", command=self.on_about)

        menubar.add_cascade(label="About", menu=about_menu)

        self.config(menu=menubar)

    def _build_ui(self):
        pad = {"padx": 8, "pady": 6}

        frm = ttk.Frame(self)
        frm.grid(row=0, column=0, sticky="nsew", **pad)
        frm.columnconfigure(1, weight=1)

        def add_row(r, label, widget):
            ttk.Label(frm, text=label).grid(row=r, column=0, sticky="w", **pad)
            widget.grid(row=r, column=1, sticky="ew", **pad)

        self.cmb_camera = ttk.Combobox(frm, textvariable=self.var_camera, values=self.camera_options, state="readonly")
        add_row(0, "Camera type", self.cmb_camera)

        self.cmb_size = ttk.Combobox(frm, textvariable=self.var_size, values=self.size_options, state="readonly")
        add_row(1, "Image size (jpg)", self.cmb_size)

        self.cmb_format = ttk.Combobox(frm, textvariable=self.var_format, values=self.format_options, state="readonly")
        add_row(2, "Image format", self.cmb_format)

        self.cmb_year = ttk.Combobox(frm, textvariable=self.var_year, values=self.year_options, state="readonly")
        add_row(3, "Year", self.cmb_year)

        month_values = [f"{m:02d}" for m in range(1, 13)]
        self.cmb_month = ttk.Combobox(frm, textvariable=self.var_month, values=month_values, state="readonly")
        add_row(4, "Month (MM)", self.cmb_month)

        self.cmb_day = ttk.Combobox(frm, textvariable=self.var_day, values=[], state="readonly")
        add_row(5, "Day (DD)", self.cmb_day)

        ttk.Label(frm, text="URL").grid(row=6, column=0, sticky="w", **pad)
        self.txt_url = tk.Text(frm, height=2, width=78, wrap="word")
        self.txt_url.grid(row=6, column=1, sticky="ew", **pad)

        # File list
        ttk.Label(frm, text="Files").grid(row=7, column=0, sticky="nw", **pad)
        files_frame = ttk.Frame(frm)
        files_frame.grid(row=7, column=1, sticky="nsew", **pad)

        self.lst_files = tk.Listbox(files_frame, height=12, width=78, selectmode="extended")
        scr = ttk.Scrollbar(files_frame, orient="vertical", command=self.lst_files.yview)
        self.lst_files.configure(yscrollcommand=scr.set)
        self.lst_files.grid(row=0, column=0, sticky="nsew")
        scr.grid(row=0, column=1, sticky="ns")
        files_frame.columnconfigure(0, weight=1)
        files_frame.rowconfigure(0, weight=1)

        # Ctrl+A select all
        self.lst_files.bind("<Control-a>", self._select_all_files)
        self.lst_files.bind("<Control-A>", self._select_all_files)

        # Progress + status
        self.var_status = tk.StringVar(value="")
        self.progress = ttk.Progressbar(frm, mode="determinate", length=420)
        self.progress.grid(row=8, column=1, sticky="w", **pad)
        ttk.Label(frm, textvariable=self.var_status).grid(row=8, column=1, sticky="e", **pad)
        # FPS selector for video creation
        fps_row = ttk.Frame(frm)
        fps_row.grid(row=9, column=0, columnspan=2, sticky="e", padx=8, pady=(0, 6))
        ttk.Label(fps_row, text="Video FPS").grid(row=0, column=0, padx=(0, 6))
        self.cmb_fps = ttk.Combobox(fps_row, textvariable=self.var_fps, values=self.fps_options, state="readonly",
                                    width=6)
        self.cmb_fps.grid(row=0, column=1)

        # Buttons
        btns = ttk.Frame(frm)
        btns.grid(row=10, column=0, columnspan=2, sticky="e", **pad)

        self.btn_configure = ttk.Button(btns, text="Configure", command=self.on_configure)
        self.btn_list = ttk.Button(btns, text="List Files", command=self.on_list)
        self.btn_load = ttk.Button(btns, text="Download", command=self.on_load)

        self.btn_configure.grid(row=0, column=0, padx=6)
        self.btn_list.grid(row=0, column=1, padx=6)
        self.btn_load.grid(row=0, column=2, padx=6)

        self.btn_video = ttk.Button(btns, text="Make Video", command=self.on_make_video)
        self.btn_video.grid(row=0, column=3, padx=6)
        #  self.btn_video.config(state=state)

        # React to date changes so day dropdown stays valid
        self.cmb_year.bind("<<ComboboxSelected>>", lambda e: self._on_year_month_changed())
        self.cmb_month.bind("<<ComboboxSelected>>", lambda e: self._on_year_month_changed())

        # Update URL preview when dropdown changes
        for cmb in (self.cmb_camera, self.cmb_size, self.cmb_format, self.cmb_day):
            cmb.bind("<<ComboboxSelected>>", lambda e: self._set_url_preview())

    def _select_all_files(self, _event=None):
        self.lst_files.select_set(0, "end")
        return "break"

    def _set_buttons_enabled(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        self.btn_configure.config(state=state)
        self.btn_list.config(state=state)
        self.btn_load.config(state=state)

        # Only if the button exists
        if hasattr(self, "btn_video") and self.btn_video is not None:
            self.btn_video.config(state=state)

    def _on_year_month_changed(self):
        self._refresh_day_options()
        self._set_url_preview()

    def _refresh_day_options(self):
        try:
            yyyy = int(self.var_year.get())
            mm = int(self.var_month.get())
        except ValueError:
            return

        max_day = calendar.monthrange(yyyy, mm)[1]
        day_values = [f"{d:02d}" for d in range(1, max_day + 1)]
        self.cmb_day["values"] = day_values

        try:
            current_dd = int(self.var_day.get())
        except ValueError:
            current_dd = 1

        if current_dd > max_day:
            current_dd = max_day

        self.var_day.set(f"{current_dd:02d}")

    def _get_inputs(self):
        camera = self.var_camera.get().strip()
        size = self.var_size.get().strip()
        fmt = self.var_format.get().strip()
        yyyy = int(self.var_year.get())
        mm = int(self.var_month.get())
        dd = int(self.var_day.get())

        _ = date(yyyy, mm, dd)  # validate date

        if camera not in self.camera_options:
            raise ValueError("Invalid camera type.")
        if size not in self.size_options:
            raise ValueError("Invalid image size.")
        if fmt not in self.format_options:
            raise ValueError("Invalid image format.")

        return camera, size, fmt, yyyy, mm, dd

    def _set_url_text(self, url: str):
        self.txt_url.delete("1.0", "end")
        self.txt_url.insert("1.0", url)

    def _get_url_text(self) -> str:
        return self.txt_url.get("1.0", "end").strip()

    def _set_url_preview(self):
        try:
            camera, size, fmt, yyyy, mm, dd = self._get_inputs()
            url = build_url(camera, size, fmt, yyyy, mm, dd)
            self._set_url_text(url)
        except Exception:
            pass

    def _set_files(self, filenames: list[str]):
        self.lst_files.delete(0, "end")
        for name in filenames:
            self.lst_files.insert("end", name)

    def _set_status(self, text: str):
        self.var_status.set(text)
        self.update_idletasks()

    # --- Button handlers ---
    def on_configure(self):
        try:
            camera, size, fmt, yyyy, mm, dd = self._get_inputs()
            url = build_url(camera, size, fmt, yyyy, mm, dd)
            self._set_url_text(url)
        except Exception as ex:
            messagebox.showerror("Configure error", str(ex))

    def on_list(self):
        try:
            url = self._get_url_text()
            if not url:
                camera, size, fmt, yyyy, mm, dd = self._get_inputs()
                url = build_url(camera, size, fmt, yyyy, mm, dd)
                self._set_url_text(url)

            if not url.endswith("/"):
                url += "/"
                self._set_url_text(url)

            filenames = fetch_directory_listing(url)

            # Filter by expected file type
            fmt = self.var_format.get().strip().lower()
            if "jpg" in fmt:
                filenames = [f for f in filenames if f.lower().endswith((".jpg", ".jpeg"))]
            elif "fits" in fmt:
                filenames = [f for f in filenames if f.lower().endswith((".fts", ".fits"))]

            # Filter by selected size (ONLY for JPGs; FITS don't have _512/_1024)
            if "fits" not in fmt:
                size = self.var_size.get().strip()
                filenames = [f for f in filenames if filename_matches_size(f, size)]

            if not filenames:
                self._set_files([])

                # Different warning for FITS vs JPG
                if "fits" in fmt:
                    msg = "No files found for this date and FITS format."
                else:
                    size = self.var_size.get().strip()
                    msg = (f"No files match image size {size} for this date.\n\n"
                           "Try switching the image size dropdown (512 ↔ 1024) and clicking List again.")

                messagebox.showinfo("No files found", msg)
                self._set_status("0 files found")
                self.progress["value"] = 0
                self.progress["maximum"] = 1
                return

            self._set_files(filenames)
            self._set_status(f"{len(filenames)} file(s) listed")
            self.progress["value"] = 0
            self.progress["maximum"] = max(1, len(filenames))

        except requests.HTTPError as ex:
            messagebox.showerror("List error", f"HTTP error:\n{ex}")
        except requests.RequestException as ex:
            messagebox.showerror("List error", f"Network error:\n{ex}")
        except Exception as ex:
            messagebox.showerror("List error", str(ex))

    def on_load(self):
        if self._downloading:
            messagebox.showinfo("Load", "A download is already running.")
            return

        try:
            selection = self.lst_files.curselection()
            if not selection:
                messagebox.showinfo("Load", "Select one or more files first (Ctrl/Shift click, or Ctrl+A).")
                return

            camera, size, fmt, yyyy, mm, dd = self._get_inputs()

            # --- Estimate Size & Warn ---
            file_count = len(selection)
            # Rough estimates: fits ~ 2 MB, jpg ~ 380 KB
            size_per_file_mb = 2.0 if "fits" in fmt.lower() else 0.38
            total_est_mb = file_count * size_per_file_mb

            msg = (f"You are about to download {file_count} files.\n"
                   f"Estimated total size: {total_est_mb:.1f} MB\n\n"
                   "Proceed?")

            if not messagebox.askokcancel("Confirm Download", msg):
                return
            # ----------------------------

            base_url = self._get_url_text().strip()
            if not base_url:
                base_url = build_url(camera, size, fmt, yyyy, mm, dd)
                self._set_url_text(base_url)
            if not base_url.endswith("/"):
                base_url += "/"
                self._set_url_text(base_url)

            filenames = [self.lst_files.get(i) for i in selection]

            # Only validate size for JPGs (FITS don't use 512/1024 in filename)
            if "fits" not in fmt:
                size = self.var_size.get().strip()
                bad = [f for f in filenames if not filename_matches_size(f, size)]
                if bad:
                    messagebox.showerror(
                        "Make Video",
                        f"Your selection includes files that don't match the selected size ({size}).\n"
                        "Change the size dropdown or re-list and select again.\n\n"
                        f"Examples:\n" + "\n".join(bad[:5])
                    )
                    return

            subfolder = build_download_subfolder(camera, fmt, yyyy, mm, dd)
            dest_dir = DOWNLOAD_ROOT / subfolder
            dest_dir.mkdir(parents=True, exist_ok=True)

            # Kick off background download
            self._downloading = True
            self._set_buttons_enabled(False)
            self.progress["value"] = 0
            self.progress["maximum"] = max(1, len(filenames))
            self._set_status(f"Downloading to: {dest_dir}")

            t = threading.Thread(
                target=self._download_worker,
                args=(base_url, filenames, dest_dir),
                daemon=True
            )
            t.start()

        except Exception as ex:
            self._downloading = False
            self._set_buttons_enabled(True)
            messagebox.showerror("Load error", str(ex))

    def on_make_video(self):
        """
        Build a video from the currently selected files (UI order),
        using files that already exist in the download folder.
        """
        if self._downloading:
            messagebox.showinfo("Make Video", "A download is currently running. Try again when it's finished.")
            return

        try:
            selection = self.lst_files.curselection()
            if not selection:
                messagebox.showinfo("Make Video", "Select one or more files first (Ctrl/Shift click, or Ctrl+A).")
                return

            camera, size, fmt, yyyy, mm, dd = self._get_inputs()
            filenames = [self.lst_files.get(i) for i in selection]
            fps = int(self.var_fps.get())

            # Only validate size for JPGs
            if "fits" not in fmt:
                size = self.var_size.get().strip()
                bad = [f for f in filenames if not filename_matches_size(f, size)]
                if bad:
                    messagebox.showerror(
                        "Make Video",
                        f"Your selection includes files that don't match the selected size ({size}).\n"
                        "Change the size dropdown or re-list and select again.\n\n"
                        f"Examples:\n" + "\n".join(bad[:5])
                    )
                    return

            subfolder = build_download_subfolder(camera, fmt, yyyy, mm, dd)
            dest_dir = DOWNLOAD_ROOT / subfolder

            if not dest_dir.exists():
                messagebox.showerror(
                    "Make Video",
                    f"Download folder does not exist yet:\n{dest_dir}\n\nDownload files first (Load), then make video."
                )
                return

            # Background worker so GUI doesn't freeze
            self._set_buttons_enabled(False)
            self._set_status("Creating video...")
            t = threading.Thread(
                target=self._video_worker,
                args=(dest_dir, filenames, camera, fmt, yyyy, mm, dd, fps),
                daemon=True
            )
            t.start()

        except Exception as ex:
            self._set_buttons_enabled(True)
            messagebox.showerror("Make Video error", str(ex))

    def _video_worker(self, dest_dir: Path, filenames: list[str],
                      camera: str, fmt: str, yyyy: int, mm: int, dd: int, fps: int):
        try:
            # Choose a default fps for now (can become a dropdown later)
            out = make_video_from_download(dest_dir, filenames, camera, fmt, yyyy, mm, dd, fps=fps)

            def finish_ok():
                self._set_buttons_enabled(True)
                self._set_status(f"Video created: {out.name}")
                messagebox.showinfo("Make Video", f"Created video:\n{out}")

            self.after(0, finish_ok)

        except subprocess.CalledProcessError as ex:
            def finish_fail():
                self._set_buttons_enabled(True)
                self._set_status("Video creation failed")
                messagebox.showerror(
                    "ffmpeg failed",
                    "ffmpeg returned an error.\n\n"
                    "Common causes:\n"
                    "- ffmpeg not installed or not on PATH\n"
                    "- some frames missing / unreadable\n\n"
                    f"Details:\n{ex}"
                )

            self.after(0, finish_fail)

        except Exception as ex:
            def finish_fail2():
                self._set_buttons_enabled(True)
                self._set_status("Video creation failed")
                messagebox.showerror("Make Video error", str(ex))

            self.after(0, finish_fail2)

    def _download_worker(self, base_url: str, filenames: list[str], dest_dir: Path):
        ok = 0
        skipped = 0
        failed: list[str] = []

        for idx, name in enumerate(filenames, start=1):
            file_url = urljoin(base_url, name)
            dest_path = dest_dir / name

            # Skip if already present
            if dest_path.exists() and dest_path.stat().st_size > 0:
                skipped += 1
                self.after(0, self._update_progress_ui, idx, len(filenames), f"Skipped (exists): {name}")
                continue

            try:
                download_file(file_url, dest_path)
                ok += 1
                self.after(0, self._update_progress_ui, idx, len(filenames), f"Downloaded: {name}")
            except Exception:
                failed.append(name)
                self.after(0, self._update_progress_ui, idx, len(filenames), f"FAILED: {name}")

        # Done
        def finish():
            self._downloading = False
            self._set_buttons_enabled(True)
            summary = f"Done. Downloaded: {ok}, Skipped: {skipped}, Failed: {len(failed)}"
            self._set_status(summary)
            if failed:
                messagebox.showwarning("Download finished (some failed)", summary + "\n\nFailed:\n" + "\n".join(failed))
            else:
                messagebox.showinfo("Download finished", summary)

        self.after(0, finish)

    def _update_progress_ui(self, idx: int, total: int, status: str):
        self.progress["value"] = idx
        self._set_status(f"{idx}/{total} - {status}")


if __name__ == "__main__":
    app = CameraUrlTool()
    app.mainloop()