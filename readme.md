# SOHO LASCO Image Downloader & Video Builder

A Python GUI tool for browsing, downloading, and creating videos from **SOHO LASCO C2/C3** coronagraph images (JPG or FITS).

This tool allows you to:
- Select camera (C2 / C3), image size (512 / 1024), format (JPG / FITS), and date
- List available images directly from NASA servers
- Download a selected subset (or all) images
- Create an MP4 video from the downloaded images
- Control playback speed via FPS selection
- Automatically organize downloads into date-based folders

---

## Features

- Tkinter-based GUI (no web browser needed)
- Supports:
  - JPG images from `soho.nascom.nasa.gov`
  - FITS images from `umbra.nascom.nasa.gov`
- Automatic filtering by:
  - Image format
  - Image size (512 / 1024)
- Multi-file selection
- Background downloading (UI remains responsive)
- Optional FITS → PNG conversion for video creation
- ffmpeg-based video creation
- Deterministic folder and video naming

---

## Folder Structure

Downloads are saved under:
downloads/
└── YYYYMMDD_LASCO_cX_format/
├── image files...
├── _png_frames/ (FITS only)
├── ffmpeg_list.txt
└── YYYYMMDD_LASCO_cX_format_10fps.mp4
