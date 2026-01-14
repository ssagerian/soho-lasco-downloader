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
## Useage

lanch the app by cli: python.exe lasco_download.py
A GUI application will appear.

Configure the date, image forat, camera type and other fields as desired.

- Click on the configure button to setup the URL.

- Click on the list button to see the entire list of available files.

- From the list of file, select those files you wish to use. Use Shift-Click or Shift-Control.

- Click on the load button to download the selected files, becareful not to unselect the files.

- After download is completed, select the frames per second speed for the video animation, the click
on the make video button.

- The video will be found in the downloads/YYYYMMDD_LASCO_cX_format/ subfolder.


## Requirements

### Python
- Python **3.10+** recommended

### Python packages
Install dependencies with:

pip install -r requirements.txt

Paste this into `requirements.txt`:

```text
requests
astropy
numpy
pillow
```

---

### External dependency

ffmpeg must be installed and available on PATH

Verify:

ffmpeg -version

## Folder Structure

Downloads are saved under:
 
 downloads/
 
└── YYYYMMDD_LASCO_cX_format/
  
      ├── image files...
  
      ├── _png_frames/ (FITS only)
  
      ├── ffmpeg_list.txt
  
      └── YYYYMMDD_LASCO_cX_format_10fps.mp4

# Disclaimer

This project is not affiliated with NASA or ESA.
All data is retrieved from publicly available SOHO LASCO archives.
