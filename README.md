# Image Processing Scripts with Local LLM

## Overview
I have a massive library of photos and simply don't have the time to examine them one by one to delete the bad ones. These scripts were created to automate the process of rating and tagging images, making them easier to manage. By generating quality scores and descriptive metadata, I can use powerful tools like DigiKam, which can read EXIF tags and search by content, to quickly filter, find, and organize my entire collection.
 Python scripts designed to process a large number of images using a local Large Language Model (LLM).

-   `classify_and_move.py`: Classifies images by quality and copies (or moves) them into a new directory structure based on their score. Screenshots, document photos and other throw-away shots go to a separate `useless` folder.
-   `tag_images.py`: Generates descriptive metadata for images and writes it to their EXIF tags.
-   HEIC/HEIF images are supported for classification and tagging. They are converted to JPEG for LLM requests, capture dates are read from embedded EXIF metadata, and ExifTool writes tags directly without re-encoding the image.

## Common Prerequisite

Both scripts require a running local LLM with an API endpoint that can process images (multimodal). You must edit the API_URL variable in each script to point to your model's endpoint.
Writing metadata to HEIC/HEIF files also requires [ExifTool](https://exiftool.org/) (download it from https://exiftool.org/). In the GUI, browse to `exiftool.exe` in the *ExifTool executable* field; leave it blank to use the executable on `PATH`. Both CLI tools accept `--exiftool-path <path>` for its location. The standalone GUI requires ExifTool to be installed separately.

**IMPORTANT: Change this URL to your local model's endpoint**

```python
API_URL = "http://localhost:1234/v1/chat/completions"
```

## Disclaimer

**Use these scripts at your own risk.** They are designed to modify and move files on your system. The developer assumes no responsibility for any data loss or damage that may occur. It is strongly recommended that you back up your files before running these scripts, especially for the first time. To protect your privacy, it is crucial to **use these scripts exclusively with local Large Language Models (LLMs)**. Sending personal images to third-party cloud services could expose your private data, so only a local LLM endpoint should be used. 
    
---

## 1. Image Classifier and Mover `classify_and_move.py`

This script recursively scans a directory for images, asks an LLM to rate each image's quality on a scale of 0-100, and then copies (default) or moves the image into a new directory structure organized by that score. Photos that are not worth keeping forever are put into a `useless` folder instead.

### Features
*   Quality Scoring: Uses an LLM to assign a quality score from 0 to 100.
*   File Organization: Copies files to `output_directory/score/original_subdirectory/filename.ext`. For example, an image scored 85 from `C:\Photos\Vacation\img1.jpg` would be copied to `output\085\Vacation\img1.jpg`. Without `--copy` the command line tool moves the files instead (see below).
*   Useless Category: The LLM also decides whether a photo is *useless* - no lasting or emotional value, e.g. screenshots, photos of documents, receipts, tickets, whiteboards or things photographed only to remember them for a short time. Such photos are not scored; they go to `output_directory/useless/original_subdirectory/filename.ext`. Photos of people, pets, travel or events are never useless, even if they are technically poor (they just get a low score).
*   Copy or Move: Files are copied by default (originals are kept); moving removes the originals.
*   Rename: Optionally renames the files from their EXIF capture date, e.g. `P1420703.JPG` -> `2014.03.01 11.48.00 - P1420703.JPG`.
* Videos: Optionally rename video files using their file modification date and place them under `output_directory/videos/`, preserving their source subfolders. If the file date cannot be read, the original filename is kept. Videos are copied or moved according to the selected file operation, and are never sent to the LLM or tagged.
*   Constrained Output: Requests use grammar-constrained decoding (a JSON schema with `useless`, `score` and `tags`, temperature 0), so the model can only answer in the expected format. If a server rejects the schema, the request is retried once without it.
*   Concurrent Processing: Uses a thread pool to process multiple images at once, significantly speeding up the process. 
*   Real-time Metrics: Displays a live progress bar showing the processing rate in images/sec.


### Installation
```shell   
pip install -r requirements.txt
```

### Usage
Run the script from your terminal, providing the source directory and a destination directory.
```shell
python classify_and_move.py <source_directory> --output <destination_directory> [--workers <number>] [--max-dimension <pixels>] [--tag-count <number>] [--copy] [--rename [PATTERN]] [--rename-videos] [--exiftool-path <path>] [--dry-run]
```

#### Arguments
- `source_directory`: The root directory to search for images.
- `-o, --output`: (Required) The base directory where the scored folder structure (and the `useless` folder) will be created.
- `-w, --workers`: (Optional) The number of concurrent threads to use. Defaults to 5.
- `-m, --model`: (Optional) Model name sent to the API. Defaults to `local-model`.
- `--api-url`: (Optional) Chat completions endpoint of the LLM server.
- `--exiftool-path`: (Optional) Path to `exiftool.exe`; otherwise it must be available on `PATH`.
- `--no-tags`: (Optional) Only ask the LLM for a score, not for tags.
- `--max-dimension`: (Optional) Limit the longest image side to this many pixels before sending it to the LLM. The other side is scaled to preserve the aspect ratio; images already within the limit are unchanged. Defaults to 1024 pixels. `--resize` is retained as an alias.
- `--tag-count`: (Optional) The number of tags the LLM is asked to generate. Defaults to 5.
- `--copy`: (Optional) Copy the images instead of moving them (the originals are kept). Without this flag the command line tool moves the files; the GUI copies by default.
- `--rename [PATTERN]`: (Optional) Rename the files using their EXIF capture date. Without a value the pattern is `YYYY.MM.DD HH24.MI.SS - %NAME%`. Tokens: `YYYY`, `MM`, `DD`, `HH24`, `MI`, `SS` (date and time) and `%NAME%` (original file name without extension). The extension is kept, and `:` is replaced by `.` because Windows does not allow it in file names. Files without an EXIF date keep their name.
- `--rename-videos`: (Optional) Rename supported video files using their filesystem modification date and the same filename pattern, then copy or move them into `output/videos/` while preserving source subfolders. If the file date cannot be read, the original filename is kept. Videos are not classified or tagged. Existing files at the destination are skipped.
- `--dry-run`: (Optional) Do not copy or move any files.

See [Graphical interface](#graphical-interface-guipy) for a GUI with the same options (including the image size slider and the number of tags).

#### Example
```shell
python classify_and_move.py "C:\Users\MyUser\Pictures" --output "D:\SortedPics" --workers 5

# Limit the longest side to 1024 pixels and ask for 8 tags
python classify_and_move.py "C:\Users\MyUser\Pictures" --output "D:\SortedPics" --max-dimension 1024 --tag-count 8

# Copy (keep the originals) and rename to "2014.03.01 11.48.00 - P1420703.JPG"
python classify_and_move.py "C:\Users\MyUser\Pictures" --output "D:\SortedPics" --copy --rename
```

---

## Graphical interface `gui.py`
A Tkinter front end for `classify_and_move.py`: configure everything in one window, run it and watch the progress live. It uses the same code as the command line tool.

### Running
```shell
pip install -r requirements.txt
python gui.py
```
Tkinter ships with the standard Python installer. Pillow is optional but needed for the live image previews.

### Using it
1. **Folders**: choose the *Source* folder (searched recursively) and the *Target* folder (the scored folder structure and the `useless` folder are created there). They must be different.
2. **Model**: pick the *Backend* (OpenAI-compatible such as LM Studio, or Ollama), check the *API URL* and select the *Model* (click *Refresh models* to reload the list; only vision models can be used). Click *Detect* to find the maximum number of parallel requests the model supports; Start uses the selected worker count without running detection.
3. **File operation**: *Copy* (default) keeps the originals and copies the images to the target folder. *Move* (marked with a warning sign) removes the originals; when dry run is off you are asked to confirm before files are moved.
4. **Rename**: tick *Rename image files* to rename images from their EXIF capture date. Tick *Rename video files* to rename videos using their filesystem modification date and put them in `target/videos/`, preserving source subfolders. Videos are never classified or tagged. Both use the pattern in the text box (default `YYYY.MM.DD HH24.MI.SS - %NAME%`); tokens are `YYYY MM DD HH24 MI SS` and `%NAME%`, and `:` becomes `.`.
5. **Options**:
   - *Dry run*: nothing is copied or moved, the results are only shown. It is on by default.
   - *Ask LLM for tags*: also request descriptive tags, shown in the Results tab.
   - *Workers*: number of parallel requests (limited to the detected maximum of the model).
   - *Benchmark*: processes up to the specified number of images in a forced dry run, regardless of the regular dry-run setting. Each completed benchmark is appended to `benchmark_results.csv` next to the application, with timestamps, elapsed/average timings, throughput, score histogram and run options.
6. **LLM request**:
   - *Tags to generate*: how many tags the LLM is asked for (default 5).
   - *Maximum image dimension*: limits the longest side sent to the LLM (default 1024 pixels). The other side is scaled to preserve the aspect ratio; images already within the limit are sent unchanged.
7. Press **Start**; **Stop** cancels the run after the in-flight requests finish.

While running you see a progress bar with rate and ETA, thumbnails of the images being processed, and two tabs:
- **Results**: a score histogram (with an extra grey bar for useless photos) and a table (file, score, tags, result, destination). Useless photos show `useless` in the score column. Click a column heading to sort, double-click a row to open the file.

The progress area shows the running counts of copied/moved, useless, skipped and failed images.
- **Log**: messages and errors.

All settings are saved automatically to `gui_settings.json` next to the script and restored on the next start.

### Building a standalone executable
To get a single `PhotoAssist.exe` that runs without a Python installation (Windows):
```shell
build.bat
```
This installs the dependencies from `requirements-build.txt` (including PyInstaller) and writes `dist\PhotoAssist.exe`. Use `build.bat clean` to remove previous build output first. The exe stores `gui_settings.json` next to itself.

---

## 2. EXIF Tagger `tag_images.py`
This script recursively scans a directory for JPEG and HEIC/HEIF images, asks an LLM to generate a title, description, and keywords for each image, and writes this information directly into the image's EXIF metadata.

### Features
*   AI-Powered Tagging: Uses an LLM to generate a relevant title, a one-sentence description, and a list of keywords.
*   Direct EXIF Writing: Modifies the image file to embed the generated metadata. HEIC files are updated in place with ExifTool, without re-encoding.
*   Concurrent Processing: Uses a thread pool for fast, parallel processing.
*   Real-time Metrics: Displays a live progress bar with the processing rate in images/sec.
*   Safe Mode: Includes a --dry-run flag to preview the tags that would be written without modifying any files.

### Installation
```shell  
pip install -r requirements.txt
```

### Usage
Run the script from your terminal, providing the directory containing the images you want to tag.
```shell
python tag_images.py <source_directory> [--workers <number>] [--exiftool-path <path>] [--dry-run]
```

### Arguments
- `source_directory`: The root directory to search for .jpg, .jpeg, .heic, and .heif images.
- `-w, --workers`: (Optional) The number of concurrent threads to use. Defaults to 5.
- `--dry-run`: (Optional) Run the script in simulation mode. It will print the tags it would have written but will not change any files. 
- `--exiftool-path`: (Optional) Path to `exiftool.exe`; otherwise it must be available on `PATH`.

### Examples
```shell
# Tag all images in the "MyAlbum" folder with 5 workers
python tag_images.py "C:\Users\MyUser\Pictures\MyAlbum" --workers 5

# See what tags would be generated without changing files
python tag_images.py "C:\Users\MyUser\Pictures\MyAlbum" --dry-run
```
  

## Tested with

- Python 3.10+
- Windows 11
- LM Studio 0.3.24 [Download](https://lmstudio.ai/)
- qwen/qwen3-vl-8b [Download](https://lmstudio.ai/models/qwen/qwen3-vl-8b)
- AMD Ryzen 7 7700X, 32GB RAM
- AMD Radeon RX 7900XT
- Important: use vulcan runtime with 4K context, turn off Unified KV cache 