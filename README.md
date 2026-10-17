# Image Processing Scripts with Local LLM

## Overview
I have a massive library of photos and simply don't have the time to examine them one by one to delete the bad ones. These scripts were created to automate the process of rating and tagging images, making them easier to manage. By generating quality scores and descriptive metadata, I can use powerful tools like DigiKam, which can read EXIF tags and search by content, to quickly filter, find, and organize my entire collection.
 Python scripts designed to process a large number of images using a local Large Language Model (LLM).

-   `classify_and_move.py`: Classifies images by quality and moves them into a new directory structure based on their score.
-   `tag_images.py`: Generates descriptive metadata for images and writes it to their EXIF tags.

## Common Prerequisite

Both scripts require a running local LLM with an API endpoint that can process images (multimodal). You must edit the API_URL variable in each script to point to your model's endpoint.

**IMPORTANT: Change this URL to your local model's endpoint**

```python
API_URL = "http://localhost:1234/v1/chat/completions"
```

## Disclaimer

**Use these scripts at your own risk.** They are designed to modify and move files on your system. The developer assumes no responsibility for any data loss or damage that may occur. It is strongly recommended that you back up your files before running these scripts, especially for the first time. To protect your privacy, it is crucial to **use these scripts exclusively with local Large Language Models (LLMs)**. Sending personal images to third-party cloud services could expose your private data, so only a local LLM endpoint should be used. 
    
---

## 1. Image Classifier and Mover `classify_and_move.py`

This script recursively scans a directory for images, asks an LLM to rate each image's quality on a scale of 0-100, and then moves the image into a new directory structure organized by that score.

### Features
*   Quality Scoring: Uses an LLM to assign a quality score from 0 to 100.
*   File Organization: Moves files to `output_directory/score/original_subdirectory/filename.ext`. For example, an image scored 85 from `C:\Photos\Vacation\img1.jpg` would be moved to `output\085\Vacation\img1.jpg`. 
*   Concurrent Processing: Uses a thread pool to process multiple images at once, significantly speeding up the process. 
*   Real-time Metrics: Displays a live progress bar showing the processing rate in images/sec.


### Installation
```shell   
pip install requests pillow
```

### Usage
Run the script from your terminal, providing the source directory and a destination directory.
```shell
python classify_and_move.py <source_directory> --output <destination_directory> [--workers <number>] [--resize <25-100>] [--tag-count <number>]
```

#### Arguments
- `source_directory`: The root directory to search for images.
- `-o, --output`: (Required) The base directory where the scored folder structure will be created.
- `-w, --workers`: (Optional) The number of concurrent threads to use. Defaults to 5.
- `-m, --model`: (Optional) Model name sent to the API. Defaults to `local-model`.
- `--api-url`: (Optional) Chat completions endpoint of the LLM server.
- `--no-tags`: (Optional) Only ask the LLM for a score, not for tags.
- `--resize`: (Optional) Resize each image to this percentage (25-100) before sending it to the LLM. Smaller images are faster to process. Defaults to 100 (original file, unchanged).
- `--tag-count`: (Optional) The number of tags the LLM is asked to generate. Defaults to 5.
- `--dry-run`: (Optional) Do not move any files.

See [Graphical interface](#graphical-interface-guipy) for a GUI with the same options (including the image size slider and the number of tags).

#### Example
```shell
python classify_and_move.py "C:\Users\MyUser\Pictures" --output "D:\SortedPics" --workers 5

# Send images at 50% size and ask for 8 tags
python classify_and_move.py "C:\Users\MyUser\Pictures" --output "D:\SortedPics" --resize 50 --tag-count 8
```

---

## Graphical interface `gui.py`
A Tkinter front end for `classify_and_move.py`: configure everything in one window, run it and watch the progress live. It uses the same code as the command line tool.

### Running
```shell
pip install requests pillow
python gui.py
```
Tkinter ships with the standard Python installer. Pillow is optional but needed for the live image previews.

### Using it
1. **Folders**: choose the *Source* folder (searched recursively) and the *Target* folder (the scored folder structure is created there). They must be different.
2. **Model**: pick the *Backend* (OpenAI-compatible such as LM Studio, or Ollama), check the *API URL* and select the *Model* (click *Refresh models* to reload the list; only vision models can be used). *Detect* finds the maximum number of parallel requests the model supports; it also runs automatically on Start.
3. **Options**:
   - *Dry run*: nothing is moved, the results are only shown. It is on by default; when it is off you are asked to confirm before files are moved.
   - *Ask LLM for tags*: also request descriptive tags, shown in the Results tab.
   - *Workers*: number of parallel requests (limited to the detected maximum of the model).
4. **LLM request**:
   - *Tags to generate*: how many tags the LLM is asked for (default 5).
   - *Image size sent to LLM*: slider from 25% to 100% (default 100%). The image is downscaled before it is sent, which speeds up processing and lowers memory use; 100% sends the original file.
5. Press **Start**; **Stop** cancels the run after the in-flight requests finish.

While running you see a progress bar with rate and ETA, thumbnails of the images being processed, and two tabs:
- **Results**: a score histogram and a table (file, score, tags, result, destination). Click a column heading to sort, double-click a row to open the file.
- **Log**: messages and errors.

All settings are saved automatically to `gui_settings.json` next to the script and restored on the next start.

---

## 2. EXIF Tagger `tag_images.py`
This script recursively scans a directory for JPEG images, asks an LLM to generate a title, description, and keywords for each image, and writes this information directly into the image's EXIF metadata.

### Features
*   AI-Powered Tagging: Uses an LLM to generate a relevant title, a one-sentence description, and a list of keywords.
*   Direct EXIF Writing: Modifies the image file to embed the generated metadata.
*   Concurrent Processing: Uses a thread pool for fast, parallel processing.
*   Real-time Metrics: Displays a live progress bar with the processing rate in images/sec.
*   Safe Mode: Includes a --dry-run flag to preview the tags that would be written without modifying any files.

### Installation
```shell  
pip install requests piexif
```

### Usage
Run the script from your terminal, providing the directory containing the images you want to tag.
```shell
python tag_images.py <source_directory> [--workers <number>] [--dry-run]
```

### Arguments
- `source_directory`: The root directory to search for .jpg or .jpeg images.
- `-w, --workers`: (Optional) The number of concurrent threads to use. Defaults to 5.
- `--dry-run`: (Optional) Run the script in simulation mode. It will print the tags it would have written but will not change any files. 

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
- Gemma 3 12B 8-bit [Download](https://huggingface.co/lmstudio-community/gemma-3-12b-it-GGUF)
- AMD Ryzen 7 7700X, 32GB RAM
- AMD Radeon RX 7900XT