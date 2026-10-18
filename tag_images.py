import json
import os
import argparse
import concurrent.futures
import time
import requests

from JpegImage import HEIF_EXTENSIONS, JpegImage, set_exiftool_path
from LLM import LLM
from ResponseParser import LLMResponseParser

# --- Configuration ---
# IMPORTANT: Change this URL to your local model's endpoint.
API_URL = "http://localhost:1234/v1/chat/completions"
# Supported image extensions for metadata writing
IMAGE_EXTENSIONS = ('.jpg', '.jpeg') + HEIF_EXTENSIONS

PROMPT = """
        Act as a photo assistant. Analyze this image and generate metadata. Provide a concise title, a one-sentence description, and a list of relevant keywords.
        Respond with ONLY in the following format:
        {"title": "...", "description": "...", "keywords": ["tag1", "tag2", "tag3"]}
  """


def _format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    else:
        return f"{m:02d}:{s:02d}"


def get_tags_from_llm(image: JpegImage) -> dict | None:
    """
    Sends an image to the local LLM endpoint and asks for metadata tags.
    """
    headers = {"Content-Type": "application/json"}
    llm = LLM(API_URL)
    try:

        content = llm.call_llm(image, PROMPT)
        if content is None:
            return None
        tags = LLMResponseParser.extract_json_object(content)
        # Basic validation of the returned structure
        if isinstance(tags, dict) and 'title' in tags and 'description' in tags and 'keywords' in tags:
            return tags
        else:
            print(
                f"Warning: LLM response for {os.path.basename(image.path)} is missing required keys. Response: {content}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"API request failed for {os.path.basename(image.path)}: {e}")
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        print(f"Could not parse API response for {os.path.basename(image.path)}: {e}")
    return None


def process_single_image(image: JpegImage, dry_run):
    """
    Gets tags and writes them to a single image file.
    Returns a status message.
    """

    tags = get_tags_from_llm(image)
    if tags:
        # This function now prints its own status, so we don't need to return one.
        image.write_exif_tags(tags, dry_run)
    else:
        # Print to a new line to avoid overwriting the progress bar
        print(f"\nCould not get tags for {os.path.basename(image.path)}. Skipping.")


def process_images_in_directory(root_path, max_workers=5, dry_run=False):
    """Uses a thread pool to find and tag JPEG/HEIF images in a directory, showing progress."""
    if not os.path.isdir(root_path):
        print(f"Error: Directory not found at '{root_path}'")
        return

    if dry_run:
        print("--- Running in Dry Run mode. No files will be modified. ---")

    print(f"Scanning for images in '{root_path}'...")
    all_image_paths = []
    for dirpath, _, filenames in os.walk(root_path):
        for filename in filenames:
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                all_image_paths.append(os.path.join(dirpath, filename))

    if not all_image_paths:
        print("No supported images found to process.")
        return

    print(f"Found {len(all_image_paths)} total images. Checking for existing tags...")

    images_to_process = []
    for path in all_image_paths:
        image = JpegImage(path)
        if image.has_existing_tags():
            print(f"Skipping '{os.path.basename(path)}', it already has tags.")
        else:
            images_to_process.append(image)

    total_images = len(images_to_process)
    if total_images == 0:
        print("No images need tagging.")
        return

    print(f"\nStarting tagging for {total_images} images with {max_workers} workers...")

    processed_count = 0
    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_image = {executor.submit(process_single_image, image, dry_run): image for image in images_to_process}

        for future in concurrent.futures.as_completed(future_to_image):
            original_image = future_to_image[future]
            try:
                future.result()
            except Exception as e:
                print(f"\nAn error occurred while processing {original_image}: {e}")

            processed_count += 1
            elapsed_time = time.time() - start_time
            avg_rate = processed_count / elapsed_time if elapsed_time > 0 else 0.0
            remaining = total_images - processed_count
            eta_seconds = (remaining / avg_rate) if avg_rate > 0 else 0.0
            avg_time_img = (elapsed_time / processed_count) if processed_count > 0 else 0.0

            progress_message = (
                f"Progress: {processed_count}/{total_images} | "
                f"Avg: {avg_rate:.2f} img/s | "
                f"Avg time/img: {avg_time_img:.2f}s | "
                f"ETA: {_format_seconds(eta_seconds)}"
            )
            print(f"\r{progress_message.ljust(120)}", end="", flush=True)

    end_time = time.time()
    total_duration = end_time - start_time
    average_rate = total_images / total_duration if total_duration > 0 else 0
    average_time_per_image = (total_duration / total_images) if total_images > 0 else 0.0

    print() # Move to the next line after the progress bar
    print("\n--- Processing Summary ---")
    print(f"Total images processed: {total_images}")
    print(f"Total time taken: {total_duration:.2f} seconds")
    print(f"Average processing rate: {average_rate:.2f} images/sec")
    print(f"Average time per image: {average_time_per_image:.2f} seconds")
    print("Processing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate and write EXIF tags to JPEG and HEIF images using a local LLM."
    )
    parser.add_argument(
        "directory",
        type=str,
        help="The root directory path to search for JPEG and HEIF images."
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=5,
        help="Number of concurrent threads to use. Default is 5."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the script without modifying any files. Prints the actions that would be taken."
    )
    parser.add_argument(
        "--exiftool-path",
        type=str,
        default=None,
        help="Path to exiftool.exe. If omitted, ExifTool must be available on PATH."
    )
    args = parser.parse_args()
    set_exiftool_path(args.exiftool_path)
    process_images_in_directory(args.directory, args.workers, args.dry_run)