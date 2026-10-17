import os
import argparse
import requests
import shutil
import re
import concurrent.futures
import time

from typing import Optional
from JpegImage import JpegImage
from LLM import LLM
from ResponseParser import LLMResponseParser

# Supported image extensions
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
API_URL = "http://localhost:1234/v1/chat/completions"
PROMPT = """You are an expert photo editor helping to cull a large personal photo library.
Rate the overall quality and keep-worthiness of the attached photo as a single integer from 0 to 100.

Evaluate these criteria:
1. Technical quality (about 40%): sharpness and correct focus on the subject, absence of motion blur and camera shake, correct exposure (no blown highlights or crushed shadows), natural colors, low noise.
2. Composition (about 25%): framing, subject placement, balance, level horizon, no distracting elements or awkwardly cropped subjects.
3. Subject and emotion (about 35%): photos of family members, friends, children, and meaningful moments or events (smiles, interaction, milestones, travel memories) are highly valuable. People should have open eyes and pleasant expressions. A technically imperfect photo of an important family moment still deserves a decent score.

Scoring guide:
- 0-10: unusable (black/blank frame, accidental shot, heavily blurred, pocket photo, finger over lens)
- 11-30: poor (out of focus, badly exposed or badly composed, no meaningful content)
- 31-50: mediocre (acceptable snapshot, forgettable or near-duplicate quality)
- 51-70: good (sharp, decent composition, pleasant content)
- 71-85: very good (strong composition, great light, or an important family/people moment)
- 86-100: outstanding (professional-quality, or an irreplaceable family memory that is also well captured)

Be critical and use the full range; do not give every photo a score in the middle.
Respond with ONLY the integer score, with no text, explanation, punctuation, or symbols. Example of a valid response: 72"""


def _format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    else:
        return f"{m:02d}:{s:02d}"


def classify_image(image_path: str) -> Optional[int]:
    """
    Sends an image to the local LLM endpoint for classification and returns a score.
    """
    llm = LLM(API_URL)
    try:

        jpeg_image = JpegImage(image_path)
        response = llm.call_llm(jpeg_image, PROMPT)
        if response is None:
            return None

        # Clean content and extract first integer score 0-100
        score = LLMResponseParser.extract_first_int_in_range(response, 0, 100)
        if score is not None:
            return score
        else:
            print(
                f"Warning: Could not parse score from response for {os.path.basename(image_path)}. Response: '{response}'")

    except requests.exceptions.RequestException as e:
        print(f"API request failed for {os.path.basename(image_path)}: {e}")
    except (KeyError, IndexError) as e:
        print(f"Could not parse API response for {os.path.basename(image_path)}: Invalid format. {e}")

    return None


def process_single_image(original_path: str, root_path: str, output_directory: str, dry_run: bool = False):
    """
    Classifies a single image and moves it. Designed to be run in a thread pool.
    """
    # Skip files that might have been processed by a previous version of the script
    if re.match(r'^\d{3}__', os.path.basename(original_path)):
        print(f"Skipping file with score prefix: {original_path}")
        return

    print(f"Processing: {original_path}")
    score = classify_image(original_path)

    if score is not None:
        try:
            # Format score with leading zeros (e.g., 85 -> "085")
            score_folder_name = f"{score:03d}"

            # Determine the relative path from the original root
            dirpath = os.path.dirname(original_path)
            relative_dir = os.path.relpath(dirpath, root_path)

            # Create the full destination directory path: output/score/relative_path
            dest_dir = os.path.join(output_directory, score_folder_name, relative_dir)
            os.makedirs(dest_dir, exist_ok=True)

            # The new path for the file, keeping the original filename
            new_path = os.path.join(dest_dir, os.path.basename(original_path))

            if os.path.exists(new_path):
                print(f"Skipping move: '{new_path}' already exists.")
                return

            if not dry_run:
                shutil.move(original_path, new_path)
            print(f"  -> {score_folder_name} - Moved to: {new_path}")
        except Exception as e:
            print(f"Error moving file {original_path}: {e}")
    else:
        print(f"  -> Could not get score for {original_path}. Skipping.")


def process_images_in_directory(root_path: str, output_directory: str, max_workers: int = 5, dry_run: bool = False):
    """
    Iterates through a directory, and uses a thread pool to classify and move images, showing progress.
    """
    if not os.path.isdir(root_path):
        print(f"Error: Input directory not found at '{root_path}'")
        return

    if not output_directory:
        print("Error: An output directory must be specified for this operation.")
        return

    print(f"Scanning for images in '{root_path}'...")
    image_paths: list[str] = []
    for dirpath, _, filenames in os.walk(root_path):
        for filename in filenames:
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                image_paths.append(os.path.join(dirpath, filename))

    total_images = len(image_paths)
    if total_images == 0:
        print("No images found to process.")
        return

    print(f"Found {total_images} images. Starting processing with {max_workers} workers...")
    print(f"Output will be saved to: '{output_directory}'")

    processed_count = 0
    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a dictionary to map futures to their original paths for better error reporting
        future_to_path = {executor.submit(process_single_image, path, root_path, output_directory, dry_run): path for path in image_paths}

        for future in concurrent.futures.as_completed(future_to_path):
            original_path = future_to_path[future]
            try:
                future.result()  # We call result() to raise any exceptions from the thread
            except Exception as e:
                # Print exceptions on a new line so they don't mess up the progress bar
                print(f"\nAn error occurred while processing {original_path}: {e}")

            processed_count += 1
            elapsed_time = time.time() - start_time
            avg_rate = processed_count / elapsed_time if elapsed_time > 0 else 0.0
            remaining = total_images - processed_count
            eta_seconds = (remaining / avg_rate) if avg_rate > 0 else 0.0
            avg_time_img = (elapsed_time / processed_count) if processed_count > 0 else 0.0

            # Display progress on a single, updating line
            progress_message = (
                f"Progress: {processed_count}/{total_images} | "
                f"Avg: {avg_rate:.2f} img/s | "
                f"Avg time/img: {avg_time_img:.2f}s | "
                f"ETA: {_format_seconds(eta_seconds)}"
            )
            print(f"\r{progress_message.ljust(120)}", end="", flush=True)

    # Final summary
    end_time = time.time()
    total_duration = end_time - start_time
    average_rate = total_images / total_duration if total_duration > 0 else 0
    average_time_per_image = (total_duration / total_images) if total_images > 0 else 0.0

    print()  # Move to the next line after the progress bar
    print("\n--- Processing Summary ---")
    print(f"Total images processed: {total_images}")
    print(f"Total time taken: {total_duration:.2f} seconds")
    print(f"Average processing rate: {average_rate:.2f} images/sec")
    print(f"Average time per image: {average_time_per_image:.2f} seconds")
    print("Processing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify images in a directory using a local LLM and move them to a scored directory structure."
    )
    parser.add_argument(
        "directory",
        type=str,
        help="The root directory path to search for images."
    )
    parser.add_argument(
        "-o", "--output",
        dest="output_directory",
        type=str,
        required=True,
        help="The output directory to move classified images to. Preserves subfolder structure."
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=5,
        help="Number of concurrent threads to use for processing images. Default is 5."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the script without moving any files. Prints the actions that would be taken."
    )
    args = parser.parse_args()
    process_images_in_directory(args.directory, args.output_directory, args.workers, args.dry_run)