import os
import argparse
import base64
import requests
import shutil
import re
import concurrent.futures
import time
import sys

# --- Configuration ---
# IMPORTANT: Change this URL to your local model's endpoint.
API_URL = "http://localhost:1234/v1/chat/completions"
# Supported image extensions
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')


def encode_image_to_base64(image_path):
    """Reads an image and encodes it into a base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except IOError as e:
        print(f"Error reading image file {image_path}: {e}")
        return None


def classify_image(image_path):
    """
    Sends an image to the local LLM endpoint for classification and returns a score.
    """
    base64_image = encode_image_to_base64(image_path)
    if not base64_image:
        return None

    headers = {
        "Content-Type": "application/json"
    }

    payload = {
        "model": "local-model",  # This might need to be adjusted
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "act as a photo assistant. and classify tis image on a scale from 0 to 100 where 0 means a totally bad image, out of focus, badly composed, and 100 means a perfect,"
                                " sharp well designed professional photo. Valuable family photos should get larger score. Respond with only the number."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 10
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes

        content = response.json()['choices'][0]['message']['content']

        # Use regex to find the first number (integer or float) in the response
        match = re.search(r'\d+', content)
        if match:
            score = int(match.group(0))
            if 0 <= score <= 100:
                return score
            else:
                print(f"Warning: Score '{score}' for {os.path.basename(image_path)} is out of 0-100 range.")
        else:
            print(
                f"Warning: Could not parse score from response for {os.path.basename(image_path)}. Response: '{content}'")

    except requests.exceptions.RequestException as e:
        print(f"API request failed for {os.path.basename(image_path)}: {e}")
    except (KeyError, IndexError) as e:
        print(f"Could not parse API response for {os.path.basename(image_path)}: Invalid format. {e}")

    return None


def process_single_image(original_path, root_path, output_directory):
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

            shutil.move(original_path, new_path)
            print(f"  -> Moved to: {new_path}")
        except Exception as e:
            print(f"Error moving file {original_path}: {e}")
    else:
        print(f"  -> Could not get score for {original_path}. Skipping.")


def process_images_in_directory(root_path, output_directory, max_workers=5):
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
    image_paths = []
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
        future_to_path = {executor.submit(process_single_image, path, root_path, output_directory): path for path in image_paths}

        for future in concurrent.futures.as_completed(future_to_path):
            original_path = future_to_path[future]
            try:
                future.result()  # We call result() to raise any exceptions from the thread
            except Exception as e:
                # Print exceptions on a new line so they don't mess up the progress bar
                print(f"\nAn error occurred while processing {original_path}: {e}")

            processed_count += 1
            elapsed_time = time.time() - start_time
            images_per_sec = processed_count / elapsed_time if elapsed_time > 0 else 0

            # Display progress on a single, updating line
            progress_message = f"Progress: {processed_count}/{total_images} | Rate: {images_per_sec:.2f} images/sec"
            sys.stdout.write(f"\r{progress_message.ljust(60)}") # ljust to clear previous line content
            sys.stdout.flush()

    # Final summary
    end_time = time.time()
    total_duration = end_time - start_time
    average_rate = total_images / total_duration if total_duration > 0 else 0

    print() # Move to the next line after the progress bar
    print("\n--- Processing Summary ---")
    print(f"Total images processed: {total_images}")
    print(f"Total time taken: {total_duration:.2f} seconds")
    print(f"Average processing rate: {average_rate:.2f} images/sec")
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
    args = parser.parse_args()
    process_images_in_directory(args.directory, args.output_directory, args.workers)