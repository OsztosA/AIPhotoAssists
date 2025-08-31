import os
import argparse
import base64
import requests
import json
import piexif
import concurrent.futures
import time
import sys

# --- Configuration ---
# IMPORTANT: Change this URL to your local model's endpoint.
API_URL = "http://localhost:1234/v1/chat/completions"
# Supported image extensions for EXIF writing
JPEG_EXTENSIONS = ('.jpg', '.jpeg')


def encode_image_to_base64(image_path):
    """Reads an image and encodes it into a base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except IOError as e:
        print(f"Error reading image file {image_path}: {e}")
        return None


def get_tags_from_llm(image_path):
    """
    Sends an image to the local LLM endpoint and asks for metadata tags.
    """
    base64_image = encode_image_to_base64(image_path)
    if not base64_image:
        return None

    headers = {"Content-Type": "application/json"}
    prompt = """
    Act as a photo assistant. Analyze this image and generate metadata. Provide a concise title, a one-sentence description, and a list of relevant keywords.
    Respond with ONLY in the following format:
    {"title": "...", "description": "...", "keywords": ["tag1", "tag2", "tag3"]}
    """

    payload = {
        "model": "local-model",  # This might need to be adjusted

        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                ]
            }
        ],
        "max_tokens": 4096
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        content = response.json()['choices'][0]['message']['content']
        tags = json.loads(content)
        # Basic validation of the returned structure
        if 'title' in tags and 'description' in tags and 'keywords' in tags:
            return tags
        else:
            print(f"Warning: LLM response for {os.path.basename(image_path)} is missing required keys. Response: {content}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"API request failed for {os.path.basename(image_path)}: {e}")
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        print(f"Could not parse API response for {os.path.basename(image_path)}: {e}")
    return None


def write_exif_tags(image_path, tags, dry_run=False):
    """Writes the given tags to the image's EXIF data."""
    try:
        # piexif requires keywords to be a semicolon-separated string, encoded in UTF-16LE.
        keywords_str = ";".join(tags.get("keywords", []))
        # The bytes must be null-terminated.
        keywords_bytes = keywords_str.encode('utf-16le') + b'\x00\x00'

        exif_dict = {
            "0th": {
                piexif.ImageIFD.ImageDescription: tags.get("description", "").encode('utf-8')
            },
            "Exif": {},
            "GPS": {},
            "1st": {},
            "thumbnail": None
        }
        # Windows specific tags for Title and Keywords
        exif_dict["0th"][piexif.ImageIFD.XPTitle] = tags.get("title", "").encode('utf-16le') + b'\x00\x00'
        exif_dict["0th"][piexif.ImageIFD.XPKeywords] = keywords_bytes

        exif_bytes = piexif.dump(exif_dict)

        if dry_run:
            print(f"[Dry Run] Would write tags to {image_path}: Title='{tags.get('title')}', Keywords='{keywords_str}'")
        else:
            piexif.insert(exif_bytes, image_path)
            print(f"  -> Successfully wrote tags to {os.path.basename(image_path)}")

    except Exception as e:
        print(f"Error writing EXIF data to {image_path}: {e}")


def process_single_image(image_path, dry_run):
    """
    Gets tags and writes them to a single image file.
    Returns a status message.
    """
    tags = get_tags_from_llm(image_path)
    if tags:
        # This function now prints its own status, so we don't need to return one.
        write_exif_tags(image_path, tags, dry_run)
    else:
        # Print to a new line to avoid overwriting the progress bar
        print(f"\nCould not get tags for {os.path.basename(image_path)}. Skipping.")


def process_images_in_directory(root_path, max_workers=5, dry_run=False):
    """Uses a thread pool to find and tag JPEG images in a directory, showing progress."""
    if not os.path.isdir(root_path):
        print(f"Error: Directory not found at '{root_path}'")
        return

    if dry_run:
        print("--- Running in Dry Run mode. No files will be modified. ---")

    print(f"Scanning for images in '{root_path}'...")
    image_paths = []
    for dirpath, _, filenames in os.walk(root_path):
        for filename in filenames:
            if filename.lower().endswith(JPEG_EXTENSIONS):
                image_paths.append(os.path.join(dirpath, filename))

    total_images = len(image_paths)
    if total_images == 0:
        print("No JPEG images found to process.")
        return

    print(f"Found {total_images} images. Starting tagging with {max_workers} workers...")

    processed_count = 0
    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {executor.submit(process_single_image, path, dry_run): path for path in image_paths}

        for future in concurrent.futures.as_completed(future_to_path):
            original_path = future_to_path[future]
            try:
                future.result()
            except Exception as e:
                print(f"\nAn error occurred while processing {original_path}: {e}")

            processed_count += 1
            elapsed_time = time.time() - start_time
            images_per_sec = processed_count / elapsed_time if elapsed_time > 0 else 0

            progress_message = f"Progress: {processed_count}/{total_images} | Rate: {images_per_sec:.2f} images/sec"
            sys.stdout.write(f"\r{progress_message.ljust(60)}")
            sys.stdout.flush()

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
        description="Generate and write EXIF tags to JPEG images using a local LLM."
    )
    parser.add_argument(
        "directory",
        type=str,
        help="The root directory path to search for JPEG images."
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
    args = parser.parse_args()
    process_images_in_directory(args.directory, args.workers, args.dry_run)