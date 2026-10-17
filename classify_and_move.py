import os
import argparse
import requests
import shutil
import re
import concurrent.futures
import threading
import time

from typing import Callable, Optional
from JpegImage import JpegImage
from LLM import LLM
from ResponseParser import LLMResponseParser

# Supported image extensions
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
API_URL = "http://localhost:1234/v1/chat/completions"
DEFAULT_MODEL = "local-model"
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
"""

PROMPT_SCORE_ONLY = PROMPT + (
    "Respond with ONLY the integer score, with no text, explanation, punctuation, or symbols. "
    "Example of a valid response: 72")

PROMPT_WITH_TAGS = PROMPT + """
Additionally, describe the photo with 3 to 8 short, lowercase English tags (e.g. people, child, beach, sunset, birthday, food, landscape, blurry, indoor).
Respond with ONLY a JSON object in exactly this format, with no other text: {"score": 72, "tags": ["family", "beach", "sunset"]}"""


def parse_score_and_tags(response: str) -> Optional[dict]:
    """Parses either a bare number or the JSON reply of PROMPT_WITH_TAGS into {'score': int, 'tags': [str]}."""
    tags: list[str] = []
    data = LLMResponseParser.extract_json_object(response)
    if data is not None:
        raw_tags = data.get("tags", data.get("keywords", []))
        if isinstance(raw_tags, str):
            raw_tags = re.split(r"[,;]", raw_tags)
        if isinstance(raw_tags, list):
            tags = [str(t).strip().lower() for t in raw_tags if str(t).strip()]
        score_text = str(data.get("score", ""))
    else:
        m = re.search(r'"?score"?\s*[:=]\s*(\d+)', response, re.IGNORECASE)
        score_text = m.group(1) if m else response
    score = LLMResponseParser.extract_first_int_in_range(score_text, 0, 100)
    return None if score is None else {"score": score, "tags": tags}

# Callback types used by the GUI (the CLI just prints)
LogCallback = Callable[[str], None]
ProgressCallback = Callable[[dict], None]
ResultCallback = Callable[[dict], None]
StartCallback = Callable[[str], None]


def _format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    else:
        return f"{m:02d}:{s:02d}"


def models_url_from_api_url(api_url: str) -> str:
    """Derives the OpenAI-compatible '/models' URL from a chat completions URL."""
    if api_url.rstrip('/').endswith('/chat/completions'):
        return api_url.rstrip('/')[:-len('/chat/completions')] + '/models'
    return api_url.rstrip('/') + '/models'


def fetch_available_models(api_url: str = API_URL, timeout: float = 5) -> list[str]:
    """Queries the server (LM Studio / OpenAI compatible) for the list of available model ids."""
    return [m["id"] for m in fetch_model_infos(api_url, timeout)]


def fetch_model_infos(api_url: str = API_URL, timeout: float = 5) -> list[dict]:
    """
    Returns [{'id': ..., 'type': 'vlm'|'llm'|'embeddings'|None}, ...]. Uses LM Studio's /api/v0/models, which
    reports the model type (only 'vlm' models accept images), and falls back to the plain OpenAI /v1/models.
    """
    models_url = models_url_from_api_url(api_url)
    try:
        r = requests.get(models_url.replace("/v1/models", "/api/v0/models"), timeout=timeout)
        r.raise_for_status()
        return [{"id": m["id"], "type": m.get("type")} for m in r.json().get("data", []) if "id" in m]
    except (requests.RequestException, ValueError, KeyError):
        r = requests.get(models_url, timeout=timeout)
        r.raise_for_status()
        return [{"id": m["id"], "type": None} for m in r.json().get("data", []) if "id" in m]


def classify_image(image_path: str, api_url: str = API_URL, model: str = DEFAULT_MODEL,
                   log: LogCallback = print, with_tags: bool = True) -> Optional[dict]:
    """
    Sends an image to the local LLM endpoint for classification.
    Returns {'score': int, 'tags': [str]} (tags empty if not requested), or None on failure.
    """
    llm = LLM(api_url, model)
    try:
        if os.path.getsize(image_path) == 0:
            log(f"Skipping empty file: {os.path.basename(image_path)}")
            return None
        jpeg_image = JpegImage(image_path)
        response = llm.call_llm(jpeg_image, PROMPT_WITH_TAGS if with_tags else PROMPT_SCORE_ONLY)
        if response is None:
            return None

        parsed = parse_score_and_tags(response)
        if parsed is not None:
            return parsed
        else:
            log(f"Warning: Could not parse score from response for {os.path.basename(image_path)}. Response: '{response}'")

    except requests.exceptions.RequestException as e:
        log(f"API request failed for {os.path.basename(image_path)}: {e}")
    except OSError as e:
        log(f"Could not read {os.path.basename(image_path)}: {e}")
    except (KeyError, IndexError) as e:
        log(f"Could not parse API response for {os.path.basename(image_path)}: Invalid format. {e}")

    return None


def process_single_image(original_path: str, root_path: str, output_directory: str, dry_run: bool = False,
                         api_url: str = API_URL, model: str = DEFAULT_MODEL,
                         log: LogCallback = print, stop_event: Optional[threading.Event] = None,
                         with_tags: bool = True, on_start: Optional[StartCallback] = None) -> dict:
    """
    Classifies a single image and moves it. Designed to be run in a thread pool.
    Returns a dict: {'path', 'status', 'score', 'dest', 'tags'} where status is one of
    'moved', 'skipped', 'failed', 'cancelled' (score/dest may be None).
    """
    tags: list[str] = []

    def result(status: str, score: Optional[int] = None, dest: Optional[str] = None) -> dict:
        return {"path": original_path, "status": status, "score": score, "dest": dest, "tags": tags}

    if stop_event is not None and stop_event.is_set():
        return result("cancelled")

    # Skip files that might have been processed by a previous version of the script
    if re.match(r'^\d{3}__', os.path.basename(original_path)):
        log(f"Skipping file with score prefix: {original_path}")
        return result("skipped")

    log(f"Processing: {original_path}")
    if on_start:
        on_start(original_path)
    classification = classify_image(original_path, api_url, model, log, with_tags)

    if classification is None:
        log(f"  -> Could not get score for {original_path}. Skipping.")
        return result("failed")
    score, tags = classification["score"], classification["tags"]

    try:
        # Format score with leading zeros (e.g., 85 -> "085")
        score_folder_name = f"{score:03d}"

        # Determine the relative path from the original root
        dirpath = os.path.dirname(original_path)
        relative_dir = os.path.relpath(dirpath, root_path)

        # Create the full destination directory path: output/score/relative_path
        dest_dir = os.path.join(output_directory, score_folder_name, relative_dir)
        new_path = os.path.join(dest_dir, os.path.basename(original_path))

        if os.path.exists(new_path):
            log(f"Skipping move: '{new_path}' already exists.")
            return result("skipped", score, new_path)

        if not dry_run:
            os.makedirs(dest_dir, exist_ok=True)
            shutil.move(original_path, new_path)
        verb = "Would move to" if dry_run else "Moved to"
        log(f"  -> {score_folder_name} - {verb}: {new_path}" + (f"  [tags: {', '.join(tags)}]" if tags else ""))
        return result("moved", score, new_path)
    except Exception as e:
        log(f"Error moving file {original_path}: {e}")
        return result("failed", score)


def run_classification(root_path: str, output_directory: str, max_workers: int = 5, dry_run: bool = False,
                       api_url: str = API_URL, model: str = DEFAULT_MODEL,
                       log: LogCallback = print, progress: Optional[ProgressCallback] = None,
                       stop_event: Optional[threading.Event] = None,
                       on_result: Optional[ResultCallback] = None,
                       on_start: Optional[StartCallback] = None,
                       with_tags: bool = True) -> Optional[dict]:
    """
    Core worker used by both CLI and GUI. Scans root_path, classifies and moves images using a
    thread pool. Reports progress through the optional `progress` callback with a dict containing:
    done, total, rate, avg_time, eta, counts (dict of moved/skipped/failed/cancelled), finished.
    `on_result` is called once per image with the dict returned by process_single_image.
    Returns the final progress dict, or None if nothing was run.
    """
    if not os.path.isdir(root_path):
        log(f"Error: Input directory not found at '{root_path}'")
        return None

    if not output_directory:
        log("Error: An output directory must be specified for this operation.")
        return None

    log(f"Scanning for images in '{root_path}'...")
    image_paths: list[str] = []
    for dirpath, _, filenames in os.walk(root_path):
        for filename in filenames:
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                image_paths.append(os.path.join(dirpath, filename))

    total_images = len(image_paths)
    if total_images == 0:
        log("No images found to process.")
        return None

    log(f"Found {total_images} images. Starting processing with {max_workers} workers "
        f"(model: {model}, dry run: {dry_run})...")
    log(f"Output will be saved to: '{output_directory}'")

    counts = {"moved": 0, "skipped": 0, "failed": 0, "cancelled": 0}
    processed_count = 0
    start_time = time.time()

    def build_stats(finished: bool) -> dict:
        elapsed = time.time() - start_time
        rate = processed_count / elapsed if elapsed > 0 else 0.0
        remaining = total_images - processed_count
        return {
            "done": processed_count,
            "total": total_images,
            "rate": rate,
            "avg_time": (elapsed / processed_count) if processed_count > 0 else 0.0,
            "eta": (remaining / rate) if rate > 0 else 0.0,
            "elapsed": elapsed,
            "counts": dict(counts),
            "finished": finished,
        }

    if progress:
        progress(build_stats(False))

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Map futures to their original paths for better error reporting
        future_to_path = {
            executor.submit(process_single_image, path, root_path, output_directory, dry_run,
                            api_url, model, log, stop_event, with_tags, on_start): path
            for path in image_paths
        }

        for future in concurrent.futures.as_completed(future_to_path):
            original_path = future_to_path[future]
            try:
                res = future.result()  # Raises any exceptions from the thread
            except Exception as e:
                log(f"An error occurred while processing {original_path}: {e}")
                res = {"path": original_path, "status": "failed", "score": None, "dest": None, "tags": []}

            status = res["status"]
            counts[status] = counts.get(status, 0) + 1
            if status != "cancelled":
                processed_count += 1
                if on_result:
                    on_result(res)
            if progress:
                progress(build_stats(False))

    stats = build_stats(True)
    if progress:
        progress(stats)

    cancelled = stop_event is not None and stop_event.is_set()
    log("")
    log("--- Processing Summary ---" + (" (stopped by user)" if cancelled else ""))
    log(f"Images processed: {processed_count}/{total_images}")
    log(f"Moved{' (dry run: would be moved)' if dry_run else ''}: {counts['moved']}, "
        f"skipped: {counts['skipped']}, failed: {counts['failed']}")
    log(f"Total time taken: {stats['elapsed']:.2f} seconds")
    log(f"Average processing rate: {stats['rate']:.2f} images/sec")
    log(f"Average time per image: {stats['avg_time']:.2f} seconds")
    log("Processing stopped." if cancelled else "Processing complete.")
    return stats


def process_images_in_directory(root_path: str, output_directory: str, max_workers: int = 5,
                                dry_run: bool = False, api_url: str = API_URL, model: str = DEFAULT_MODEL,
                                with_tags: bool = True):
    """CLI wrapper around run_classification that shows a single-line progress bar."""

    def cli_progress(s: dict):
        if s["done"] == 0:
            return
        message = (
            f"Progress: {s['done']}/{s['total']} | "
            f"Avg: {s['rate']:.2f} img/s | "
            f"Avg time/img: {s['avg_time']:.2f}s | "
            f"ETA: {_format_seconds(s['eta'])}"
        )
        print(f"\r{message.ljust(120)}", end="", flush=True)

    run_classification(root_path, output_directory, max_workers, dry_run, api_url, model,
                       log=lambda m: print(f"\n{m}"), progress=cli_progress, with_tags=with_tags)


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
        "-m", "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model name to send to the API. Default is '{DEFAULT_MODEL}'."
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=API_URL,
        help=f"Chat completions endpoint. Default is {API_URL}."
    )
    parser.add_argument(
        "--no-tags",
        action="store_true",
        help="Only ask the LLM for a score, not for descriptive tags."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the script without moving any files. Prints the actions that would be taken."
    )
    args = parser.parse_args()
    process_images_in_directory(args.directory, args.output_directory, args.workers, args.dry_run,
                                args.api_url, args.model, not args.no_tags)
