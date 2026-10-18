import base64
import io
import os
from datetime import datetime
from typing import Optional

import exiftool
import piexif
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener(thumbnails=False)

HEIF_EXTENSIONS = ('.heic', '.heif', '.heics', '.heifs', '.hif')
EXIFTOOL_PATH: Optional[str] = None


def set_exiftool_path(path: Optional[str]) -> None:
    """Set the optional ExifTool executable path; None uses the executable on PATH."""
    global EXIFTOOL_PATH
    EXIFTOOL_PATH = os.path.abspath(os.path.expanduser(path.strip())) if path and path.strip() else None


def _exiftool_helper() -> exiftool.ExifToolHelper:
    return exiftool.ExifToolHelper(executable=EXIFTOOL_PATH)


def _empty_exif_dict() -> dict:
    return {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}


class JpegImage:
    def __init__(self, path):
        self.path = path

    def get_as_base64(self, max_dimension: int = 1024):
        """Returns the image as base64, limiting its longest side to max_dimension pixels."""
        try:
            with Image.open(self.path) as im:
                if max_dimension < 1:
                    raise ValueError("Maximum image dimension must be at least 1 pixel.")
                if max(im.size) <= max_dimension and im.format == "JPEG":
                    with open(self.path, "rb") as image_file:
                        return base64.b64encode(image_file.read()).decode('utf-8')
                if max(im.size) > max_dimension:
                    im.thumbnail((max_dimension, max_dimension), Image.LANCZOS)
                im = im.convert("RGB")
                buf = io.BytesIO()
                im.save(buf, format="JPEG", quality=90)
                return base64.b64encode(buf.getvalue()).decode('utf-8')
        except (IOError, OSError, ValueError) as e:
            print(f"Error reading image file {self.path}: {e}")
            return None

    def _load_exif(self) -> dict:
        if os.path.splitext(self.path)[1].lower() in HEIF_EXTENSIONS:
            with Image.open(self.path) as im:
                exif_bytes = im.info.get("exif")
            return piexif.load(exif_bytes) if exif_bytes else _empty_exif_dict()
        return piexif.load(self.path)

    def get_exif_datetime(self) -> Optional[datetime]:
        """The capture date from EXIF (DateTimeOriginal, then DateTimeDigitized, then DateTime), or None."""
        try:
            exif_data = self._load_exif()
        except Exception:
            return None
        candidates = (
            exif_data.get("Exif", {}).get(piexif.ExifIFD.DateTimeOriginal),
            exif_data.get("Exif", {}).get(piexif.ExifIFD.DateTimeDigitized),
            exif_data.get("0th", {}).get(piexif.ImageIFD.DateTime),
        )
        for raw in candidates:
            if not raw:
                continue
            try:
                text = raw.decode("ascii", errors="ignore") if isinstance(raw, bytes) else str(raw)
                return datetime.strptime(text.strip("\x00 "), "%Y:%m:%d %H:%M:%S")
            except ValueError:
                continue  # empty / zeroed / malformed value, try the next tag
        return None

    def has_existing_tags(self):
        try:
            if os.path.splitext(self.path)[1].lower() in HEIF_EXTENSIONS:
                with _exiftool_helper() as tool:
                    metadata = tool.get_metadata(self.path)
                if not metadata:
                    return False
                image_metadata = metadata[0]
                return bool(image_metadata.get("XMP-dc:Title") and image_metadata.get("XMP-dc:Subject"))

            exif_data = self._load_exif()
            # Check for the presence of Windows-specific Title and Keywords tags.
            # We check if the key exists and if the value is not empty/null.
            zeroth_ifd = exif_data.get("0th", {})
            has_title = piexif.ImageIFD.XPTitle in zeroth_ifd and zeroth_ifd[piexif.ImageIFD.XPTitle]
            has_keywords = piexif.ImageIFD.XPKeywords in zeroth_ifd and zeroth_ifd[piexif.ImageIFD.XPKeywords]
            return has_title and has_keywords
        except Exception:
            # If there's any error reading EXIF (e.g., no EXIF data), assume no tags.
            return False

    def write_exif_tags(self, tags, dry_run=False, score: Optional[int] = None):
        """
        Writes the given tags to the image's EXIF data, preserving the original
        file modification time and any other existing EXIF data.
        HEIF metadata is written in place as XMP using ExifTool.
        If score is given, it is stored as the first keyword in the form "Q:<score>"
        (an older "Q:" keyword is replaced).
        """
        try:
            keywords = [k for k in tags.get("keywords", []) if not str(k).startswith("Q:")]
            if score is not None:
                keywords.insert(0, f"Q:{score}")
            # piexif requires keywords to be a semicolon-separated string, encoded in UTF-16LE.
            keywords_str = ";".join(keywords)
            # The bytes must be null-terminated.
            keywords_bytes = keywords_str.encode('utf-16le') + b'\x00\x00'

            is_heif = os.path.splitext(self.path)[1].lower() in HEIF_EXTENSIONS
            if is_heif:
                if dry_run:
                    print(
                        f"[Dry Run] Would write tags to {self.path}: Title='{tags.get('title')}', "
                        f"Keywords='{keywords_str}'")
                    return

                stat_info = os.stat(self.path)
                params = ["-overwrite_original"]
                if "title" in tags:
                    params.append(f"-XMP-dc:Title={tags.get('title', '')}")
                if "description" in tags:
                    params.append(f"-XMP-dc:Description={tags.get('description', '')}")
                for index, keyword in enumerate(keywords):
                    operator = "=" if index == 0 else "+="
                    params.append(f"-XMP-dc:Subject{operator}{keyword}")
                if not keywords:
                    params.append("-XMP-dc:Subject=")
                with _exiftool_helper() as tool:
                    tool.execute(*params, self.path)
                os.utime(self.path, (stat_info.st_atime, stat_info.st_mtime))
                print(f"  -> Successfully wrote tags to {os.path.basename(self.path)}")
                return

            # Keep existing EXIF (capture date, camera, ...) instead of replacing it
            try:
                exif_dict = self._load_exif()
            except Exception:
                exif_dict = _empty_exif_dict()

            if "description" in tags:
                exif_dict["0th"][piexif.ImageIFD.ImageDescription] = tags.get("description", "").encode('utf-8')
            # Windows specific tags for Title and Keywords
            if "title" in tags:
                exif_dict["0th"][piexif.ImageIFD.XPTitle] = tags.get("title", "").encode('utf-16le') + b'\x00\x00'
            exif_dict["0th"][piexif.ImageIFD.XPKeywords] = keywords_bytes

            exif_bytes = piexif.dump(exif_dict)

            if dry_run:
                print(
                    f"[Dry Run] Would write tags to {self.path}: Title='{tags.get('title')}', Keywords='{keywords_str}'")
            else:
                # Preserve original file timestamps
                stat_info = os.stat(self.path)
                original_atime = stat_info.st_atime
                original_mtime = stat_info.st_mtime

                piexif.insert(exif_bytes, self.path)

                # Restore original timestamps
                os.utime(self.path, (original_atime, original_mtime))
                print(f"  -> Successfully wrote tags to {os.path.basename(self.path)}")

        except Exception as e:
            print(f"Error writing EXIF data to {self.path}: {e}")
