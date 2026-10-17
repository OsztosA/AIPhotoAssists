import base64
import io
import os
from datetime import datetime
from typing import Optional

import piexif
from PIL import Image


class JpegImage:
    def __init__(self, path):
        self.path = path

    def get_as_base64(self, scale_percent: int = 100):
        """Returns the image as base64 JPEG, downscaled to scale_percent (25-100) of its size."""
        try:
            if scale_percent >= 100:
                with open(self.path, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode('utf-8')
            with Image.open(self.path) as im:
                width, height = im.size
                size = (max(1, round(width * scale_percent / 100)), max(1, round(height * scale_percent / 100)))
                im.draft("RGB", size)  # fast downscaled JPEG decoding
                im = im.convert("RGB").resize(size, Image.LANCZOS)
                buf = io.BytesIO()
                im.save(buf, format="JPEG", quality=90)
                return base64.b64encode(buf.getvalue()).decode('utf-8')
        except (IOError, OSError) as e:
            print(f"Error reading image file {self.path}: {e}")
            return None

    def get_exif_datetime(self) -> Optional[datetime]:
        """The capture date from EXIF (DateTimeOriginal, then DateTimeDigitized, then DateTime), or None."""
        try:
            exif_data = piexif.load(self.path)
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
            exif_data = piexif.load(self.path)
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

            # Keep existing EXIF (capture date, camera, ...) instead of replacing it
            try:
                exif_dict = piexif.load(self.path)
            except Exception:
                exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

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
