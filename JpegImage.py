import base64
import os

import piexif


class JpegImage:
    def __init__(self, path):
        self.path = path

    def get_as_base64(self):
            try:
                with open(self.path, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode('utf-8')
            except IOError as e:
                print(f"Error reading image file {self.path}: {e}")
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

    def write_exif_tags(self, tags, dry_run=False):
        """
        Writes the given tags to the image's EXIF data, preserving the original
        file modification time.
        """
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