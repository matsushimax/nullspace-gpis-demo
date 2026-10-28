"""Check public documentation links and bundled media without network access."""

from pathlib import Path
import re
import unittest

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


class DocumentationTests(unittest.TestCase):
    def test_local_links(self):
        documents = ("README.md", "DATABASE.md", "COLLISION_DATABASE.md", "INTERACTIVE.md", "CONTRIBUTING.md")
        for name in documents:
            path = ROOT / name
            for target in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
                if target.startswith(("https://", "http://", "#")):
                    continue
                with self.subTest(document=name, target=target):
                    self.assertTrue((path.parent / target.split("#", 1)[0]).is_file())

    def test_bundled_media(self):
        for name in ("self_motion.gif", "self_motion.png", "tps_projection.png", "obstacle_boundary.png"):
            path = ROOT / "docs" / "images" / name
            with self.subTest(image=name), Image.open(path) as image:
                self.assertGreaterEqual(image.width, 800)
                self.assertGreaterEqual(image.height, 400)
                self.assertFalse(image.getexif(), "Do not publish identifying EXIF metadata")
                if path.suffix == ".gif":
                    self.assertGreater(image.n_frames, 1)
            with Image.open(path) as image:
                image.verify()


if __name__ == "__main__":
    unittest.main()
