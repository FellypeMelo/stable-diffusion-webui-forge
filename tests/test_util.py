import sys
import unittest
from unittest.mock import MagicMock, patch

# Mock modules to avoid dependencies and side effects
sys.modules["gradio"] = MagicMock()

# Mock modules.shared
shared_mock = MagicMock()
sys.modules["modules.shared"] = shared_mock

# Mock modules.paths_internal
paths_internal_mock = MagicMock()
paths_internal_mock.cwd = "/app"
paths_internal_mock.script_path = "/app"
sys.modules["modules.paths_internal"] = paths_internal_mock

# Now import the module under test
# We need to make sure 'modules' package exists or is mockable if not in path
# Assuming /app is in pythonpath
from modules.util import truncate_path

class TestTruncatePath(unittest.TestCase):
    def test_relative_path(self):
        """Should return relative path when target is inside base."""
        base = "/a/b"
        target = "/a/b/c"
        # Mock abspath to ensure consistent behavior
        with patch('os.path.abspath', side_effect=lambda p: p):
            with patch('os.path.commonpath', return_value=base):
                with patch('os.path.relpath', return_value="c"):
                    result = truncate_path(target, base)
                    self.assertEqual(result, "c")

    def test_identity_path(self):
        """Should return '.' or relative path when target is the base."""
        base = "/a/b"
        target = "/a/b"
        with patch('os.path.abspath', side_effect=lambda p: p):
            with patch('os.path.commonpath', return_value=base):
                with patch('os.path.relpath', return_value="."):
                    result = truncate_path(target, base)
                    self.assertEqual(result, ".")

    def test_outside_path_parent(self):
        """Should return absolute path when target is outside base (parent)."""
        base = "/a/b"
        target = "/a"
        with patch('os.path.abspath', side_effect=lambda p: p):
            # commonpath of /a/b and /a is /a
            with patch('os.path.commonpath', return_value="/a"):
                result = truncate_path(target, base)
                self.assertEqual(result, target)

    def test_outside_path_sibling(self):
        """Should return absolute path when target is outside base (sibling)."""
        base = "/a/b"
        target = "/a/c"
        with patch('os.path.abspath', side_effect=lambda p: p):
            # commonpath of /a/b and /a/c is /a
            with patch('os.path.commonpath', return_value="/a"):
                result = truncate_path(target, base)
                self.assertEqual(result, target)

    def test_different_drives(self):
        """Should return absolute path when commonpath raises ValueError (e.g. different drives)."""
        base = "C:/a/b"
        target = "D:/x/y"
        with patch('os.path.abspath', side_effect=lambda p: p):
            with patch('os.path.commonpath', side_effect=ValueError):
                result = truncate_path(target, base)
                self.assertEqual(result, target)

if __name__ == '__main__':
    unittest.main()
