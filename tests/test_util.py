import os
import sys
import re
from unittest.mock import MagicMock

# Set environment variable to ignore command line argument errors
os.environ['IGNORE_CMD_ARGS_ERRORS'] = '1'

# Mock dependencies that are not installed or heavy
sys.modules["gradio"] = MagicMock()
sys.modules["modules.shared"] = MagicMock()
sys.modules["modules.paths_internal"] = MagicMock()
sys.modules["modules.paths_internal"].script_path = "."
sys.modules["modules.paths_internal"].cwd = "."
sys.modules["modules.cmd_args"] = MagicMock()

# Ensure the project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the module under test
try:
    from modules import util
except ImportError:
    pass

import pytest

class TestNaturalSortKey:
    def test_basic_sorting(self):
        """Test basic sorting of filenames with numbers."""
        files = ["file1.txt", "file10.txt", "file2.txt"]
        sorted_files = sorted(files, key=util.natural_sort_key)
        assert sorted_files == ["file1.txt", "file2.txt", "file10.txt"]

    def test_mixed_case(self):
        """Test that sorting is case-insensitive."""
        files = ["File1.txt", "file10.txt", "FILE2.txt"]
        sorted_files = sorted(files, key=util.natural_sort_key)
        assert sorted_files == ["File1.txt", "FILE2.txt", "file10.txt"]

    def test_version_strings(self):
        """Test sorting of version strings."""
        versions = ["v1.0.0", "v1.10.0", "v1.2.0"]
        sorted_versions = sorted(versions, key=util.natural_sort_key)
        assert sorted_versions == ["v1.0.0", "v1.2.0", "v1.10.0"]

    def test_empty_strings(self):
        """Test sorting with empty strings."""
        files = ["file1.txt", "", "file2.txt"]
        sorted_files = sorted(files, key=util.natural_sort_key)
        assert sorted_files == ["", "file1.txt", "file2.txt"]

    def test_pure_numbers(self):
        """Test sorting of strings that are just numbers."""
        nums = ["10", "1", "2"]
        sorted_nums = sorted(nums, key=util.natural_sort_key)
        assert sorted_nums == ["1", "2", "10"]

    def test_special_characters(self):
        """Test sorting with special characters."""
        files = ["file_1.txt", "file-2.txt", "file_10.txt"]
        sorted_files = sorted(files, key=util.natural_sort_key)
        assert sorted_files == ["file-2.txt", "file_1.txt", "file_10.txt"]

    def test_unicode(self):
        """Test sorting with unicode characters."""
        files = ["café1.txt", "cafe2.txt", "café10.txt"]
        sorted_files = sorted(files, key=util.natural_sort_key)
        assert sorted_files == ["cafe2.txt", "café1.txt", "café10.txt"]

    def test_custom_regex(self):
        """Test using a custom regex for splitting."""
        regex = re.compile('(-)')
        items = ["beta-alpha", "alpha-beta"]
        sorted_items = sorted(items, key=lambda x: util.natural_sort_key(x, regex=regex))
        assert sorted_items == ["alpha-beta", "beta-alpha"]

    def test_structure(self):
        """Verify the internal structure of the key."""
        key = util.natural_sort_key("file10.txt")
        # Should be ['file', 10, '.txt']
        assert key == ['file', 10, '.txt']
