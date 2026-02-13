import sys
from unittest.mock import MagicMock

# Mock all the required modules before importing modules.images
mock_shared = MagicMock()
mock_shared.cmd_opts.unix_filenames_sanitization = False
mock_shared.cmd_opts.filenames_max_length = 128
sys.modules["modules.shared"] = mock_shared

sys.modules["pytz"] = MagicMock()
sys.modules["numpy"] = MagicMock()
sys.modules["piexif"] = MagicMock()
sys.modules["piexif.helper"] = MagicMock()
sys.modules["PIL"] = MagicMock()
sys.modules["PIL.Image"] = MagicMock()
sys.modules["PIL.ImageFont"] = MagicMock()
sys.modules["PIL.ImageDraw"] = MagicMock()
sys.modules["PIL.ImageColor"] = MagicMock()
sys.modules["PIL.PngImagePlugin"] = MagicMock()
sys.modules["PIL.ImageOps"] = MagicMock()
sys.modules["pillow_avif"] = MagicMock()

sys.modules["modules.sd_samplers"] = MagicMock()
sys.modules["modules.script_callbacks"] = MagicMock()
sys.modules["modules.errors"] = MagicMock()
sys.modules["modules.stealth_infotext"] = MagicMock()
sys.modules["modules.paths_internal"] = MagicMock()

# Now import the function to test
from modules.images import sanitize_filename_part
import modules.images

def test_sanitize_filename_part_none():
    assert sanitize_filename_part(None) is None

def test_sanitize_filename_part_replace_spaces():
    assert sanitize_filename_part("hello world") == "hello_world"
    assert sanitize_filename_part("hello world", replace_spaces=False) == "hello world"

def test_sanitize_filename_part_invalid_chars():
    # default invalid_filename_chars = '#<>:"/\\|?*\n\r\t'
    original_chars = modules.images.invalid_filename_chars
    modules.images.invalid_filename_chars = '#<>:"/\\|?*\n\r\t'
    try:
        assert sanitize_filename_part("a#b<c>d:e\"f/g\\h|i?j*k\nl\rm\tn") == "a_b_c_d_e_f_g_h_i_j_k_l_m_n"
    finally:
        modules.images.invalid_filename_chars = original_chars

def test_sanitize_filename_part_lstrip():
    # invalid_filename_prefix = ' '
    # If replace_spaces=True (default), "  hello" becomes "__hello"
    # and lstrip(' ') does nothing to underscores.
    assert sanitize_filename_part("  hello") == "__hello"
    assert sanitize_filename_part("  hello", replace_spaces=False) == "hello"

def test_sanitize_filename_part_rstrip():
    # invalid_filename_postfix = ' .'
    # If replace_spaces=True (default), "hello. " becomes "hello._"
    # and rstrip(' .') does nothing to underscores.
    assert sanitize_filename_part("hello. ") == "hello._"
    assert sanitize_filename_part("hello. ", replace_spaces=False) == "hello"

    assert sanitize_filename_part("hello..") == "hello"
    assert sanitize_filename_part("hello ") == "hello_"
    # "hello . ." -> "hello_._." -> rstrip(' .') -> "hello_._"
    assert sanitize_filename_part("hello . .") == "hello_._"

def test_sanitize_filename_part_max_length():
    old_max = modules.images.max_filename_part_length
    modules.images.max_filename_part_length = 5
    try:
        assert sanitize_filename_part("verylongfilename") == "veryl"
    finally:
        modules.images.max_filename_part_length = old_max

def test_sanitize_filename_part_unix_sanitization():
    old_chars = modules.images.invalid_filename_chars
    modules.images.invalid_filename_chars = '/'
    try:
        assert sanitize_filename_part("a#b/c") == "a#b_c"
    finally:
        modules.images.invalid_filename_chars = old_chars

def test_sanitize_filename_part_empty_and_restricted():
    assert sanitize_filename_part("") == ""
    # Space becomes _ then lstrip ' ' doesn't remove it.
    assert sanitize_filename_part(" ") == "_"
    # replace_spaces=False, " " lstrip ' ' becomes ""
    assert sanitize_filename_part(" ", replace_spaces=False) == ""
    # " . " -> "_._"
    assert sanitize_filename_part(" . ") == "_._"
    # " . " replace_spaces=False -> " . " -> lstrip ' ' -> ". " -> rstrip ' .' -> ""
    assert sanitize_filename_part(" . ", replace_spaces=False) == ""
