import pytest
from unittest.mock import MagicMock, patch
from modules import safe
import zipfile

class TestCheckZipFilenames:
    @pytest.mark.parametrize("names", [
        ['archive/data.pkl'],
        ['archive/version'],
        ['archive/byteorder'],
        ['archive/data/0'],
        ['archive/data/123'],
        ['archive/.data/serialization_id'],
        # Multiple valid files
        ['archive/data.pkl', 'archive/version', 'archive/data/0']
    ])
    def test_valid_filenames(self, names):
        """Test that valid filenames do not raise an exception."""
        safe.check_zip_filenames("test_archive.zip", names)

    @pytest.mark.parametrize("names", [
        ['archive/script.py'],          # Invalid extension
        ['/absolute/path'],             # Absolute path
        ['../traversal'],               # Path traversal
        ['archive/data.pkl', 'malicious.exe'], # Mixed valid and invalid
        ['archive/sub/data.pkl'],       # Nested directory structure not allowed by regex
        ['archive/data/not_a_number'],  # Invalid data file format
        ['archive/.data/other_id'],     # Invalid dot file
        ['archive/'],                   # Directory entry
        ['data.pkl']                    # Missing root directory
    ])
    def test_invalid_filenames(self, names):
        """Test that invalid filenames raise an exception."""
        with pytest.raises(Exception, match="bad file inside"):
            safe.check_zip_filenames("test_archive.zip", names)

class TestCheckPt:
    @patch('zipfile.ZipFile')
    def test_check_pt_valid_zip(self, mock_zip_cls):
        """Test check_pt with a valid zip file."""
        # Setup mock zip
        mock_zip = MagicMock()
        mock_zip_cls.return_value.__enter__.return_value = mock_zip

        # Valid files
        mock_zip.namelist.return_value = ['archive/data.pkl', 'archive/version']

        # Setup mock for data.pkl open
        mock_file = MagicMock()
        mock_zip.open.return_value.__enter__.return_value = mock_file

        # Mock RestrictedUnpickler to avoid actual unpickling logic
        with patch('modules.safe.RestrictedUnpickler') as mock_unpickler_cls:
            safe.check_pt("test_model.pt", None)

            # Verify interactions
            mock_zip.namelist.assert_called()
            mock_zip.open.assert_called_with('archive/data.pkl')
            mock_unpickler_cls.assert_called()
            mock_unpickler_cls.return_value.load.assert_called()

    @patch('zipfile.ZipFile')
    def test_check_pt_bad_zip_content(self, mock_zip_cls):
        """Test check_pt with a zip containing malicious files."""
        mock_zip = MagicMock()
        mock_zip_cls.return_value.__enter__.return_value = mock_zip
        mock_zip.namelist.return_value = ['archive/malicious.py']

        with pytest.raises(Exception, match="bad file inside"):
            safe.check_pt("bad_model.pt", None)

    @patch('zipfile.ZipFile')
    def test_check_pt_no_data_pkl(self, mock_zip_cls):
        """Test check_pt with a zip missing data.pkl."""
        mock_zip = MagicMock()
        mock_zip_cls.return_value.__enter__.return_value = mock_zip
        mock_zip.namelist.return_value = ['archive/version'] # No data.pkl

        with pytest.raises(Exception, match="data.pkl not found"):
            safe.check_pt("no_pkl.pt", None)

    @patch('zipfile.ZipFile')
    def test_check_pt_multiple_data_pkl(self, mock_zip_cls):
        """Test check_pt with a zip containing multiple data.pkl files."""
        mock_zip = MagicMock()
        mock_zip_cls.return_value.__enter__.return_value = mock_zip
        # Two data.pkl in different folders
        mock_zip.namelist.return_value = ['folder1/data.pkl', 'folder2/data.pkl']

        with pytest.raises(Exception, match="Multiple data.pkl found"):
            safe.check_pt("multi_pkl.pt", None)

    def test_check_pt_not_zip_fallback(self):
        """Test fallback to legacy pickle format when file is not a zip."""
        # Test the fallback to pickle loading when ZipFile raises BadZipfile
        with patch('zipfile.ZipFile', side_effect=zipfile.BadZipfile),              patch('builtins.open', MagicMock()) as mock_open,              patch('modules.safe.RestrictedUnpickler') as mock_unpickler_cls:

            safe.check_pt("legacy_model.ckpt", None)

            mock_open.assert_called_with("legacy_model.ckpt", "rb")
            # Should load 5 times (legacy format convention)
            assert mock_unpickler_cls.return_value.load.call_count == 5
