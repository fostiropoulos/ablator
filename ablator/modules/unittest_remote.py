import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch
import sys
sys.path.append('/app/')

from ablator.modules.storage.remote import RemoteConfig

class TestRemoteConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = Path(tempfile.mkdtemp())

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir)

    def setUp(self):
        self.config = RemoteConfig(
            remote_path="/tmp",
            username="testuser",
            hostname="example.com",
            port=22,
            exclude_glob="*.txt",
            exclude_chkpts=True,
        )
        self.srcdir = self.tmpdir / "src"
        self.srcdir.mkdir()
        self.dstdir = self.tmpdir / "dst"

    def tearDown(self):
        if self.dstdir.exists():
            shutil.rmtree(self.dstdir)
        if self.srcdir.exists():
            shutil.rmtree(self.srcdir)

    def test_rsync_up(self):
        with (self.srcdir / "test.txt").open("w") as f:
            f.write("hello")
        with patch("ablator.modules.storage.remote.run_cmd_wait") as mock_run_cmd_wait:
            self.config.rsync_up(self.srcdir, "test")
            mock_run_cmd_wait.assert_called_once_with(
                'rsync -art --rsync-path="mkdir -p /tmp/test && rsync" -e "ssh -o \\"StrictHostKeyChecking=no\\" -p 22" --exclude="*.txt" --exclude="*.pt" {} testuser@example.com:/tmp/test'.format(self.srcdir),
                None,
            )

    def test_rsync_down(self):
        (self.dstdir / "test.txt").mkdir(parents=True)
        with patch("ablator.modules.storage.remote.run_cmd_wait") as mock_run_cmd_wait:
            self.config.rsync_down(self.dstdir / "test", "")
            mock_run_cmd_wait.assert_called_once_with(
                'rsync -art --rsync-path="mkdir -p  && rsync" -e "ssh -o \\"StrictHostKeyChecking=no\\" -p 22" testuser@example.com:/tmp/test/ {}'.format(
                    self.dstdir
                ),
                None,
            )


if __name__ == "__main__":
    unittest.main()
