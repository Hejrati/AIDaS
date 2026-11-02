from pathlib import Path
import os
import subprocess
import tempfile
import threading
import unittest
from unittest import mock

from aidas.utils import filesystem
from aidas.utils.filesystem import (
    DirectoryScanLimitExceeded,
    find_candidate_directories,
    find_sdb_directories,
    walk_accessible_directories,
    walk_unique_directories,
)


class FilesystemScanLimitTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def _scan(self, kind, *, cancel_event=None):
        if kind == "sdb":
            matches, errors = find_sdb_directories(self.root, cancel_event=cancel_event)
            return list(matches), errors
        if kind == "candidate":
            matches, _count, errors = find_candidate_directories(
                self.root, (("light.img",),), cancel_event=cancel_event,
            )
            return matches, errors
        return walk_accessible_directories(self.root, cancel_event=cancel_event)

    def test_all_discovery_helpers_accept_exactly_500_subfolders_and_exclude_root(self):
        for number in range(500):
            (self.root / str(number)).mkdir()
        (self.root / "scan.sdb").touch()
        (self.root / "Light.img").touch()

        self.assertEqual(self._scan("sdb"), ([self.root], []))
        self.assertEqual(self._scan("candidate"), ([self.root], []))
        folders, errors = self._scan("accessible")
        self.assertEqual(len(folders), 501)
        self.assertEqual(folders[0], self.root)
        self.assertEqual(errors, [])

    def test_overflow_rejects_all_discovery_helpers_instead_of_returning_partial_matches(self):
        (self.root / "scan.sdb").touch()
        (self.root / "Light.img").touch()
        for number in range(501):
            (self.root / str(number)).mkdir()

        for kind in ("sdb", "candidate", "accessible"):
            with self.subTest(kind=kind), self.assertRaises(DirectoryScanLimitExceeded) as raised:
                self._scan(kind)
            self.assertEqual(raised.exception.limit, 500)
            self.assertEqual(str(raised.exception), (
                "The selected folder contains too many subfolders and exceeds the limit of 500. "
                "Please select another folder."
            ))

    def test_nested_total_and_deep_legitimate_results_have_no_depth_cap(self):
        deep = self.root.joinpath(*("d" for _ in range(20)))
        deep.mkdir(parents=True)
        (deep / "sample.sdb").touch()
        (deep / "Light.img").touch()
        self.assertEqual(self._scan("sdb"), ([deep], []))
        self.assertEqual(self._scan("candidate"), ([deep], []))
        self.assertEqual(len(self._scan("accessible")[0]), 21)
        with mock.patch.object(filesystem, "MAX_SCAN_SUBFOLDERS", 19):
            for kind in ("sdb", "candidate", "accessible"):
                with self.subTest(kind=kind), self.assertRaises(DirectoryScanLimitExceeded):
                    self._scan(kind)

    def test_oversized_walk_stops_on_first_folder_over_budget(self):
        for kind in ("sdb", "candidate", "accessible"):
            with self.subTest(kind=kind):
                visits = []

                def walk(_root, *, topdown, onerror):
                    yield str(self.root), [], ["scan.sdb", "Light.img"]
                    for number in range(10_000):
                        visits.append(number)
                        if number > 500:
                            self.fail("The scan continued after its budget was exceeded")
                        yield str(self.root / f"folder_{number}"), [], ["scan.sdb", "Light.img"]

                with mock.patch.object(filesystem.os, "walk", side_effect=walk):
                    with self.assertRaises(DirectoryScanLimitExceeded):
                        self._scan(kind)
                self.assertEqual(len(visits), 501)

    def test_pre_cancelled_scans_do_not_touch_the_tree(self):
        cancelled = threading.Event()
        cancelled.set()
        with mock.patch.object(filesystem.os, "walk") as walk:
            for kind in ("sdb", "candidate", "accessible"):
                with self.subTest(kind=kind):
                    self.assertEqual(self._scan(kind, cancel_event=cancelled), ([], []))
        walk.assert_not_called()

    def test_cancelling_after_a_match_discards_partial_results(self):
        for kind in ("sdb", "candidate", "accessible"):
            with self.subTest(kind=kind):
                cancelled = threading.Event()

                def walk(_root, *, topdown, onerror):
                    yield str(self.root), ["next"], ["scan.sdb", "Light.img"]
                    cancelled.set()
                    yield str(self.root / "next"), [], ["scan.sdb", "Light.img"]

                with mock.patch.object(filesystem.os, "walk", side_effect=walk):
                    self.assertEqual(self._scan(kind, cancel_event=cancelled), ([], []))

    def test_cancellation_interrupts_identity_checks_in_a_wide_directory(self):
        original_stat = os.stat
        for kind in ("sdb", "candidate", "accessible"):
            with self.subTest(kind=kind):
                cancelled = threading.Event()
                checked = []

                def stat(path, *args, **kwargs):
                    if Path(path).name.startswith("child_"):
                        checked.append(path)
                        if len(checked) == 3:
                            cancelled.set()
                    return original_stat(path, *args, **kwargs)

                def walk(_root, *, topdown, onerror):
                    yield str(self.root), [f"child_{number}" for number in range(10_000)], ["scan.sdb", "Light.img"]

                with mock.patch.object(filesystem.os, "walk", side_effect=walk), \
                        mock.patch.object(filesystem.os, "stat", side_effect=stat):
                    self.assertEqual(self._scan(kind, cancel_event=cancelled), ([], []))
                self.assertEqual(len(checked), 3)

    def test_parent_junction_cycle_is_pruned_before_recursion_for_every_helper(self):
        original_realpath = os.path.realpath

        def realpath(path, *args, **kwargs):
            return str(self.root) if Path(path).name == "back" else original_realpath(path, *args, **kwargs)

        def walk(root, *, topdown, onerror):
            pending = [os.fspath(root)]
            visits = 0
            while pending:
                folder = pending.pop()
                visits += 1
                self.assertLessEqual(visits, 2, "The parent junction was followed repeatedly")
                branches = ["back"]
                yield folder, branches, ["scan.sdb", "Light.img"]
                pending.extend(os.path.join(folder, name) for name in branches)

        with mock.patch.object(filesystem.os.path, "realpath", side_effect=realpath), \
                mock.patch.object(filesystem.os, "walk", side_effect=walk):
            for kind in ("sdb", "candidate", "accessible"):
                with self.subTest(kind=kind):
                    self.assertEqual(self._scan(kind), ([self.root], []))

    def test_sorting_and_access_errors_are_preserved(self):
        def walk(root, *, topdown, onerror):
            onerror(PermissionError(13, "denied", str(root / "denied")))
            branches = ["Z", "a"]
            yield str(root), branches, ["b.sdb", "A.SDB", "Light.img"]
            for branch in branches:
                yield str(root / branch), [], ["scan.sdb", "Light.img"]

        with mock.patch.object(filesystem.os, "walk", side_effect=walk):
            for kind in ("sdb", "candidate", "accessible"):
                with self.subTest(kind=kind):
                    folders, errors = self._scan(kind)
                    self.assertEqual(folders, [self.root, self.root / "a", self.root / "Z"])
                    self.assertEqual(errors[0][0], self.root / "denied")
            matches, _errors = find_sdb_directories(self.root)
            self.assertEqual(matches[self.root], [self.root / "A.SDB", self.root / "b.sdb"])

    def test_real_link_cycle_cannot_repeat_a_batch_folder(self):
        study = self.root / "study"
        study.mkdir()
        (study / "scan.sdb").touch()
        (study / "Light.img").touch()
        link = study / "back"
        junction = False
        try:
            link.symlink_to(self.root, target_is_directory=True)
        except (OSError, NotImplementedError) as error:
            if os.name != "nt":
                self.skipTest(f"Directory links are unavailable: {error}")
            created = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(link), str(self.root)],
                capture_output=True, text=True, check=False, creationflags=subprocess.CREATE_NO_WINDOW,
            )
            if created.returncode:
                self.skipTest(f"Directory links are unavailable: {created.stderr.strip()}")
            junction = True
        try:
            self.assertEqual(self._scan("sdb"), ([study], []))
            self.assertEqual(self._scan("candidate"), ([study], []))
            self.assertEqual(self._scan("accessible"), ([self.root, study], []))
        finally:
            link.rmdir() if junction else link.unlink()

    def test_direct_unique_walk_remains_unbounded(self):
        def walk(_root, *, topdown, onerror):
            yield str(self.root), [], []
            for number in range(501):
                yield str(self.root / str(number)), [], []

        with mock.patch.object(filesystem.os, "walk", side_effect=walk):
            self.assertEqual(len(list(walk_unique_directories(self.root))), 502)


if __name__ == "__main__":
    unittest.main()
