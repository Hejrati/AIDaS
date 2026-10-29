"""Exercise the actual inline release runner without launching the full suite."""

import contextlib
import gc
import io
import textwrap
import threading
import unittest
from unittest import mock

from tests.support.paths import PROJECT_ROOT


class WorkflowTestRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        workflow = (PROJECT_ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
        step = workflow.split("      - name: Run tests\n", 1)[1].split("      - name:", 1)[0]
        cls.script = textwrap.dedent(step.split("          @'\n", 1)[1].split("          '@ | python -", 1)[0])

    def setUp(self):
        collection_enabled = gc.isenabled()
        self.addCleanup(gc.enable if collection_enabled else gc.disable)

    def _execute(self, suite):
        output = io.StringIO()
        with mock.patch.object(unittest.defaultTestLoader, "discover", return_value=suite) as discover:
            with contextlib.redirect_stderr(output), self.assertRaises(SystemExit) as exited:
                exec(compile(self.script, "release.yml:Run tests", "exec"), {})
        discover.assert_called_once_with("tests", top_level_dir=".")
        return exited.exception.code, output.getvalue()

    def test_old_cycles_and_worker_allocations_are_collected_on_the_main_thread(self):
        finalized = []
        main_thread = threading.get_ident()

        class Cycle:
            def __init__(self):
                self.cycle = self

            def __del__(self):
                finalized.append(threading.get_ident())

        def allocate():
            for _ in range(4000):
                Cycle()

        def first_case():
            self.assertFalse(gc.isenabled())
            Cycle()
            worker = threading.Thread(target=allocate)
            worker.start()
            worker.join(timeout=3)
            self.assertFalse(worker.is_alive())
            self.assertEqual(finalized, [])

        def second_case():
            self.assertEqual(len(finalized), 4001)
            self.assertEqual(set(finalized), {main_thread})
            Cycle()  # The runner's final collection must also collect the last case.

        gc.enable()
        code, output = self._execute(unittest.TestSuite([
            unittest.FunctionTestCase(first_case), unittest.FunctionTestCase(second_case),
        ]))
        self.assertEqual(code, 0, output)
        self.assertEqual(len(finalized), 4002)
        self.assertEqual(set(finalized), {main_thread})
        self.assertTrue(gc.isenabled())
        self.assertIn("Ran 2 tests", output)

    def test_failure_and_error_still_fail_the_run_without_skipping_later_cases(self):
        executed = []

        def failure():
            executed.append("failure")
            raise AssertionError("a real assertion failure")

        def error():
            executed.append("error")
            raise RuntimeError("a real test error")

        def success():
            executed.append("success")

        gc.enable()
        code, output = self._execute(unittest.TestSuite([
            unittest.FunctionTestCase(case) for case in (failure, error, success)
        ]))
        self.assertEqual(code, 1)
        self.assertEqual(executed, ["failure", "error", "success"])
        self.assertIn("FAILED (failures=1, errors=1)", output)
        self.assertTrue(gc.isenabled())

    def test_discovery_error_restores_collection_and_propagates(self):
        for enabled in (True, False):
            with self.subTest(initial_collection=enabled):
                (gc.enable if enabled else gc.disable)()
                with mock.patch.object(unittest.defaultTestLoader, "discover", side_effect=RuntimeError("discovery failed")):
                    with self.assertRaisesRegex(RuntimeError, "discovery failed"):
                        exec(compile(self.script, "release.yml:Run tests", "exec"), {})
                self.assertEqual(gc.isenabled(), enabled)

    def test_disabled_collection_setting_is_preserved_after_success(self):
        gc.disable()
        code, output = self._execute(unittest.TestSuite([unittest.FunctionTestCase(lambda: None)]))
        self.assertEqual(code, 0, output)
        self.assertFalse(gc.isenabled())


if __name__ == "__main__":
    unittest.main()
