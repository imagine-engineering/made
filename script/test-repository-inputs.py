from contextlib import redirect_stdout
import io
from pathlib import Path
import runpy
import subprocess
import tempfile
import unittest


checker = runpy.run_path(str(Path(__file__).with_name("check-repository-inputs")))


class RepositoryInputTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.git("init", "--quiet")

    def git(self, *arguments):
        return subprocess.run(
            ["git", *arguments], cwd=self.root, check=True, capture_output=True, text=True
        )

    def write(self, name, contents, tracked=True):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
        if tracked:
            self.git("add", "--", name)
        return path

    def check(self):
        with redirect_stdout(io.StringIO()):
            checker["check"](self.root)

    def test_relative_fixture_is_tracked(self):
        self.write("src/lib.rs", 'const DATA: &str = include_str!("../fixtures/data.json");')
        self.write("fixtures/data.json", "{}")
        self.check()

    def test_manifest_relative_fixture_is_tracked(self):
        self.write("crates/example/Cargo.toml", '[package]\nname = "example"\n')
        self.write(
            "crates/example/src/example.rs",
            'const DATA: &[u8] = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/data.bin"));',
        )
        self.write("crates/example/fixtures/data.bin", "fixture")
        self.check()

    def test_missing_or_untracked_fixture_outside_migrated_owners_is_rejected(self):
        self.write("src/lib.rs", 'const DATA: &str = include_str!("../fixtures/data.json");')
        with self.assertRaisesRegex(RuntimeError, "missing or untracked compile input"):
            self.check()
        self.write("fixtures/data.json", "{}", tracked=False)
        with self.assertRaisesRegex(RuntimeError, "missing or untracked compile input"):
            self.check()

    def test_private_agent_input_is_rejected(self):
        self.write("src/lib.rs", 'const DATA: &str = include_str!("../.agents/input.txt");')
        self.write(".agents/input.txt", "fixture", tracked=False)
        with self.assertRaisesRegex(RuntimeError, "local agent guidance is a compile input"):
            self.check()

    def test_repository_escape_is_rejected(self):
        self.write("src/lib.rs", 'const DATA: &str = include_str!("../../outside.txt");')
        with self.assertRaisesRegex(RuntimeError, "compile input escapes the repository"):
            self.check()

    def test_eval_excerpt_is_data_rather_than_a_compiled_module(self):
        fixture = "tools/evals/fixtures/use_wasi_sdk_in_compile_parser_to_wasm/before.rs"
        self.write("crates/agent/src/agent.rs", f'const INPUT: &str = include_str!("{fixture}");')
        self.write(f"crates/agent/src/{fixture}", '#![doc = include_str!("../README.md")]')
        self.check()


if __name__ == "__main__":
    unittest.main()
