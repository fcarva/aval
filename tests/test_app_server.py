import importlib.util
import unittest


# app_server imports uvicorn at module load; the core CI job runs without the web
# stack, so skip the whole module there instead of failing to import (mirrors the
# FastAPI guard in test_api.py).
UVICORN_AVAILABLE = importlib.util.find_spec("uvicorn") is not None

if UVICORN_AVAILABLE:
    import app_server


@unittest.skipUnless(UVICORN_AVAILABLE, "uvicorn web dependency is not installed.")
class ResolveHostTest(unittest.TestCase):
    def test_default_when_nothing_set(self) -> None:
        self.assertEqual(app_server.resolve_host(None, env={}), "127.0.0.1")

    def test_env_overrides_default(self) -> None:
        self.assertEqual(
            app_server.resolve_host(None, env={"AVAL_HOST": "0.0.0.0"}),
            "0.0.0.0",
        )

    def test_cli_wins_over_env(self) -> None:
        self.assertEqual(
            app_server.resolve_host("10.0.0.5", env={"AVAL_HOST": "0.0.0.0"}),
            "10.0.0.5",
        )

    def test_blank_env_falls_back_to_default(self) -> None:
        self.assertEqual(app_server.resolve_host(None, env={"AVAL_HOST": ""}), "127.0.0.1")


@unittest.skipUnless(UVICORN_AVAILABLE, "uvicorn web dependency is not installed.")
class ResolvePortTest(unittest.TestCase):
    def test_default_when_nothing_set(self) -> None:
        self.assertEqual(app_server.resolve_port(None, env={}), 8000)

    def test_env_overrides_default(self) -> None:
        self.assertEqual(app_server.resolve_port(None, env={"AVAL_PORT": "9100"}), 9100)

    def test_cli_wins_over_env(self) -> None:
        self.assertEqual(app_server.resolve_port(4000, env={"AVAL_PORT": "9100"}), 4000)

    def test_non_integer_env_raises(self) -> None:
        with self.assertRaises(ValueError):
            app_server.resolve_port(None, env={"AVAL_PORT": "not-a-port"})

    def test_out_of_range_port_raises(self) -> None:
        with self.assertRaises(ValueError):
            app_server.resolve_port(70000, env={})


@unittest.skipUnless(UVICORN_AVAILABLE, "uvicorn web dependency is not installed.")
class ParseArgsTest(unittest.TestCase):
    def test_defaults_are_none_and_reload_false(self) -> None:
        args = app_server.parse_args([])
        self.assertIsNone(args.host)
        self.assertIsNone(args.port)
        self.assertFalse(args.reload)

    def test_flags_are_parsed(self) -> None:
        args = app_server.parse_args(["--host", "0.0.0.0", "--port", "9000", "--reload"])
        self.assertEqual(args.host, "0.0.0.0")
        self.assertEqual(args.port, 9000)
        self.assertTrue(args.reload)


@unittest.skipUnless(UVICORN_AVAILABLE, "uvicorn web dependency is not installed.")
class MainTest(unittest.TestCase):
    def test_main_dispatches_resolved_config_to_runner(self) -> None:
        calls: list[tuple] = []

        def fake_runner(*args, **kwargs):
            calls.append((args, kwargs))

        app_server.main(["--host", "1.2.3.4", "--port", "5555"], runner=fake_runner)

        self.assertEqual(len(calls), 1)
        args, kwargs = calls[0]
        self.assertEqual(args, (app_server.APP_PATH,))
        self.assertEqual(kwargs["host"], "1.2.3.4")
        self.assertEqual(kwargs["port"], 5555)
        self.assertFalse(kwargs["reload"])

    def test_main_does_not_start_real_server(self) -> None:
        # Sanity guard: the injected runner replaces uvicorn.run entirely, so
        # invoking main() must not touch the real server.
        started = []
        app_server.main([], runner=lambda *a, **k: started.append(True))
        self.assertEqual(started, [True])


if __name__ == "__main__":
    unittest.main()
