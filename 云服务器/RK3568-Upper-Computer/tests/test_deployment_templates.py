import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_renderer():
    path = ROOT / "scripts" / "render-deployment.py"
    spec = importlib.util.spec_from_file_location("render_deployment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DeploymentTemplateTests(unittest.TestCase):
    def test_all_templates_render_without_unresolved_placeholders(self):
        renderer = load_renderer()
        placeholders = set()
        for source_dir, names in renderer.TEMPLATE_GROUPS.values():
            for name in names:
                placeholders.update(
                    renderer.PLACEHOLDER.findall(
                        (source_dir / name).read_text(encoding="utf-8")
                    )
                )
        values = {
            placeholder: f"value-{index}"
            for index, placeholder in enumerate(sorted(placeholders))
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            for role in renderer.TEMPLATE_GROUPS:
                written = renderer.render_group(role, output, values)
                self.assertTrue(written)
                for path in written:
                    self.assertFalse(
                        renderer.PLACEHOLDER.search(
                            path.read_text(encoding="utf-8")
                        )
                    )

    def test_deployment_requires_canary_and_explicit_activation(self):
        script = (
            ROOT / "scripts" / "deploy-cloud.ps1"
        ).read_text(encoding="utf-8")
        self.assertIn('"Canary", "Upload", "Activate"', script)
        self.assertIn("-not $ConfirmProduction", script)
        self.assertIn("/var/backups/rk3568-manager", script)
        self.assertIn("continuity_canary", (
            ROOT / "deploy" / "cloud" / "mediamtx.yml.template"
        ).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
