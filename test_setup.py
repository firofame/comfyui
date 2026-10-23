import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import main


class PreparationTests(unittest.TestCase):
    def setUp(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(directory)
        self.source_root = self.root / "training"
        self.source_root.mkdir()
        self.source = self.source_root / "final.safetensors"
        self.source.write_bytes(b"trained weight")
        self.models_root = self.root / "models"
        self.destination = self.models_root / "loras" / "identity.safetensors"
        self.enterContext(patch.object(main, "SOURCE_DIR", str(self.source_root)))
        self.enterContext(patch.object(main, "MODELS_DIR", str(self.models_root)))
        self.commit = self.enterContext(patch.object(main.models, "commit"))

    def install(self, source_path="final.safetensors", filename="identity.safetensors"):
        return main.install_lora.local(source_path, filename)

    def test_missing_lora_does_not_commit(self):
        with self.assertRaisesRegex(FileNotFoundError, "Final trained LoRA"):
            self.install("missing.safetensors")
        self.commit.assert_not_called()
        self.assertFalse(self.destination.exists())

    def test_installs_and_verifies_final_lora(self):
        self.assertEqual(self.install(), str(self.destination))
        self.assertEqual(main.file_sha256(self.source), main.file_sha256(self.destination))
        self.commit.assert_called_once_with()
        self.assertEqual(list(self.destination.parent.iterdir()), [self.destination])

    def test_accepts_nested_source_path(self):
        directory = self.source_root / "output" / "run"
        directory.mkdir(parents=True)
        self.source.rename(directory / "final.safetensors")
        self.install("/output/run/final.safetensors")
        self.assertEqual(self.destination.read_bytes(), b"trained weight")

    def test_reuses_matching_weight(self):
        self.install()
        self.commit.reset_mock()
        with patch.object(main.shutil, "copy2") as copy:
            self.install()
            copy.assert_not_called()
        self.commit.assert_not_called()

    def test_replaces_outdated_weight_without_touching_other_models(self):
        self.destination.parent.mkdir(parents=True)
        self.destination.write_bytes(b"old weight")
        unrelated = self.destination.parent / "unrelated.safetensors"
        unrelated.write_bytes(b"unrelated weight")
        self.install()
        self.assertEqual(self.destination.read_bytes(), self.source.read_bytes())
        self.assertEqual(unrelated.read_bytes(), b"unrelated weight")
        self.commit.assert_called_once_with()

    def test_verification_failure_preserves_old_weight_and_cleans_up(self):
        self.destination.parent.mkdir(parents=True)
        self.destination.write_bytes(b"old weight")
        with patch.object(
            main.shutil, "copy2",
            side_effect=lambda source, target: Path(target).write_bytes(b"corrupted"),
        ):
            with self.assertRaisesRegex(RuntimeError, "SHA-256 verification failed"):
                self.install()
        self.assertEqual(self.destination.read_bytes(), b"old weight")
        self.assertEqual(list(self.destination.parent.iterdir()), [self.destination])
        self.commit.assert_not_called()

    def test_copy_failure_cleans_up_without_committing(self):
        with patch.object(main.shutil, "copy2", side_effect=OSError("copy failed")):
            with self.assertRaisesRegex(OSError, "copy failed"):
                self.install()
        self.assertEqual(list(self.destination.parent.iterdir()), [])
        self.commit.assert_not_called()

    def test_commit_failure_is_reported(self):
        self.commit.side_effect = RuntimeError("commit failed")
        with self.assertRaisesRegex(RuntimeError, "commit failed"):
            self.install()

    def test_rejects_invalid_destination_names(self):
        for filename in ("", ".", "..", "../escape.safetensors", "/tmp/escape.safetensors"):
            with self.subTest(filename=filename):
                with self.assertRaisesRegex(ValueError, "plain filename"):
                    self.install(filename=filename)
        self.commit.assert_not_called()

    def test_rejects_source_path_outside_volume(self):
        with self.assertRaisesRegex(ValueError, "inside the source volume"):
            self.install("../escape.safetensors")
        self.commit.assert_not_called()

    def test_mounts_requested_source_volume_on_cpu_installer(self):
        source_volume = Mock()
        with (
            patch.object(main.modal.Volume, "from_name", return_value=source_volume) as volume,
            patch.object(main.install_lora, "with_options") as configure,
        ):
            main.prepare_lora("training-volume", "output/final.safetensors", "final.safetensors")
            volume.assert_called_once_with("training-volume")
            configure.assert_called_once_with(
                volumes={main.MODELS_DIR: main.models, main.SOURCE_DIR: source_volume}
            )
            configure.return_value.remote.assert_called_once_with(
                "output/final.safetensors", "final.safetensors"
            )


class DownloadTests(unittest.TestCase):
    def test_downloads_requested_model_using_existing_cache(self):
        with patch.object(main, "download_models") as download:
            main.download_model.local("example/base-model", "split_files/model.safetensors", "diffusion_models")
            download.assert_called_once_with(
                "example/base-model",
                [("split_files/model.safetensors", "diffusion_models")],
                main.HUB_CACHE_DIR,
            )

    def test_rejects_unsupported_model_subfolders(self):
        with patch.object(main, "download_models") as download:
            with self.assertRaisesRegex(ValueError, "Unsupported model subfolder"):
                main.download_model.local("example/base-model", "model.safetensors", "../escape")
            download.assert_not_called()

    def test_download_errors_are_reported(self):
        with patch.object(main, "download_models", side_effect=RuntimeError("download failed")):
            with self.assertRaisesRegex(RuntimeError, "download failed"):
                main.download_model.local("example/base-model", "model.safetensors")


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).parent / "workflows" / "aaliya_zimage_lora.json"
        self.workflow = json.loads(path.read_text())
        self.nodes = {node["id"]: node for node in self.workflow["nodes"]}

    def test_model_files_trigger_and_settings(self):
        widgets = {node["type"]: node["widgets_values"] for node in self.nodes.values()}
        self.assertEqual(widgets["UNETLoader"][0], "z_image_bf16.safetensors")
        self.assertEqual(widgets["CLIPLoader"], ["qwen_3_4b.safetensors", "lumina2", "default"])
        self.assertEqual(widgets["VAELoader"], ["ae.safetensors"])
        self.assertEqual(
            widgets["LoraLoaderModelOnly"], ["aaliya_person_zimage_600.safetensors", 0.8]
        )
        self.assertEqual(widgets["EmptySD3LatentImage"], [768, 768, 1])
        self.assertEqual(widgets["KSampler"], [42, "fixed", 30, 4, "euler", "simple", 1])
        prompts = [
            node["widgets_values"][0]
            for node in self.nodes.values()
            if node["type"] == "CLIPTextEncode"
        ]
        self.assertTrue(any("aaliya_person" in prompt for prompt in prompts))
        self.assertEqual(widgets["ModelSamplingAuraFlow"], [3])

    def test_all_connections_are_consistent_and_lora_is_used(self):
        self.assertEqual(len(self.nodes), len(self.workflow["nodes"]))
        link_ids = {link[0] for link in self.workflow["links"]}
        self.assertEqual(len(link_ids), len(self.workflow["links"]))
        for link_id, source_id, source_slot, target_id, target_slot, kind in self.workflow["links"]:
            source = self.nodes[source_id]["outputs"][source_slot]
            target = self.nodes[target_id]["inputs"][target_slot]
            self.assertIn(link_id, source["links"])
            self.assertEqual(target["link"], link_id)
            self.assertEqual(source["type"], kind)
            self.assertEqual(target["type"], kind)
            self.assertLess(self.nodes[source_id]["order"], self.nodes[target_id]["order"])
        self.assertEqual(self.nodes[2]["type"], "LoraLoaderModelOnly")
        self.assertEqual(self.nodes[3]["inputs"][0]["link"], 2)
        self.assertEqual(self.nodes[9]["inputs"][0]["link"], 5)


if __name__ == "__main__":
    unittest.main()
