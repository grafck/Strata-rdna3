"""Unit tests for AMD GPU UID selection (e.g. GPU-110069a31ca0e438) across rdna3/run.py, serve/server.py, and serve/telemetry.py."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rdna3 import run as rdna3_run
from serve import server, telemetry

VEGA_UID_DEC = "149739282462019812"      # 0x0213fb15841210e4
XTX_UID_DEC = "1225095247925666872"      # 0x110069a31ca0e438
XTX_UID = "GPU-110069a31ca0e438"
VEGA_UID = "GPU-0213fb15841210e4"


def _make_fake_sysfs(root: Path) -> tuple[Path, Path]:
    """Create a fake KFD + DRM sysfs tree with Vega at KFD index 0 (renderD128) and 7900 XTX at KFD index 1 (renderD129)."""
    kfd = root / "class" / "kfd" / "kfd" / "topology" / "nodes"
    (kfd / "0").mkdir(parents=True)
    (kfd / "0" / "properties").write_text("simd_count 0\ngfx_target_version 0\ndrm_render_minor 0\n", encoding="utf-8")

    (kfd / "1").mkdir(parents=True)
    (kfd / "1" / "properties").write_text(
        f"simd_count 224\ngfx_target_version 90000\ndrm_render_minor 128\nunique_id {VEGA_UID_DEC}\n",
        encoding="utf-8",
    )

    (kfd / "2").mkdir(parents=True)
    (kfd / "2" / "properties").write_text(
        f"simd_count 192\ngfx_target_version 110000\ndrm_render_minor 129\nunique_id {XTX_UID_DEC}\n",
        encoding="utf-8",
    )

    for minor, uid_hex, vram in ((128, "0213fb15841210e4", 8 * 1024**3), (129, "110069a31ca0e438", 24 * 1024**3)):
        d = root / "class" / "drm" / f"renderD{minor}" / "device"
        d.mkdir(parents=True)
        (d / "unique_id").write_text(uid_hex + "\n", encoding="utf-8")
        (d / "mem_info_vram_total").write_text(str(vram) + "\n", encoding="utf-8")
        (d / "mem_info_vram_used").write_text(str(256 * 1024**2) + "\n", encoding="utf-8")
    return root, kfd


class GpuUidSelectionTests(unittest.TestCase):
    def test_parse_gpu_spec_accepts_uid_and_ordinals(self):
        self.assertEqual(rdna3_run._parse_gpu_spec(XTX_UID), [XTX_UID])
        self.assertEqual(rdna3_run._parse_gpu_spec(f"{XTX_UID}, {VEGA_UID}"), [XTX_UID, VEGA_UID])
        self.assertEqual(rdna3_run._parse_gpu_spec("1"), [1])

    def test_kfd_uid_resolution_picks_7900xtx_not_vega(self):
        with tempfile.TemporaryDirectory() as tmp:
            sysfs, kfd = _make_fake_sysfs(Path(tmp))
            self.assertEqual(server._kfd_uid_to_index(VEGA_UID, root=str(kfd)), 0)
            self.assertEqual(server._kfd_uid_to_index(XTX_UID, root=str(kfd)), 1)
            self.assertEqual(telemetry.amd_device_dir(XTX_UID, sysfs=str(sysfs)), str(sysfs / "class" / "drm" / "renderD129" / "device"))
            self.assertEqual(telemetry.amd_device_dir(VEGA_UID, sysfs=str(sysfs)), str(sysfs / "class" / "drm" / "renderD128" / "device"))
            self.assertIsNone(telemetry.amd_device_dir("GPU-deadbeefdeadbeef", sysfs=str(sysfs)))

    def test_child_env_pins_all_visible_devices_to_uid(self):
        cfg = {"backend": "hip", "gpu": [XTX_UID], "args": []}
        with patch.dict(os.environ, {}, clear=True):
            env = server.child_env(cfg)
        self.assertEqual(env["ROCR_VISIBLE_DEVICES"], XTX_UID)
        self.assertEqual(env["HIP_VISIBLE_DEVICES"], XTX_UID)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], XTX_UID)
        self.assertEqual(env["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")

    def test_child_env_preserves_inherited_rocr_uid_even_with_numeric_gpu(self):
        cfg = {"backend": "hip", "gpu": [1], "args": []}
        inherited = {
            "ROCR_VISIBLE_DEVICES": XTX_UID,
            "HIP_VISIBLE_DEVICES": XTX_UID,
            "CUDA_VISIBLE_DEVICES": XTX_UID,
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        }
        with patch.dict(os.environ, inherited, clear=True):
            env = server.child_env(cfg)
        self.assertEqual(env["ROCR_VISIBLE_DEVICES"], XTX_UID)
        self.assertEqual(env["HIP_VISIBLE_DEVICES"], XTX_UID)

    def test_visible_device_order_resolves_uid_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, kfd = _make_fake_sysfs(Path(tmp))
            orig = server._kfd_uid_to_index
            with patch.object(server, "_kfd_uid_to_index", side_effect=lambda u: orig(u, root=str(kfd))):
                order = server.visible_device_order(
                    {"ROCR_VISIBLE_DEVICES": XTX_UID, "HIP_VISIBLE_DEVICES": XTX_UID},
                    "hip",
                )
        self.assertEqual(order, [1])


if __name__ == "__main__":
    unittest.main()
