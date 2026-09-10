#!/usr/bin/env python3
"""QUDA gauss-gauge 单功能入口；先验证 QDP↔PyQCU 规范场排布。"""
from common import layout_roundtrip, identity_gauge, pyqcu_gauge_to_quda, quda_status

def main():
    r = layout_roundtrip(); print({"function":"gauss-gauge", **r, "quda":quda_status(), "status":"layout-reference"})
    assert r["gauge_error"] == 0.0 and r["fermion_error"] == 0.0
    return 0

def test_gauss_gauge_layout(): assert main() == 0
if __name__ == "__main__": raise SystemExit(main())
