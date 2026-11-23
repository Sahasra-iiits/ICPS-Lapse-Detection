"""
One-time setup: lets the ICPS Focus Alerts Chrome extension start the local
detector by itself (Chrome "native messaging"). Normally run by double-clicking
install_helper.command (Mac) or install_helper.bat (Windows).

    python install_native_host.py              # install
    python install_native_host.py --model path/to/final_model_face_only.joblib
    python install_native_host.py --uninstall
"""

import argparse
import json
import os
import shutil
import stat
import subprocess
import sys

HOST = "com.icps.focus_alerts"
EXTENSION_ID = "nfopjdpmmbcaofikbdinbelhjonimekn"          # fixed by the "key" in meet_extension/manifest.json
HERE = os.path.dirname(os.path.abspath(__file__))


def chrome_dirs():
    home = os.path.expanduser("~")
    if sys.platform == "darwin":
        base = os.path.join(home, "Library", "Application Support")
        names = ["Google/Chrome", "Google/Chrome Beta", "Chromium", "Microsoft Edge", "BraveSoftware/Brave-Browser"]
    else:
        base = os.path.join(home, ".config")
        names = ["google-chrome", "chromium", "microsoft-edge", "BraveSoftware/Brave-Browser"]
    dirs = [os.path.join(base, n, "NativeMessagingHosts") for n in names if os.path.isdir(os.path.join(base, n))]
    return dirs or [os.path.join(base, names[0], "NativeMessagingHosts")]


WIN_KEYS = [r"Software\Google\Chrome\NativeMessagingHosts", r"Software\Microsoft\Edge\NativeMessagingHosts",
            r"Software\BraveSoftware\Brave-Browser\NativeMessagingHosts"]


def write_launcher():
    if sys.platform == "win32":
        path = os.path.join(HERE, "focus_host_launcher.bat")
        with open(path, "w") as fh:
            fh.write(f'@echo off\r\ncd /d "{HERE}"\r\n"{sys.executable}" -u "{os.path.join(HERE, "focus_host.py")}"\r\n')
    else:
        path = os.path.join(HERE, "focus_host_launcher.sh")
        with open(path, "w") as fh:
            fh.write(f'#!/bin/bash\ncd "{HERE}"\nexec "{sys.executable}" -u "{os.path.join(HERE, "focus_host.py")}"\n')
        os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def manifest(launcher):
    return {"name": HOST, "description": "ICPS Focus Alerts local detector", "path": launcher,
            "type": "stdio", "allowed_origins": [f"chrome-extension://{EXTENSION_ID}/"]}


def install(model):
    try:
        import cv2, mediapipe, sklearn  # noqa: F401
    except ImportError as e:
        sys.exit(f"Missing package ({e.name}). Install the requirements first: pip install -r requirements.txt")
    # model
    os.makedirs(os.path.join(HERE, "models"), exist_ok=True)
    src = model or next((p for p in [os.path.join(HERE, "..", "final_results", "final_model_face_only.joblib")]
                         if os.path.exists(p)), None)
    dst = os.path.join(HERE, "models", "final_model_face_only.joblib")
    if src and os.path.abspath(src) != os.path.abspath(dst):
        shutil.copy(src, dst)
        print("Model:    copied", os.path.basename(src))
    elif os.path.exists(dst):
        print("Model:    models/final_model_face_only.joblib")
    else:
        print("Model:    none found - zoning out will use the simple staring rule.\n"
              "          (put final_model_face_only.joblib in the models/ folder to use the trained model)")
    launcher = write_launcher()
    man = manifest(launcher)
    if sys.platform == "win32":
        man_path = os.path.join(HERE, f"{HOST}.json")
        with open(man_path, "w") as fh:
            json.dump(man, fh, indent=2)
        for key in WIN_KEYS:
            subprocess.run(["reg", "add", rf"HKCU\{key}\{HOST}", "/ve", "/t", "REG_SZ", "/d", man_path, "/f"],
                           check=False, capture_output=True)
        print("Registered for Chrome / Edge / Brave (current user).")
    else:
        for d in chrome_dirs():
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, f"{HOST}.json"), "w") as fh:
                json.dump(man, fh, indent=2)
            print("Registered:", d)
    print(f"\nDone. In Chrome, load (or reload) the extension from:\n  {os.path.join(HERE, 'meet_extension')}\n"
          f"Its ID should be {EXTENSION_ID}. Then switch Focus Alerts on in Google Meet.")


def uninstall():
    if sys.platform == "win32":
        for key in WIN_KEYS:
            subprocess.run(["reg", "delete", rf"HKCU\{key}\{HOST}", "/f"], check=False, capture_output=True)
    else:
        for d in chrome_dirs():
            p = os.path.join(d, f"{HOST}.json")
            if os.path.exists(p):
                os.remove(p)
                print("Removed", p)
    print("Helper unregistered.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="trained model to use (train_final.py output, face-only)")
    ap.add_argument("--uninstall", action="store_true")
    a = ap.parse_args()
    uninstall() if a.uninstall else install(a.model)
