"""UI helpers — native OS folder picker with cross-platform fallbacks."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys


def pick_folder(initial: str = "") -> str:
    """Open a native OS folder dialog. Returns the selected absolute path, or "" if cancelled/unavailable.

    Strategy:
      1. tkinter.filedialog.askdirectory — stdlib, uses the native dialog on Windows
         (Common Item Dialog) and macOS (NSOpenPanel via Tk 8.6+).
      2. Platform subprocess fallback if tkinter is unavailable:
         - macOS: AppleScript `choose folder` via osascript.
         - Windows: PowerShell + System.Windows.Forms.FolderBrowserDialog.
    """
    initial = initial or os.path.expanduser("~")

    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.wm_attributes("-topmost", 1)
        try:
            path = filedialog.askdirectory(
                master=root,
                initialdir=initial,
                title="選擇 base 資料夾",
                mustexist=True,
            )
        finally:
            root.destroy()
        return path or ""
    except Exception:
        pass

    if sys.platform == "darwin":
        script = (
            'POSIX path of (choose folder with prompt "選擇 base 資料夾" '
            f"default location (POSIX file {shlex.quote(initial)}))"
        )
        try:
            r = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, check=False,
            )
            return r.stdout.strip() if r.returncode == 0 else ""
        except FileNotFoundError:
            return ""

    if sys.platform == "win32":
        initial_ps = initial.replace("'", "''")
        ps = (
            "Add-Type -AssemblyName System.Windows.Forms | Out-Null; "
            "$f = New-Object System.Windows.Forms.FolderBrowserDialog; "
            f"$f.SelectedPath = '{initial_ps}'; "
            "$f.Description = '選擇 base 資料夾'; "
            "if ($f.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { "
            "  [Console]::Out.Write($f.SelectedPath) "
            "}"
        )
        try:
            r = subprocess.run(
                ["powershell", "-NoProfile", "-STA", "-Command", ps],
                capture_output=True, text=True, check=False,
            )
            return r.stdout.strip() if r.returncode == 0 else ""
        except FileNotFoundError:
            return ""

    return ""
