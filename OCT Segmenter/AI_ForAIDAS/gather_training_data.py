"""
Gather .img training pairs into a single training folder.
Usage: python gather_training_data.py

Opens folder-picker dialogs to select:
  1. Source folder  -- scanned recursively for raw + MARKED .img pairs
  2. Destination    -- where renamed pairs are copied to

Output naming: scan_001.img / scan_001.hdr / scan_001_MARKED.img / scan_001_MARKED.hdr
"""

import os
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox


def find_pairs(root_dir):
    """Walk root_dir and return list of (raw_base, marked_base) path pairs."""
    pairs = []
    for dirpath, _, filenames in os.walk(root_dir):
        img_files = {f for f in filenames if f.lower().endswith('.img')}
        for fname in img_files:
            if '_MARKED' in fname.upper():
                continue  # skip MARKED files on this pass
            stem     = os.path.splitext(fname)[0]
            # Look for a MARKED counterpart with any capitalisation
            for candidate in img_files:
                c_stem = os.path.splitext(candidate)[0]
                if c_stem.upper() == (stem + '_MARKED').upper():
                    raw_base    = os.path.join(dirpath, stem)
                    marked_base = os.path.join(dirpath, c_stem)
                    pairs.append((raw_base, marked_base))
                    break
    return pairs


def copy_pair(raw_base, marked_base, dest_dir, index):
    """Copy one pair (+ headers) to dest_dir with sequential naming."""
    tag  = f'scan_{index:03d}'
    copied = []
    for src_base, suffix in [(raw_base, ''), (marked_base, '_MARKED')]:
        for ext in ['.img', '.hdr']:
            src = src_base + ext
            if os.path.exists(src):
                dst = os.path.join(dest_dir, tag + suffix + ext)
                shutil.copy2(src, dst)
                copied.append(os.path.basename(dst))
    return copied


def next_index(dst_dir):
    """Return the next available scan index (continues from existing files)."""
    import re
    existing = [f for f in os.listdir(dst_dir) if re.match(r'scan_\d+\.img$', f, re.IGNORECASE)
                and 'MARKED' not in f.upper()]
    if not existing:
        return 1
    nums = [int(re.search(r'(\d+)', f).group(1)) for f in existing]
    return max(nums) + 1


def main():
    root = tk.Tk()
    root.withdraw()

    # Pick source folder
    messagebox.showinfo('Step 1', 'Select the SOURCE folder to scan for .img pairs')
    src_dir = filedialog.askdirectory(title='Select source folder')
    if not src_dir:
        print('Cancelled.')
        return

    # Pick destination folder
    messagebox.showinfo('Step 2', 'Select the DESTINATION folder to copy pairs into')
    dst_dir = filedialog.askdirectory(title='Select destination folder')
    if not dst_dir:
        print('Cancelled.')
        return

    os.makedirs(dst_dir, exist_ok=True)

    print(f'Scanning: {src_dir}')
    pairs = find_pairs(src_dir)

    if not pairs:
        messagebox.showwarning('No pairs found',
            'No raw + MARKED .img pairs were found in the selected folder.')
        return

    start = next_index(dst_dir)
    print(f'Found {len(pairs)} pair(s).  Starting from scan_{start:03d}.  Copying to: {dst_dir}')
    for i, (raw_base, marked_base) in enumerate(pairs, start=start):
        files = copy_pair(raw_base, marked_base, dst_dir, i)
        print(f'  [{i:03d}]  {os.path.basename(raw_base)}  ->  ' +
              ', '.join(files))

    msg = f'Done. {len(pairs)} new pair(s) added (scan_{start:03d} to scan_{start+len(pairs)-1:03d}).\n\nDestination:\n{dst_dir}'
    print(msg)
    messagebox.showinfo('Done', msg)


if __name__ == '__main__':
    main()
