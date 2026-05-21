"""
OCT Segmenter — one-click GUI
Run: python app.py
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)


# ── Interactive vertical-line picker ─────────────────────────────────────────

class VLinePicker(tk.Toplevel):
    """Shows the OCT image; user drags a vertical line to mark the foveal centre."""

    DISPLAY_W = 1100   # max canvas width
    DISPLAY_H = 300    # max canvas height

    def __init__(self, parent, raw_slice: np.ndarray):
        super().__init__(parent)
        self.title('Mark Foveal Centre — drag the line, then click Confirm')
        self.resizable(False, False)
        self.grab_set()          # modal

        H, W = raw_slice.shape
        # Scale to fit display
        scale    = min(self.DISPLAY_W / W, self.DISPLAY_H / H)
        self._dw = int(W * scale)
        self._dh = int(H * scale)
        self._scale  = scale
        self._orig_W = W
        self.result  = None      # column in ORIGINAL image coords

        # Build uint8 image for display (flip vertically to match clinical orientation)
        s    = raw_slice.astype(np.float32)
        u8   = ((s - s.min()) / (s.max() - s.min() + 1e-8) * 255).astype(np.uint8)
        pil  = Image.fromarray(np.flipud(u8), mode='L').resize((self._dw, self._dh), Image.BILINEAR)
        self._tk_img = ImageTk.PhotoImage(pil)

        # Canvas
        self._canvas = tk.Canvas(self, width=self._dw, height=self._dh, cursor='sb_h_double_arrow')
        self._canvas.pack()
        self._canvas.create_image(0, 0, anchor='nw', image=self._tk_img)

        # Vertical line starting at centre
        self._line_x = self._dw // 2
        self._line   = self._canvas.create_line(
            self._line_x, 0, self._line_x, self._dh,
            fill='yellow', width=2, tags='vline')

        # Column label
        self._label_var = tk.StringVar(value=f'Column: {self._orig_W // 2}')
        tk.Label(self, textvariable=self._label_var, font=('Consolas', 10)).pack(pady=4)

        # Buttons
        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=(0, 8))
        tk.Button(btn_frame, text='Confirm', bg='#2196F3', fg='white',
                  font=('', 11, 'bold'), command=self._confirm).pack(side='left', padx=8, ipadx=10)
        tk.Button(btn_frame, text='Skip (no line)', command=self._skip).pack(side='left', padx=8)

        # Drag bindings
        self._canvas.bind('<ButtonPress-1>',   self._on_press)
        self._canvas.bind('<B1-Motion>',        self._on_drag)
        self._canvas.bind('<ButtonRelease-1>',  self._on_drag)

        # Arrow key bindings (1px in display = ~1/scale px in original)
        self._canvas.focus_set()
        self._canvas.bind('<Left>',       lambda e: self._move_line(self._line_x - 1))
        self._canvas.bind('<Right>',      lambda e: self._move_line(self._line_x + 1))
        self._canvas.bind('<Shift-Left>',  lambda e: self._move_line(self._line_x - 10))
        self._canvas.bind('<Shift-Right>', lambda e: self._move_line(self._line_x + 10))

        self.protocol('WM_DELETE_WINDOW', self._skip)

    def _move_line(self, x):
        x = max(0, min(self._dw - 1, x))
        self._line_x = x
        self._canvas.coords('vline', x, 0, x, self._dh)
        orig_col = int(round(x / self._scale))
        self._label_var.set(f'Column: {orig_col}')

    def _on_press(self, event):  self._move_line(event.x)
    def _on_drag(self, event):   self._move_line(event.x)

    def _confirm(self):
        self.result = int(round(self._line_x / self._scale))
        self.destroy()

    def _skip(self):
        self.result = None
        self.destroy()


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('OCT Segmenter')
        self.resizable(False, False)
        self._build_ui()
        self._model_path = os.path.join(BASE_DIR, 'model_img.pth')
        if not os.path.exists(self._model_path):
            self._model_path = os.path.join(BASE_DIR, 'model.pth')

    def _build_ui(self):
        pad = dict(padx=10, pady=6)

        # File selection row
        tk.Label(self, text='Input .img file:', anchor='w').grid(
            row=0, column=0, sticky='w', **pad)
        self._img_var = tk.StringVar(value='No file selected')
        tk.Label(self, textvariable=self._img_var, width=45, anchor='w',
                 relief='sunken').grid(row=0, column=1, **pad)
        tk.Button(self, text='Browse...', command=self._browse).grid(
            row=0, column=2, **pad)

        # Model selection row
        tk.Label(self, text='Model:', anchor='w').grid(
            row=1, column=0, sticky='w', **pad)
        self._model_var = tk.StringVar(value='model_img.pth  (default)')
        tk.Label(self, textvariable=self._model_var, width=45, anchor='w',
                 relief='sunken').grid(row=1, column=1, **pad)
        tk.Button(self, text='Change...', command=self._browse_model).grid(
            row=1, column=2, **pad)

        # Run button
        self._run_btn = tk.Button(self, text='Run Segmentation', font=('', 12, 'bold'),
                                  bg='#2196F3', fg='white', command=self._run)
        self._run_btn.grid(row=2, column=0, columnspan=3, pady=(12, 4), ipadx=20, ipady=6)

        # Gather data button
        self._gather_btn = tk.Button(self, text='Gather Training Data', font=('', 10),
                                     bg='#4CAF50', fg='white', command=self._gather)
        self._gather_btn.grid(row=3, column=0, columnspan=3, pady=(0, 4), ipadx=10, ipady=4)

        # Batch segment button
        self._batch_btn = tk.Button(self, text='Batch Segment Folder', font=('', 10),
                                    bg='#9C27B0', fg='white', command=self._batch)
        self._batch_btn.grid(row=4, column=0, columnspan=3, pady=(0, 4), ipadx=10, ipady=4)

        # Retrain button
        self._train_btn = tk.Button(self, text='Retrain Models', font=('', 10),
                                    bg='#FF9800', fg='white', command=self._retrain)
        self._train_btn.grid(row=5, column=0, columnspan=3, pady=(0, 8), ipadx=10, ipady=4)

        # Log area
        self._log = tk.Text(self, height=12, width=65, state='disabled',
                            bg='#1e1e1e', fg='#d4d4d4', font=('Consolas', 9))
        self._log.grid(row=6, column=0, columnspan=3, padx=10, pady=(0, 10))


    # ── helpers ──────────────────────────────────────────────────────────────

    def _browse(self):
        path = filedialog.askopenfilename(
            title='Select .img file',
            filetypes=[('Analyze image', '*.img'), ('All files', '*.*')])
        if path:
            self._img_path = path
            self._img_var.set(os.path.basename(path))

    def _browse_model(self):
        path = filedialog.askopenfilename(
            title='Select model file',
            initialdir=BASE_DIR,
            filetypes=[('PyTorch model', '*.pth'), ('All files', '*.*')])
        if path:
            self._model_path = path
            self._model_var.set(os.path.basename(path))

    def _log_write(self, text):
        self._log.configure(state='normal')
        self._log.insert('end', text)
        self._log.see('end')
        self._log.configure(state='disabled')
        self.update_idletasks()

    def _retrain(self):
        self._train_btn.configure(state='disabled', text='Training...')
        self._gather_btn.configure(state='disabled')
        self._run_btn.configure(state='disabled')
        threading.Thread(target=self._retrain_worker, daemon=True).start()

    def _retrain_worker(self):
        class _Writer:
            def __init__(self, cb): self._cb = cb
            def write(self, s):
                if s: self._cb(s)
            def flush(self): pass

        old_stdout = sys.stdout
        sys.stdout = _Writer(self._log_write)
        try:
            import train as tr
            import train_vline as trv
            import sys as _sys

            self._log_write('\n--- Training boundary model (60 epochs) ---\n')
            _sys.argv = ['train.py', '--epochs', '60', '--save-path', 'model_img.pth']
            tr.main()

            self._log_write('\n--- Training vertical line model (200 epochs) ---\n')
            _sys.argv = ['train_vline.py', '--epochs', '200', '--save-path', 'vline_model.pth']
            trv.main()

            self._log_write('\nRetraining complete!\n')
            self.after(0, lambda: messagebox.showinfo('Done', 'Both models retrained successfully!'))
        except Exception as e:
            import traceback
            self._log_write(f'\nERROR: {e}\n{traceback.format_exc()}\n')
            self.after(0, lambda: messagebox.showerror('Error', str(e)))
        finally:
            sys.stdout = old_stdout
            self.after(0, lambda: [
                self._train_btn.configure(state='normal', text='Retrain Models'),
                self._gather_btn.configure(state='normal'),
                self._run_btn.configure(state='normal'),
            ])

    def _batch(self):
        if not os.path.exists(self._model_path):
            messagebox.showerror('No model',
                f'Model not found:\n{self._model_path}\n\nRun train.py first.')
            return
        folder = filedialog.askdirectory(title='Select folder to scan for Light.img / Dark.img')
        if not folder:
            return

        # Find all Light.img and Dark.img files recursively
        targets = []
        for dirpath, _, filenames in os.walk(folder):
            for fname in filenames:
                if fname.lower() in ('light.img', 'dark.img'):
                    targets.append(os.path.join(dirpath, fname))
        targets.sort()

        if not targets:
            messagebox.showwarning('None found', 'No Light.img or Dark.img files found in that folder.')
            return

        self._log_write(f'\nFound {len(targets)} file(s) to segment.\n')

        # Load models once
        import torch
        from train import UNet, _read_analyze, NUM_BOUNDARIES
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ckpt   = torch.load(self._model_path, map_location=device)
        model  = UNet(num_bnd=ckpt.get('num_bnd', NUM_BOUNDARIES),
                      features=tuple(ckpt.get('features', (32, 64, 128, 256))))
        model.load_state_dict(ckpt['model_state'])
        model.to(device).eval()

        # For each file: show picker, then segment in background
        for img_path in targets:
            base   = os.path.splitext(img_path)[0]
            arr, _ = _read_analyze(base)
            self._log_write(f'\n{img_path}\n')

            picker = VLinePicker(self, arr[0])
            self.wait_window(picker)
            vline_col = picker.result

            # Segment this file
            self._batch_btn.configure(state='disabled', text='Segmenting...')
            self._run_btn.configure(state='disabled')
            done = threading.Event()

            def _worker(p=img_path, vc=vline_col, m=model, d=device, ev=done):
                import segment_img_via_tiff as seg
                import sys
                class _W:
                    def __init__(self, cb): self._cb = cb
                    def write(self, s):
                        if s: self._cb(s)
                    def flush(self): pass
                old = sys.stdout; sys.stdout = _W(self._log_write)
                try:
                    seg.process_file(p, m, d, vline_model=None, vline_col=vc, out_suffix='_MARKED')
                except Exception as e:
                    self._log_write(f'ERROR: {e}\n')
                finally:
                    sys.stdout = old
                    ev.set()

            t = threading.Thread(target=_worker, daemon=True)
            t.start()
            # Wait for this file to finish before showing next picker
            while not done.wait(timeout=0.05):
                self.update()

            self._batch_btn.configure(state='normal', text='Batch Segment Folder')
            self._run_btn.configure(state='normal')

        messagebox.showinfo('Done', f'Batch segmentation complete.\n{len(targets)} file(s) processed.')

    def _gather(self):
        from gather_training_data import find_pairs, copy_pair, next_index
        src_dir = filedialog.askdirectory(title='Select SOURCE folder to scan for .img pairs')
        if not src_dir:
            return
        dst_dir = os.path.join(BASE_DIR, 'test', 'gathered_data')
        os.makedirs(dst_dir, exist_ok=True)
        pairs = find_pairs(src_dir)
        if not pairs:
            messagebox.showwarning('No pairs found', 'No raw + MARKED .img pairs were found.')
            return
        start = next_index(dst_dir)
        self._log_write(f'\nGathering {len(pairs)} pair(s) starting at scan_{start:03d}...\n')
        for i, (raw_base, marked_base) in enumerate(pairs, start=start):
            copy_pair(raw_base, marked_base, dst_dir, i)
            self._log_write(f'  Copied scan_{i:03d}\n')
        msg = f'{len(pairs)} pair(s) added (scan_{start:03d} to scan_{start+len(pairs)-1:03d})'
        self._log_write(f'{msg}\n')
        messagebox.showinfo('Done', msg)

    def _run(self):
        if not hasattr(self, '_img_path') or not os.path.exists(self._img_path):
            messagebox.showwarning('No file', 'Please select an .img file first.')
            return
        if not os.path.exists(self._model_path):
            messagebox.showerror('No model',
                f'Model not found:\n{self._model_path}\n\nRun train.py first.')
            return
        # Show interactive image picker for vertical line
        import struct
        from train import _read_analyze
        base      = os.path.splitext(self._img_path)[0]
        arr, _    = _read_analyze(base)
        picker    = VLinePicker(self, arr[0])
        self.wait_window(picker)
        self._vline_col = picker.result   # None if skipped
        self._run_btn.configure(state='disabled', text='Running...')
        threading.Thread(target=self._segment_worker, daemon=True).start()

    def _segment_worker(self):
        import importlib, io, contextlib

        self._log_write(f'Input : {self._img_path}\n')
        self._log_write(f'Model : {self._model_path}\n')
        self._log_write('-' * 50 + '\n')

        # Redirect stdout so progress appears in the log box
        class _Writer:
            def __init__(self, cb): self._cb = cb
            def write(self, s):
                if s: self._cb(s)
            def flush(self): pass

        old_stdout = sys.stdout
        sys.stdout = _Writer(self._log_write)

        try:
            import torch
            from train import UNet, load_image, soft_argmax_y, NUM_BOUNDARIES
            from train_vline import VLineNet
            import segment_img_via_tiff as seg

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self._log_write(f'Device: {device}\n')

            ckpt  = torch.load(self._model_path, map_location=device)
            model = UNet(num_bnd=ckpt.get('num_bnd', NUM_BOUNDARIES),
                         features=tuple(ckpt.get('features', (32, 64, 128, 256))))
            model.load_state_dict(ckpt['model_state'])
            model.to(device).eval()

            vline_model = None
            vline_path  = os.path.join(BASE_DIR, 'vline_model.pth')
            if os.path.exists(vline_path):
                vl_ckpt     = torch.load(vline_path, map_location=device)
                vline_model = VLineNet()
                vline_model.load_state_dict(vl_ckpt['model_state'])
                vline_model.to(device).eval()
                self._log_write(f'VLine model loaded\n')
            else:
                self._log_write(f'VLine model not found — using centre fallback\n')

            seg.process_file(self._img_path, model, device,
                             vline_model=None, vline_col=self._vline_col)

            out = os.path.splitext(self._img_path)[0] + '_segmented.img'
            self._log_write(f'\nDone! Output saved to:\n{out}\n')
            self.after(0, lambda: messagebox.showinfo(
                'Done', f'Segmentation complete!\n\nSaved to:\n{out}'))

        except Exception as e:
            import traceback
            self._log_write(f'\nERROR: {e}\n{traceback.format_exc()}\n')
            self.after(0, lambda: messagebox.showerror('Error', str(e)))
        finally:
            sys.stdout = old_stdout
            self.after(0, lambda: self._run_btn.configure(
                state='normal', text='Run Segmentation'))


if __name__ == '__main__':
    App().mainloop()
