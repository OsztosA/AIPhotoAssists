"""Tkinter GUI for classify_and_move: configure parameters, run and watch progress."""
import json
import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import classify_and_move as cm
import LLMBackend
import LLMCapacity

try:  # Pillow is optional: without it the in-progress preview is simply disabled
    from PIL import Image, ImageOps, ImageTk
except ImportError:  # pragma: no cover
    Image = ImageOps = ImageTk = None

SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gui_settings.json")
THUMB_SIZE = 240  # max edge (px) of the in-progress preview thumbnails


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PhotoAssist - Classify & Move")
        self.geometry("900x760")
        self.minsize(760, 600)

        self.msg_queue: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None

        s = self._load_settings()
        self.source_var = tk.StringVar(value=s.get("source", ""))
        self.target_var = tk.StringVar(value=s.get("target", ""))
        self.api_var = tk.StringVar(value=s.get("api_url", cm.API_URL))
        # The backend (OpenAI-compatible vs. Ollama) follows from the URL; remember one URL per backend
        self.backend_var = tk.StringVar(value=LLMBackend.backend_for_url(self.api_var.get()))
        self.backend_urls = dict(LLMBackend.DEFAULT_URLS)
        self.backend_urls.update({k: v for k, v in (s.get("api_urls") or {}).items() if k in self.backend_urls})
        self.backend_urls[self.backend_var.get()] = self.api_var.get()
        self.model_var = tk.StringVar(value=s.get("model", cm.DEFAULT_MODEL))
        self.workers_var = tk.IntVar(value=s.get("workers", 5))
        self.dry_run_var = tk.BooleanVar(value=s.get("dry_run", True))  # safe default
        self.tags_var = tk.BooleanVar(value=s.get("with_tags", True))
        self.resize_var = tk.IntVar(value=min(100, max(25, int(s.get("resize_percent", 100)))))
        self.tag_count_var = tk.IntVar(value=max(1, int(s.get("tag_count", 5))))
        self.limit: int | None = None          # detected max parallel requests for current api+model
        self.model_types: dict[str, str | None] = {}  # model id -> 'vlm' / 'llm' / ... (LM Studio)
        self.limit_key: tuple | None = None
        self.limit_var = tk.StringVar(value="unknown - click Detect (also runs automatically on Start)")
        self.status_var = tk.StringVar(value="Ready.")
        self.stats_var = tk.StringVar(value="")
        self.counts_var = tk.StringVar(value="")

        self._build_ui()
        self.api_var.trace_add("write", lambda *_: self._on_url_changed())
        self.api_var.trace_add("write", lambda *_: self._invalidate_limit())
        self.model_var.trace_add("write", lambda *_: self._invalidate_limit())
        # Persist every setting as soon as it changes (debounced), not only on Start / close
        self._save_job = None
        for var in (self.source_var, self.target_var, self.api_var, self.model_var,
                    self.workers_var, self.dry_run_var, self.tags_var,
                    self.resize_var, self.tag_count_var):
            var.trace_add("write", lambda *_: self._schedule_save())
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._poll_queue)
        self.after(200, self.refresh_models)

    # ---------- UI ----------
    def _build_ui(self):
        pad = {"padx": 8, "pady": 4}
        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)

        # Folders
        folders = ttk.LabelFrame(root, text="Folders", padding=8)
        folders.pack(fill="x", **pad)
        folders.columnconfigure(1, weight=1)
        self._folder_row(folders, 0, "Source folder:", self.source_var)
        self._folder_row(folders, 1, "Target folder:", self.target_var)

        # Model / server
        model = ttk.LabelFrame(root, text="Model", padding=8)
        model.pack(fill="x", **pad)
        model.columnconfigure(1, weight=1)
        ttk.Label(model, text="Backend:").grid(row=0, column=0, sticky="w", **pad)
        backend_combo = ttk.Combobox(model, textvariable=self.backend_var, values=list(LLMBackend.BACKENDS),
                                     state="readonly")
        backend_combo.grid(row=0, column=1, sticky="ew", **pad)
        backend_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_backend_selected())
        ttk.Label(model, text="API URL:").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(model, textvariable=self.api_var).grid(row=1, column=1, columnspan=2, sticky="ew", **pad)
        ttk.Label(model, text="Model:").grid(row=2, column=0, sticky="w", **pad)
        self.model_combo = ttk.Combobox(model, textvariable=self.model_var, values=[cm.DEFAULT_MODEL])
        self.model_combo.grid(row=2, column=1, sticky="ew", **pad)
        self.refresh_btn = ttk.Button(model, text="Refresh models", command=self.refresh_models)
        self.refresh_btn.grid(row=2, column=2, **pad)
        ttk.Label(model, text="Parallel requests:").grid(row=3, column=0, sticky="w", **pad)
        ttk.Label(model, textvariable=self.limit_var).grid(row=3, column=1, sticky="w", **pad)
        self.detect_btn = ttk.Button(model, text="Detect", command=self.detect_limit)
        self.detect_btn.grid(row=3, column=2, sticky="ew", **pad)

        # Options
        opts = ttk.LabelFrame(root, text="Options", padding=8)
        opts.pack(fill="x", **pad)
        ttk.Checkbutton(opts, text="Dry run (don't move any files)", variable=self.dry_run_var).pack(side="left", **pad)
        ttk.Checkbutton(opts, text="Ask LLM for tags", variable=self.tags_var).pack(side="left", **pad)
        ttk.Label(opts, text="Workers:").pack(side="left", padx=(24, 4))
        self.workers_spin = ttk.Spinbox(opts, from_=1, to=32, width=5, textvariable=self.workers_var)
        self.workers_spin.pack(side="left")
        ttk.Label(opts, text="(limited to the model's max parallel requests)", foreground="#666").pack(
            side="left", padx=8)

        opts2 = ttk.LabelFrame(root, text="LLM request", padding=8)
        opts2.pack(fill="x", **pad)
        ttk.Label(opts2, text="Tags to generate:").pack(side="left", padx=(8, 4))
        ttk.Spinbox(opts2, from_=1, to=50, width=5, textvariable=self.tag_count_var).pack(side="left")
        ttk.Label(opts2, text="Image size sent to LLM:").pack(side="left", padx=(24, 4))
        self.resize_label = ttk.Label(opts2, width=5)
        ttk.Scale(opts2, from_=25, to=100, orient="horizontal", length=180, variable=self.resize_var,
                  command=lambda v: self.resize_var.set(round(float(v)))).pack(side="left")
        self.resize_label.pack(side="left", padx=4)
        self.resize_var.trace_add("write", lambda *_: self.resize_label.configure(text=f"{self.resize_var.get()}%"))
        self.resize_label.configure(text=f"{self.resize_var.get()}%")

        # Controls
        controls = ttk.Frame(root)
        controls.pack(fill="x", **pad)
        self.start_btn = ttk.Button(controls, text="▶  Start", command=self.start)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(controls, text="■  Stop", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=8)

        # Progress
        prog = ttk.LabelFrame(root, text="Progress", padding=8)
        prog.pack(fill="x", **pad)
        self.progress = ttk.Progressbar(prog, mode="determinate", maximum=100)
        self.progress.pack(fill="x", pady=(0, 4))
        ttk.Label(prog, textvariable=self.status_var).pack(anchor="w")
        ttk.Label(prog, textvariable=self.stats_var).pack(anchor="w")
        ttk.Label(prog, textvariable=self.counts_var).pack(anchor="w")

        # Small preview of the images currently being processed
        self._build_preview(root, pad)

        # Results / Log tabs
        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True, **pad)
        self._build_results_tab(notebook)

        log_frame = ttk.Frame(notebook, padding=4)
        notebook.add(log_frame, text="Log")
        self.log_text = tk.Text(log_frame, height=10, wrap="none", state="disabled")
        ysb = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=ysb.set)
        ysb.pack(side="right", fill="y")
        self.log_text.pack(side="left", fill="both", expand=True)

    def _build_preview(self, root, pad):
        self.preview_frame = ttk.LabelFrame(root, text="In progress", padding=4)
        self.preview_frame.pack(fill="x", **pad)
        self.preview_inner = ttk.Frame(self.preview_frame, height=THUMB_SIZE + 24)
        self.preview_inner.pack(fill="x")
        self.preview_inner.pack_propagate(False)  # constant height: the layout must not jump around
        self._previews: dict[str, ttk.Frame] = {}  # image path -> thumbnail widget
        if Image is None:
            ttk.Label(self.preview_inner, foreground="#666",
                      text="Install Pillow (pip install pillow) to see previews of the images in progress.").pack(anchor="w")

    @staticmethod
    def _make_thumbnail(path: str):
        """Loads a small thumbnail (PIL image). Runs in a worker thread; returns None on failure."""
        if Image is None:
            return None
        try:
            with Image.open(path) as im:
                im.draft("RGB", (THUMB_SIZE * 2, THUMB_SIZE * 2))  # fast JPEG downscaled decoding
                thumb = ImageOps.exif_transpose(im)
                thumb.thumbnail((THUMB_SIZE, THUMB_SIZE))
                return thumb.convert("RGB")
        except Exception:
            return None

    def _preview_start(self, path: str):
        """Called from a worker thread when an image starts being processed."""
        self.msg_queue.put(("preview_add", (path, self._make_thumbnail(path))))

    def _add_preview(self, path: str, thumb):
        if Image is None or path in self._previews:
            return
        cell = ttk.Frame(self.preview_inner)
        photo = ImageTk.PhotoImage(thumb) if thumb is not None else None
        lbl = ttk.Label(cell, image=photo, text="" if photo else "?")
        lbl.image = photo  # keep a reference, or Tk garbage-collects the image
        lbl.pack()
        name = os.path.basename(path)
        ttk.Label(cell, text=name if len(name) <= 16 else name[:7] + "..." + name[-6:],
                  font=("Segoe UI", 8)).pack()
        cell.pack(side="left", padx=4)
        self._previews[path] = cell

    def _remove_preview(self, path: str):
        cell = self._previews.pop(path, None)
        if cell is not None:
            cell.destroy()

    def _clear_previews(self):
        for path in list(self._previews):
            self._remove_preview(path)

    def _build_results_tab(self, notebook):
        frame = ttk.Frame(notebook, padding=4)
        notebook.add(frame, text="Results")

        # Score distribution (10 buckets: 0-9, 10-19, ... 90-100)
        self.score_counts = [0] * 10
        self.hist = tk.Canvas(frame, height=90, highlightthickness=0)
        self.hist.pack(fill="x", pady=(0, 6))
        self.hist.bind("<Configure>", lambda e: self._draw_histogram())

        cols = ("file", "score", "tags", "status", "dest")
        self.tree = ttk.Treeview(frame, columns=cols, show="headings", selectmode="browse")
        headings = {"file": "File", "score": "Score", "tags": "Tags", "status": "Result", "dest": "Destination"}
        widths = {"file": 160, "score": 55, "tags": 260, "status": 80, "dest": 300}
        for c in cols:
            self.tree.heading(c, text=headings[c], command=lambda col=c: self._sort_tree(col))
            self.tree.column(c, width=widths[c], anchor="center" if c == "score" else "w",
                             stretch=c in ("file", "tags", "dest"))
        self.tree.tag_configure("good", foreground="#1a7f37")
        self.tree.tag_configure("bad", foreground="#c62828")
        self.tree.tag_configure("failed", foreground="#9e9e9e")
        ysb = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=ysb.set)
        ysb.pack(side="right", fill="y")
        self.tree.pack(side="left", fill="both", expand=True)
        self.tree.bind("<Double-1>", self._open_selected)
        self._sort_reverse = {}
        self._result_paths: dict[str, str] = {}  # tree item id -> best path to open

    def _clear_results(self):
        self.tree.delete(*self.tree.get_children())
        self._result_paths.clear()
        self.score_counts = [0] * 10
        self._draw_histogram()

    def _add_result(self, r: dict):
        score, status = r["score"], r["status"]
        if status == "moved":
            label = "would move" if self.dry_run_var.get() else "moved"
        else:
            label = status
        tag = "failed" if status in ("failed", "skipped") and score is None else (
            "good" if score is not None and score >= 70 else "bad" if score is not None and score < 40 else "")
        item = self.tree.insert("", "end", values=(
            os.path.basename(r["path"]), "" if score is None else score, ", ".join(r.get("tags") or []), label,
            r["dest"] or r["path"]),
            tags=(tag,) if tag else ())
        dest = r["dest"]
        self._result_paths[item] = dest if dest and os.path.exists(dest) else r["path"]
        if score is not None:
            self.score_counts[min(score // 10, 9)] += 1
            self._draw_histogram()
        self.tree.see(item)

    def _draw_histogram(self):
        c = self.hist
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        if w < 50:
            return
        mx = max(self.score_counts) or 1
        slot = (w - 10) / 10
        base = h - 16
        for i, n in enumerate(self.score_counts):
            x0 = 5 + i * slot + 4
            x1 = 5 + (i + 1) * slot - 4
            bar_h = (base - 14) * n / mx
            color = "#c62828" if i < 4 else "#f9a825" if i < 7 else "#1a7f37"
            c.create_rectangle(x0, base - bar_h, x1, base, fill=color, outline="")
            if n:
                c.create_text((x0 + x1) / 2, base - bar_h - 7, text=str(n), font=("Segoe UI", 8))
            label = f"{i * 10}-{i * 10 + 9}" if i < 9 else "90-100"
            c.create_text((x0 + x1) / 2, h - 7, text=label, font=("Segoe UI", 8))

    def _sort_tree(self, col):
        rows = [(self.tree.set(i, col), i) for i in self.tree.get_children("")]
        if col == "score":
            key = lambda t: float(t[0]) if t[0] != "" else -1.0
        else:
            key = lambda t: t[0].lower()
        reverse = not self._sort_reverse.get(col, True)
        self._sort_reverse[col] = reverse
        for idx, (_, item) in enumerate(sorted(rows, key=key, reverse=reverse)):
            self.tree.move(item, "", idx)

    def _open_selected(self, _event=None):
        sel = self.tree.selection()
        path = self._result_paths.get(sel[0]) if sel else None
        if path and os.path.exists(path):
            try:
                os.startfile(path)
            except (OSError, AttributeError):
                pass

    def _folder_row(self, parent, row, label, var):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=8, pady=4)
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, sticky="ew", padx=8, pady=4)
        ttk.Button(parent, text="Browse...", command=lambda: self._browse(var)).grid(row=row, column=2, padx=8, pady=4)

    def _browse(self, var):
        # Start in the last used folder, or the closest existing parent if it no longer exists / not yet created
        initial = os.path.abspath(var.get()) if var.get().strip() else None
        while initial and not os.path.isdir(initial):
            parent = os.path.dirname(initial)
            initial = parent if parent != initial else None
        path = filedialog.askdirectory(initialdir=initial, title="Select folder")
        if path:
            var.set(os.path.normpath(path))

    # ---------- Backend ----------
    def _on_url_changed(self):
        """Keeps the backend selector and the per-backend URL memory in sync with a hand-edited URL."""
        url = self.api_var.get()
        backend = LLMBackend.backend_for_url(url)
        self.backend_urls[backend] = url
        if self.backend_var.get() != backend:
            self.backend_var.set(backend)

    def _on_backend_selected(self):
        """User switched the backend: restore that backend's URL and reload its model list."""
        self.api_var.set(self.backend_urls[self.backend_var.get()])
        self.model_combo.configure(values=[])
        self.model_types = {}
        self.refresh_models()

    # ---------- Settings ----------
    def _load_settings(self) -> dict:
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def _save_settings(self):
        if self._save_job is not None:
            self.after_cancel(self._save_job)
            self._save_job = None
        data = {
            "source": self.source_var.get(), "target": self.target_var.get(),
            "api_url": self.api_var.get(), "api_urls": self.backend_urls, "model": self.model_var.get(),
            "workers": self._workers(), "dry_run": self.dry_run_var.get(),
            "with_tags": self.tags_var.get(),
            "resize_percent": self._resize_percent(), "tag_count": self._tag_count(),
        }
        try:
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except OSError:
            pass

    def _schedule_save(self):
        if self._save_job is not None:
            self.after_cancel(self._save_job)
        self._save_job = self.after(500, self._save_settings)

    def _resize_percent(self) -> int:
        try:
            return min(100, max(25, int(self.resize_var.get())))
        except (tk.TclError, ValueError):
            return 100

    def _tag_count(self) -> int:
        try:
            return max(1, int(self.tag_count_var.get()))
        except (tk.TclError, ValueError):
            return 5

    def _workers(self) -> int:
        try:
            n = max(1, int(self.workers_var.get()))
        except (tk.TclError, ValueError):
            n = 5
        return min(n, self.limit) if self.limit else n

    # ---------- Parallel request limit ----------
    def _current_key(self) -> tuple:
        return self.api_var.get().strip(), self.model_var.get().strip() or cm.DEFAULT_MODEL

    def _invalidate_limit(self):
        if self.limit_key is not None and self.limit_key != self._current_key():
            self.limit = None
            self.limit_key = None
            self.limit_var.set("unknown - click Detect (also runs automatically on Start)")
            self.workers_spin.configure(to=32)

    def detect_limit(self, then=None):
        """Detects the max parallel requests in a background thread; calls `then()` on success."""
        key = self._current_key()
        self.detect_btn.configure(state="disabled")
        self.start_btn.configure(state="disabled")
        self.limit_var.set("detecting... (may load the model and take a while)")
        self._detect_then = then
        log = lambda m: self.msg_queue.put(("log", f"[detect] {m}"))

        def work():
            try:
                n, src = LLMCapacity.detect_parallel_limit(key[0], key[1], log=log)
                self.msg_queue.put(("limit", (key, n, src)))
            except Exception as e:
                self.msg_queue.put(("limit_error", str(e)))

        threading.Thread(target=work, daemon=True).start()

    def _set_limit(self, key, n, src):
        self.detect_btn.configure(state="normal")
        self.start_btn.configure(state="normal")
        if key != self._current_key():  # user changed model/url meanwhile
            self.limit_var.set("unknown - click Detect (also runs automatically on Start)")
            return
        self.limit, self.limit_key = n, key
        self.limit_var.set(f"max {n} ({src})")
        self.workers_spin.configure(to=n)
        if self._workers() != self.workers_var.get():
            self.workers_var.set(self._workers())
        self._append_log(f"Max parallel requests for '{key[1]}': {n} ({src}). Workers limited to {n}.")
        then, self._detect_then = getattr(self, "_detect_then", None), None
        if then:
            then()

    def _set_limit_error(self, msg):
        self.detect_btn.configure(state="normal")
        self.start_btn.configure(state="normal")
        self.limit_var.set("detection failed - see log")
        self._detect_then = None
        self._append_log(f"Could not detect parallel request limit: {msg}")
        messagebox.showerror("Detection failed", f"Could not determine the max parallel requests:\n\n{msg}")

    # ---------- Models ----------
    def refresh_models(self):
        self.refresh_btn.configure(state="disabled")
        api_url = self.api_var.get()

        def work():
            try:
                self.msg_queue.put(("models", cm.fetch_model_infos(api_url)))
            except Exception as e:
                self.msg_queue.put(("models_error", str(e)))

        threading.Thread(target=work, daemon=True).start()

    # ---------- Run control ----------
    def start(self):
        source, target = self.source_var.get().strip(), self.target_var.get().strip()
        if not os.path.isdir(source):
            messagebox.showerror("Source folder", "Please select an existing source folder.")
            return
        if not target:
            messagebox.showerror("Target folder", "Please select a target folder.")
            return
        if os.path.normcase(os.path.abspath(source)) == os.path.normcase(os.path.abspath(target)):
            messagebox.showerror("Folders", "Source and target folder must be different.")
            return
        if not self._is_vision_model_ok():
            return
        if not self.dry_run_var.get() and not messagebox.askyesno(
                "Move files?", "Dry run is OFF. Images will be MOVED from the source to the target folder.\n\nContinue?"):
            return

        self._save_settings()
        # The number of workers must never exceed what the model can process in parallel
        if self.limit_key == self._current_key():
            self._launch()
        else:
            self.status_var.set("Detecting max parallel requests...")
            self.detect_limit(then=self._launch)

    def _launch(self):
        source, target = self.source_var.get().strip(), self.target_var.get().strip()
        self.stop_event.clear()
        self._set_running(True)
        self.progress.configure(value=0)
        self.status_var.set("Starting...")
        self.stats_var.set("")
        self.counts_var.set("")
        self._clear_results()
        self._clear_previews()

        args = dict(
            root_path=source, output_directory=target, max_workers=self._workers(),
            dry_run=self.dry_run_var.get(), api_url=self.api_var.get().strip(),
            model=self.model_var.get().strip() or cm.DEFAULT_MODEL,
            with_tags=self.tags_var.get(),
            resize_percent=self._resize_percent(), tag_count=self._tag_count(),
        )

        def work():
            try:
                cm.run_classification(
                    **args,
                    log=lambda m: self.msg_queue.put(("log", m)),
                    progress=lambda s: self.msg_queue.put(("progress", s)),
                    stop_event=self.stop_event,
                    on_result=lambda r: self.msg_queue.put(("result", r)),
                    on_start=self._preview_start,
                )
            except Exception as e:
                self.msg_queue.put(("log", f"Unexpected error: {e}"))
            finally:
                self.msg_queue.put(("done", None))

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def stop(self):
        self.stop_event.set()
        self.stop_btn.configure(state="disabled")
        self.status_var.set("Stopping... waiting for running requests to finish.")

    def _set_running(self, running: bool):
        self.start_btn.configure(state="disabled" if running else "normal")
        self.stop_btn.configure(state="normal" if running else "disabled")
        self.detect_btn.configure(state="disabled" if running else "normal")

    # ---------- Queue handling ----------
    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.msg_queue.get_nowait()
                if kind == "log":
                    self._append_log(payload)
                elif kind == "progress":
                    self._update_progress(payload)
                elif kind == "result":
                    self._remove_preview(payload["path"])
                    self._add_result(payload)
                elif kind == "preview_add":
                    self._add_preview(*payload)
                elif kind == "models":
                    self._set_models(payload)
                elif kind == "limit":
                    self._set_limit(*payload)
                elif kind == "limit_error":
                    self._set_limit_error(payload)
                elif kind == "models_error":
                    self.refresh_btn.configure(state="normal")
                    self._append_log(f"Could not load models from server: {payload}")
                elif kind == "done":
                    self._clear_previews()
                    self._set_running(False)
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def _is_vision_model_ok(self) -> bool:
        """Blocks models that LM Studio reports as non-vision (they reject images with HTTP 400)."""
        model = self.model_var.get().strip()
        mtype = self.model_types.get(model)
        if mtype is not None and mtype != "vlm":
            messagebox.showerror(
                "Model cannot read images",
                f"'{model}' is a '{mtype}' model without image support, so the server rejects every "
                f"image with HTTP 400.\n\nPick a vision model (type 'vlm') from the list.")
            return False
        return True

    def _set_models(self, infos):
        self.refresh_btn.configure(state="normal")
        if not infos:
            self._append_log("Server returned no models.")
            return
        self.model_types = {m["id"]: m["type"] for m in infos}
        has_types = any(m["type"] for m in infos)
        # Only vision models can classify images; hide the rest when the server tells us the type
        models = [m["id"] for m in infos if not has_types or m["type"] == "vlm"] or [m["id"] for m in infos]
        self.model_combo.configure(values=models)
        if self.model_var.get() not in models:
            self.model_var.set(models[0])
        hidden = len(infos) - len(models)
        self._append_log(f"Loaded {len(models)} vision model(s) from server"
                         + (f" ({hidden} text/embedding-only model(s) hidden)." if hidden else "."))

    def _append_log(self, text: str):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text + "\n")
        # keep the log from growing without bound
        if int(self.log_text.index("end-1c").split(".")[0]) > 5000:
            self.log_text.delete("1.0", "1000.0")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _update_progress(self, s: dict):
        total, done = s["total"], s["done"]
        self.progress.configure(value=(done / total * 100) if total else 0)
        if s["finished"]:
            self.status_var.set(f"Finished: {done}/{total} images.")
        elif not self.stop_event.is_set():
            self.status_var.set(f"Processing {done}/{total} images ({done / total * 100:.0f}%)")
        self.stats_var.set(
            f"Rate: {s['rate']:.2f} img/s   |   Avg: {s['avg_time']:.2f} s/img   |   "
            f"Elapsed: {cm._format_seconds(s['elapsed'])}   |   ETA: {cm._format_seconds(s['eta'])}")
        c = s["counts"]
        verb = "Would move" if self.dry_run_var.get() else "Moved"
        self.counts_var.set(f"{verb}: {c['moved']}   Skipped: {c['skipped']}   Failed: {c['failed']}")

    def _on_close(self):
        if self.worker and self.worker.is_alive():
            if not messagebox.askyesno("Quit", "Processing is still running. Stop and quit?"):
                return
            self.stop_event.set()
        self._save_settings()
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
