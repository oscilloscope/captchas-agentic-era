# metrics.py - Per-session evaluation counters dumped to metrics.txt.
# Each run appends a new session to metrics_state.json on first event.
import json
from datetime import datetime
from pathlib import Path


class Metrics:
    STATE_FILE = "metrics_state.json"

    SESSION_KEYS = [
        'distinct_puzzles',
        'distinct_dynamic_puzzles',
        'classification_analyses',
        'segmentation_analyses',
        'dynamic_analyses',
        'yolo_cell_calls',
        'vlm_cell_calls',
        'yolo_grid_calls',
        'vlm_grid_calls',
        'successes',
        'reloads',
    ]

    def __init__(self, path="metrics.txt"):
        self.path = Path(path)
        self.state_path = Path(self.STATE_FILE)

        # Load prior sessions. Current session created lazy on first event.
        self.sessions = self._load_sessions()
        self.current = None

        # Runtime flag (not persisted): True between mark_new_puzzle and the next
        # add_classification/add_segmentation, so add_dynamic can attribute to the
        # puzzle's first analysis only.
        self._pending_new_puzzle = False

    # ---------- session lifecycle ----------
    def _new_session(self):
        now = datetime.now().isoformat()
        return {
            'start_time': now,
            'end_time': now,
            **{key: 0 for key in self.SESSION_KEYS},
        }

    def _ensure_session(self):
        if self.current is None:
            self.current = self._new_session()
            self.sessions.append(self.current)

    # ---------- state IO ----------
    def _load_sessions(self):
        if not self.state_path.exists():
            return []
        try:
            data = json.loads(self.state_path.read_text())
            if isinstance(data, dict) and 'sessions' in data:
                return list(data.get('sessions', []))
            # Old cumulative format: fold into a single legacy session entry
            if isinstance(data, dict):
                legacy = self._new_session()
                for key in self.SESSION_KEYS:
                    legacy[key] = data.get(key, 0)
                legacy['start_time'] = data.get('start_time', legacy['start_time'])
                legacy['end_time'] = legacy['start_time']
                return [legacy]
            return []
        except Exception:
            return []

    def _save_state(self):
        try:
            self.state_path.write_text(json.dumps({'sessions': self.sessions}, indent=2))
        except Exception:
            pass

    # ---------- counter API ----------
    def _bump(self, key, n=1):
        self._ensure_session()
        self.current[key] = self.current.get(key, 0) + n
        self.current['end_time'] = datetime.now().isoformat()

    def add_new_puzzle(self):
        self._bump('distinct_puzzles')
        self._pending_new_puzzle = True
        self.dump()

    def add_classification(self):
        self._bump('classification_analyses')
        self._pending_new_puzzle = False
        self.dump()

    def add_segmentation(self):
        self._bump('segmentation_analyses')
        self._pending_new_puzzle = False
        self.dump()

    def add_dynamic(self):
        self._bump('dynamic_analyses')
        # Count distinct dynamic puzzles only on the first analysis after a new puzzle.
        if self._pending_new_puzzle:
            self._bump('distinct_dynamic_puzzles')

    def add_yolo_cells(self, n=1):
        self._bump('yolo_cell_calls', n)

    def add_vlm_cells(self, n=1):
        self._bump('vlm_cell_calls', n)

    def add_yolo_grid(self):
        self._bump('yolo_grid_calls')

    def add_vlm_grid(self):
        self._bump('vlm_grid_calls')

    def add_success(self):
        self._bump('successes')
        self.dump()

    def add_reload(self):
        self._bump('reloads')
        self.dump()

    # ---------- text output ----------
    def dump(self):
        if self.current is None:
            return
        s = self.current
        total_cells = s['yolo_cell_calls'] + s['vlm_cell_calls']
        vlm_rate = (s['vlm_cell_calls'] / total_cells * 100) if total_cells else 0.0
        start_dt = datetime.fromisoformat(s['start_time'])
        end_dt = datetime.fromisoformat(s['end_time'])
        elapsed = (end_dt - start_dt).total_seconds()
        session_idx = len(self.sessions)  # current is the last appended

        text = (
            f"Session {session_idx} Metrics  (elapsed: {elapsed:.0f}s, started {start_dt.strftime('%Y-%m-%d %H:%M:%S')})\n"
            f"=================================================================\n"
            f"Distinct puzzles encountered: {s['distinct_puzzles']}\n"
            f"  of which dynamic: {s.get('distinct_dynamic_puzzles', 0)}\n"
            f"\n"
            f"Puzzle analyses (inference rounds; dynamic puzzles count >1):\n"
            f"  Classification: {s['classification_analyses']}\n"
            f"  Segmentation:   {s['segmentation_analyses']}\n"
            f"  Dynamic:        {s['dynamic_analyses']}\n"
            f"  Total:          {s['classification_analyses'] + s['segmentation_analyses']}\n"
            f"\n"
            f"Outcomes:\n"
            f"  Solved:  {s['successes']}\n"
            f"  Reloads: {s['reloads']}\n"
            f"\n"
            f"Classification backend calls (per cell):\n"
            f"  YOLO:  {s['yolo_cell_calls']}\n"
            f"  VLM:   {s['vlm_cell_calls']}\n"
            f"  Total: {total_cells}\n"
            f"  VLM%:  {vlm_rate:.1f}%\n"
            f"\n"
            f"Segmentation backend calls (per grid):\n"
            f"  YOLO:  {s['yolo_grid_calls']}\n"
            f"  VLM:   {s['vlm_grid_calls']}\n"
            f"\n"
            f"-----------------------------------------------------------------\n"
            f"Prior sessions on file: {len(self.sessions) - 1}  (full history in {self.STATE_FILE})\n"
        )
        try:
            self.path.write_text(text)
        except Exception:
            pass
        self._save_state()


metrics = Metrics()
