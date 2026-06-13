"""Durable storage: a persistent affordance library + session reports under ~/.wmos (inspectable JSON)."""
import os, json


class SessionStore:
    def __init__(self, base="~/.wmos"):
        self.base = os.path.expanduser(base)
        self.sessions = os.path.join(self.base, "sessions")
        os.makedirs(self.sessions, exist_ok=True)
        self._lib = os.path.join(self.base, "library.json")

    def load_library(self):
        try:
            with open(self._lib) as f: return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}

    def save_library(self, lib):
        tmp = self._lib + ".tmp"
        with open(tmp, "w") as f: json.dump(lib, f, indent=2, sort_keys=True)
        os.replace(tmp, self._lib)                              # atomic write (no half-files)

    def save_session(self, session_id, report):
        path = os.path.join(self.sessions, f"{session_id}.json")
        with open(path, "w") as f: json.dump(report, f, indent=2, default=str)
        return path

    def list_sessions(self):
        return sorted(f[:-5] for f in os.listdir(self.sessions) if f.endswith(".json"))

    def load_session(self, session_id):
        with open(os.path.join(self.sessions, f"{session_id}.json")) as f: return json.load(f)
