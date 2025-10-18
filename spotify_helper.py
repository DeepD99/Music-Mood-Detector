# spotify_helper.py
# Playlist-search only (no deprecated /recommendations or /available-genre-seeds).
# Strategy:
#   1) Search mood playlists by name (strict + loose; with market, then without)
#   2) If needed, try multiple offsets inside the playlist to avoid empty pages
#   3) If still empty, search by genre keywords (normalized)
#   4) Special broader fallback for "wind-down"/"sleep"
# Returns: {"tracks": [...], "source": "playlist-search", "playlist": "<name or None>"}

import os
import random
from typing import List, Dict, Tuple, Optional

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.exceptions import SpotifyException

# ── Credentials ──────────────────────────────────────────────────────────────
SPOTIFY_CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID", "YOUR_CLIENT_ID_HERE")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET", "YOUR_CLIENT_SECRET_HERE")

sp = spotipy.Spotify(
    auth_manager=SpotifyClientCredentials(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
    )
)

# ── Normalize your internal genres into search keywords ──────────────────────
GENRE_NORMALIZER = {
    "lofi": "study",
    "lo-fi": "study",
    "rnb": "r-n-b",
    "r&b": "r-n-b",
    "alt": "indie",
    "alt-rock": "rock",
    "synthwave": "synth-pop",
    "cinematic": "movies",
    "chillhop": "chill",
    # pass-throughs
    "electronic": "electronic",
    "house": "house",
    "techno": "techno",
    "dance": "dance",
    "pop": "pop",
    "hip-hop": "hip-hop",
    "edm": "edm",
    "jazz": "jazz",
    "classical": "classical",
    "chill": "chill",
    "acoustic": "acoustic",
    "soul": "soul",
    "indie": "indie",
    "indie-pop": "indie-pop",
    "synth-pop": "synth-pop",
    "study": "study",
    "sleep": "sleep",
    "movies": "movies",
    "rock": "rock",
    "ambient": "ambient",
}

def _normalize_seeds(seed_genres: List[str]) -> List[str]:
    out = []
    for g in (seed_genres or []):
        k = GENRE_NORMALIZER.get(g.lower().strip(), g.lower().strip())
        if k and k not in out:
            out.append(k)
    if not out:
        out = ["chill"]
    return out[:3]

# ── Mood → playlist search queries (expanded wind-down) ──────────────────────
MOOD_PLAYLIST_QUERIES: Dict[str, List[str]] = {
    "focus":        ["Deep Focus", "Focus Flow", "Instrumental Study", "Lo-Fi Beats"],
    "relaxed":      ["Chill Hits", "Chill Vibes", "Relax & Unwind"],
    "wind-down":    ["Sleep", "Calm Vibes", "Peaceful Piano", "Deep Sleep", "Calm Piano",
                     "Ambient Chill", "Lo-Fi Sleep", "Rest & Unwind", "Sleep Sounds"],
    "sleep":        ["Sleep", "Deep Sleep", "Peaceful Piano", "Sleep Sounds"],
    "workout":      ["Beast Mode", "Power Workout", "Cardio"],
    "hype":         ["Dance Hits", "Hype", "Party"],
    "social":       ["Dance Hits", "Pop Party", "All Out 2010s"],
    "morning":      ["Morning Acoustic", "Wake Up Happy", "Morning Coffee"],
    "drive":        ["Night Rider", "Driving Rock", "Drive"],
    "rainy day":    ["Rainy Day", "Cozy Acoustic", "Lush Lofi"],
    "romantic":     ["Love Pop", "Romance", "R&B Love"],
    "creative flow":["Jazz Vibes", "Lo-Fi Beats", "Brain Food"],
    "melancholy":   ["Life Sucks", "Deep Dark Indie", "Sad Songs"],
    "inspired":     ["Motivation Mix", "Feelin' Good", "Confidence Boost"],
    "confident":    ["Motivation Mix", "Confidence Boost", "All Out Pop"],
    "ambiguous":    ["Chill Hits", "Feelin' Good", "Pop Right Now"],
}

# ── Simple in-memory cache for playlist id lookups ───────────────────────────
_playlist_cache: Dict[str, Tuple[str, str]] = {}  # key -> (playlist_id, playlist_name)

# ── Helpers: robust playlist search & track fetch ────────────────────────────
def _search_playlist_id_by_name(query: str, market: Optional[str]) -> Optional[Tuple[str, str]]:
    """
    Try strict fielded search first (playlist:"Name"), then loose ("Name").
    If nothing is found, retry WITHOUT market (some regions restrict results).
    """
    key = f"{(market or '').lower()}::strict::{query.lower()}"
    if key in _playlist_cache:
        return _playlist_cache[key]

    def _try(q: str, mkt: Optional[str]):
        try:
            res = sp.search(q=q, type="playlist", limit=1, market=mkt)
            items = res.get("playlists", {}).get("items", [])
            if items:
                pid = items[0]["id"]
                pname = items[0].get("name", query)
                return (pid, pname)
        except Exception:
            return None
        return None

    # 1) strict w/ market
    pid_name = _try(f'playlist:"{query}"', market)
    # 2) loose w/ market
    if not pid_name:
        pid_name = _try(query, market)
    # 3) strict no market
    if not pid_name and market:
        pid_name = _try(f'playlist:"{query}"', None)
    # 4) loose no market
    if not pid_name and market:
        pid_name = _try(query, None)

    if pid_name:
        _playlist_cache[key] = pid_name
        return pid_name
    return None

def _tracks_from_playlist_id_with_offsets(pid: str, market: Optional[str], limit: int) -> List[Dict]:
    """
    Try a few offsets to avoid empty/region-limited early pages, then shuffle.
    """
    for _ in range(3):
        try:
            offset = random.choice([0, 25, 50])
            res = sp.playlist_items(pid, market=market, additional_types=("track",), limit=100, offset=offset)
            items = res.get("items", [])
            if not items:
                continue
            random.shuffle(items)
            tracks: List[Dict] = []
            for it in items:
                t = it.get("track")
                if not t:
                    continue
                tracks.append({
                    "name": t["name"],
                    "artist": ", ".join(a["name"] for a in t.get("artists", [])),
                    "url": t["external_urls"]["spotify"],
                })
                if len(tracks) >= limit:
                    return tracks
        except SpotifyException:
            continue
        except Exception:
            continue
    return []

def _search_playlist_and_get_tracks(queries: List[str], market: Optional[str], limit: int) -> Tuple[List[Dict], Optional[str]]:
    """
    Try each candidate name; return (tracks, playlist_name) for the first
    playlist that yields tracks (with random offsets).
    """
    for q in (queries or []):
        pid_name = _search_playlist_id_by_name(q, market)
        if not pid_name:
            continue
        pid, pname = pid_name
        tracks = _tracks_from_playlist_id_with_offsets(pid, market, limit)
        if tracks:
            return tracks, pname
    return [], None

def _search_genre_playlists(seeds: List[str], market: Optional[str], limit: int) -> Tuple[List[Dict], Optional[str]]:
    """
    Search by genre keywords (normalized seeds). First that returns tracks wins.
    """
    for g in seeds:
        pid_name = _search_playlist_id_by_name(g, market)
        if not pid_name:
            continue
        pid, pname = pid_name
        tracks = _tracks_from_playlist_id_with_offsets(pid, market, limit)
        if tracks:
            return tracks, pname
    # Absolute last resort: generic chill
    pid_name = _search_playlist_id_by_name("Chill", market)
    if pid_name:
        pid, pname = pid_name
        tracks = _tracks_from_playlist_id_with_offsets(pid, market, limit)
        if tracks:
            return tracks, pname
    return [], None

# ── Public entrypoint ────────────────────────────────────────────────────────
def get_spotify_recommendations(
    seed_genres: List[str],
    features: Dict,                    # kept for signature compatibility; unused with search
    limit: int = 10,
    market: Optional[str] = "US",
    mood_for_fallback: str = "ambiguous",
) -> Dict:
    """
    Playlist-search-only implementation (no deprecated endpoints).
    1) Try curated mood playlist names (strict/loose, with/without market, offsets).
    2) If none, search by normalized genre keywords.
    3) Special broader fallback for 'wind-down'/'sleep'.
    Returns: {"tracks": [...], "source": "playlist-search", "playlist": "<name or None>"}
    """
    seeds = _normalize_seeds(seed_genres)

    # 1) Mood playlist search
    queries = MOOD_PLAYLIST_QUERIES.get(mood_for_fallback, []) or MOOD_PLAYLIST_QUERIES["ambiguous"]
    tracks, plist_name = _search_playlist_and_get_tracks(queries, market, limit)
    if tracks:
        return {"tracks": tracks, "source": "playlist-search", "playlist": plist_name}

    # 2) Genre playlist search
    tracks, plist_name = _search_genre_playlists(seeds, market, limit)
    if tracks:
        return {"tracks": tracks, "source": "playlist-search", "playlist": plist_name}

    # 3) Special broader fallback for "wind-down"/"sleep"
    if mood_for_fallback in ("wind-down", "sleep"):
        broad_terms = ["sleep", "calm", "peaceful", "lofi sleep", "ambient", "rest"]
        tracks, plist_name = _search_playlist_and_get_tracks(broad_terms, market, limit)
        if tracks:
            return {"tracks": tracks, "source": "playlist-search", "playlist": plist_name}

    # Last resort: empty but consistent payload
    return {"tracks": [], "source": "playlist-search", "playlist": None}
