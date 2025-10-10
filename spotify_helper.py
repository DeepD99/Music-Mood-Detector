import os
import random
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.exceptions import SpotifyException

# Set your credentials as environment variables (recommended)
SPOTIFY_CLIENT_ID = "314ff7c4615f4f36bf152bd13c3870da"
SPOTIFY_CLIENT_SECRET = "8c26158bec9e4759a3cae0156e5a3769"

sp = spotipy.Spotify(
    auth_manager=SpotifyClientCredentials(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET
    )
)

# Keep a stable subset of valid seeds (we won't query Spotify for these)
KNOWN_SEEDS = {
    "acoustic","ambient","chill","classical","dance","edm","electronic","hip-hop",
    "house","indie","indie-pop","jazz","movies","pop","r-n-b","rock","sleep",
    "soul","study","synth-pop","techno"
}

GENRE_NORMALIZER = {
    "lofi": "study", "lo-fi": "study",
    "rnb": "r-n-b", "r&b": "r-n-b",
    "alt": "indie", "alt-rock": "rock",
    "synthwave": "synth-pop",
    "cinematic": "movies",
    "chillhop": "chill",
    # pass-throughs
    "electronic":"electronic","house":"house","techno":"techno","dance":"dance",
    "pop":"pop","hip-hop":"hip-hop","edm":"edm","jazz":"jazz","classical":"classical",
    "chill":"chill","acoustic":"acoustic","soul":"soul","indie":"indie",
    "indie-pop":"indie-pop","synth-pop":"synth-pop","study":"study","sleep":"sleep",
    "movies":"movies","rock":"rock",
}

# Preferred playlist names to search per mood (we will NOT hardcode IDs)
MOOD_PLAYLIST_QUERIES = {
    "focus":        ["Deep Focus", "Lo-Fi Beats", "Focus Flow", "Instrumental Study"],
    "relaxed":      ["Chill Hits", "Chill Vibes", "Relax & Unwind"],
    "wind-down":    ["Sleep", "Calm Vibes", "Peaceful Piano"],
    "sleep":        ["Sleep", "Deep Sleep", "Peaceful Piano"],
    "workout":      ["Beast Mode", "Power Workout", "Cardio"],
    "hype":         ["Dance Hits", "Hype", "Party"],
    "social":       ["Dance Hits", "All Out 00s/10s", "Pop Party"],
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

def _normalize_and_filter_seeds(seed_genres):
    out = []
    for g in (seed_genres or []):
        g = GENRE_NORMALIZER.get(g.lower().strip(), g.lower().strip())
        if g in KNOWN_SEEDS:
            out.append(g)
    if not out:
        out = ["pop", "indie"]
    return out[:5]

def _tracks_from_playlist_id(playlist_id, market="US", limit=10):
    items = sp.playlist_items(playlist_id, market=market, additional_types=("track",)).get("items", [])
    random.shuffle(items)
    tracks = []
    for it in items:
        t = it.get("track")
        if not t:
            continue
        tracks.append({
            "name": t["name"],
            "artist": ", ".join(a["name"] for a in t["artists"]),
            "url": t["external_urls"]["spotify"]
        })
        if len(tracks) >= limit:
            break
    return tracks

def _search_playlist_and_get_tracks(queries, market="US", limit=10):
    """Search by a list of names; return first playlist's tracks that works."""
    for q in queries:
        try:
            res = sp.search(q=f'playlist:"{q}"', type="playlist", limit=1, market=market)
            items = res.get("playlists", {}).get("items", [])
            if not items:
                continue
            pid = items[0]["id"]
            tracks = _tracks_from_playlist_id(pid, market=market, limit=limit)
            if tracks:
                return tracks
        except SpotifyException:
            continue
        except Exception:
            continue
    return []

def _genre_playlist_fallback(seeds, market="US", limit=10):
    """As a last resort, search playlists by genre keywords."""
    for g in seeds:
        tracks = _search_playlist_and_get_tracks([g], market=market, limit=limit)
        if tracks:
            return tracks
    # absolute last resort: generic "Chill" search
    return _search_playlist_and_get_tracks(["Chill"], market=market, limit=limit)

def get_spotify_recommendations(seed_genres, features, limit=10, market="US", mood_for_fallback="ambiguous"):
    seeds = _normalize_and_filter_seeds(seed_genres)
    params = dict(limit=limit, seed_genres=seeds, market=market)

    allowed = {
        "target_tempo","min_tempo","max_tempo",
        "target_energy","min_energy","max_energy",
        "target_valence","min_valence","max_valence",
        "target_danceability","min_danceability","max_danceability",
        "target_instrumentalness","min_instrumentalness","max_instrumentalness",
        "target_acousticness","min_acousticness","max_acousticness"
    }
    for k, v in (features or {}).items():
        if k in allowed:
            params[k] = v

    # 1) Try recommendations
    try:
        recs = sp.recommendations(**params)
        tracks = [{
            "name": t["name"],
            "artist": ", ".join(a["name"] for a in t["artists"]),
            "url": t["external_urls"]["spotify"]
        } for t in recs.get("tracks", [])]
        if tracks:
            return tracks
    except SpotifyException:
        pass
    except Exception:
        pass

    # 2) Mood playlist search fallback
    tracks = _search_playlist_and_get_tracks(MOOD_PLAYLIST_QUERIES.get(mood_for_fallback, []), market=market, limit=limit)
    if tracks:
        return tracks

    # 3) Genre playlist search fallback
    # return _genre_playlist_fallback(seeds, market=market, limit=limit)
    # replace the 3 return points in get_spotify_recommendations with:
    return {"tracks": tracks, "source": "recommendations"}

    # ...
    return {"tracks": _fallback_tracks_for_mood(mood_for_fallback, limit=limit), "source": "mood-playlist"}

    # ...
    return {"tracks": _genre_playlist_fallback(seeds, market=market, limit=limit), "source": "genre-playlist"}
