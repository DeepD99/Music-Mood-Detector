from spotify_helper import get_spotify_recommendations

features = {"target_tempo": 90, "target_energy": 0.45, "target_valence": 0.5}
tracks = get_spotify_recommendations(["jazz","study"], features, limit=5, market="US", mood_for_fallback="focus")
for i, t in enumerate(tracks, 1):
    print(f"{i}. {t['name']} — {t['artist']}\n   {t['url']}")
