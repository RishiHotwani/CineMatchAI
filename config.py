from dotenv import load_dotenv
import os

load_dotenv()   # Load variables from .env
# ============================================================
#  CineMatch AI — Configuration  (OMDb-only edition)
# ============================================================

# ── OMDb API ─────────────────────────────────────────────────
OMDB_API_KEY  = os.getenv("OMDB_API_KEY")
OMDB_BASE_URL = "http://www.omdbapi.com/"

# ── MovieLens 100K ────────────────────────────────────────────
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

# ── Dataset building ──────────────────────────────────────────
# How many movies to pull from OMDb for content-based filtering.
# We search OMDb using a curated seed list of popular titles.
# Each title = 1 API call.  Free tier = 1000 req/day.
OMDB_SEED_MOVIES = [
    # Action / Adventure
    "The Dark Knight", "Inception", "Interstellar", "The Matrix",
    "Avengers Endgame", "Iron Man", "Thor Ragnarok", "Black Panther",
    "Guardians of the Galaxy", "Captain America Civil War",
    "John Wick", "Mad Max Fury Road", "Mission Impossible Fallout",
    "Top Gun Maverick", "Die Hard", "Speed", "The Fugitive",
    "Face Off", "Con Air", "The Rock",

    # Sci-Fi
    "Blade Runner 2049", "Arrival", "Ex Machina", "Her",
    "The Martian", "Gravity", "Avatar", "Edge of Tomorrow",
    "District 9", "Children of Men", "2001 A Space Odyssey",
    "Alien", "Aliens", "Terminator 2", "RoboCop",

    # Drama / Oscar
    "The Shawshank Redemption", "Forrest Gump", "Schindler's List",
    "The Godfather", "The Godfather Part II", "Goodfellas",
    "Pulp Fiction", "Fight Club", "American Beauty",
    "A Beautiful Mind", "The Silence of the Lambs",
    "No Country for Old Men", "There Will Be Blood",
    "12 Angry Men", "Casablanca", "Citizen Kane",
    "Parasite", "Whiplash", "La La Land", "Moonlight",

    # Comedy
    "The Grand Budapest Hotel", "Superbad", "The Hangover",
    "Bridesmaids", "Anchorman", "Step Brothers", "Tropic Thunder",
    "Game Night", "Knives Out", "Clue",

    # Thriller / Mystery
    "Gone Girl", "Se7en", "Zodiac", "Prisoners", "Memento",
    "The Usual Suspects", "Heat", "Chinatown", "Rear Window",
    "Vertigo", "Psycho", "The Sixth Sense", "Shutter Island",
    "Black Swan", "Mulholland Drive",

    # Horror
    "Get Out", "Hereditary", "A Quiet Place", "It Chapter One",
    "The Conjuring", "Midsommar", "Us", "The Witch",
    "Sinister", "Insidious", "The Shining", "Rosemary's Baby",

    # Romance / Drama
    "Titanic", "The Notebook", "Before Sunrise", "Before Sunset",
    "Eternal Sunshine of the Spotless Mind", "500 Days of Summer",
    "Pride and Prejudice", "Brokeback Mountain", "Carol",

    # Animation
    "Toy Story", "The Lion King", "Spirited Away", "Up",
    "WALL-E", "Finding Nemo", "The Incredibles", "Coco",
    "Spider-Man Into the Spider-Verse", "Howl's Moving Castle",
    "Princess Mononoke", "Akira", "Your Name",

    # Crime / Gangster
    "The Departed", "Heat", "Scarface", "Carlito's Way",
    "City of God", "Oldboy", "Memories of Murder",
    "L.A. Confidential", "Fargo", "Reservoir Dogs",

    # War / History
    "Saving Private Ryan", "Hacksaw Ridge", "Dunkirk",
    "Apocalypse Now", "Full Metal Jacket", "Platoon",
    "The Hurt Locker", "1917", "Braveheart", "Gladiator",

    # Fantasy / Adventure
    "The Lord of the Rings The Fellowship of the Ring",
    "The Lord of the Rings The Two Towers",
    "The Lord of the Rings The Return of the King",
    "Harry Potter and the Sorcerer's Stone",
    "The Princess Bride", "Labyrinth", "The NeverEnding Story",
    "Willow", "Stardust", "Pan's Labyrinth",

    # Classics
    "Lawrence of Arabia", "Sunset Boulevard", "Some Like It Hot",
    "Dr. Strangelove", "A Clockwork Orange", "Barry Lyndon",
    "The Shining", "Full Metal Jacket", "Eyes Wide Shut",

    # Korean / World Cinema
    "Train to Busan", "Snowpiercer", "The Handmaiden",
    "A Tale of Two Sisters", "Burning", "The Wailing",
    "Crouching Tiger Hidden Dragon", "Hero", "House of Flying Daggers",
    "In the Mood for Love", "Chungking Express", "Infernal Affairs",

    # Bollywood / Indian
    "3 Idiots", "Dangal", "Lagaan", "Dil Chahta Hai",
    "Swades", "Taare Zameen Par", "Rang De Basanti",
    "Mughal-E-Azam", "Sholay", "Dilwale Dulhania Le Jayenge",
    "Deewar", "Mother India", "Guide",
]

# ── SVD Hyperparameters ───────────────────────────────────────
SVD_N_FACTORS = 100
SVD_N_EPOCHS  = 20

# ── Hybrid weights ────────────────────────────────────────────
HYBRID_CONTENT_WEIGHT = 0.4
HYBRID_COLLAB_WEIGHT  = 0.6

# ── UI defaults ───────────────────────────────────────────────
TOP_N         = 10
CACHE_DIR     = ".cache"
