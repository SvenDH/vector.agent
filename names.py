adjectives = [
    "one", "two", "three", "four", "five", "six", "seven", "eight",
    "red", "blue", "green", "golden", "ancient", "brave", "mighty", "shadow",
    "swift", "silent", "bright", "dark", "lone", "wild", "wise", "bold",
    "deep", "last", "lost", "pale", "sage", "true", "false", "dead",
    "void", "calm", "dual", "pure", "raw", "vast", "zero", "ever",
    "new", "old", "free", "high", "low", "near", "far", "open",
    "strong", "weak", "young", "late", "early", "great", "small", "wide",
    "dread", "foul", "grim", "harsh", "keen", "meek", "mild", "rude",
]

nouns = [
    "wolf", "star", "moon", "sun", "wind", "rain", "sage", "bird",
    "dawn", "dusk", "echo", "fate", "fire", "flow", "hawk", "hope",
    "king", "leaf", "light", "mind", "oath", "path", "soul", "void",
    "wave", "wing", "byte", "code", "data", "gear", "pulse", "storm",
    "pebbles", "wind", "knight", "dragon", "forest", "river", "mountain", "blade",
    
    # Natural Features
    "mountain", "river", "ocean", "forest", "desert", "glacier", "volcano", "canyon", "valley",
    "meadow", "prairie", "tundra", "island", "reef", "cave", "spring", "stream", "lake", "sea",
    "cliff", "peak", "summit", "horizon", "shore", "beach", "dune", "oasis", "marsh", "swamp",
    
    # Celestial Objects
    "star", "sun", "moon", "planet", "comet", "asteroid", "meteor", "galaxy", "nebula",
    "constellation", "cosmos", "universe", "void", "aurora", "light", "shadow", "darkness",
    
    # Weather & Elements
    "storm", "thunder", "lightning", "rain", "snow", "hail", "wind", "breeze", "tornado",
    "hurricane", "cyclone", "blizzard", "frost", "ice", "flame", "fire", "spark", "ember",
    
    # Plants
    "tree", "flower", "rose", "lily", "oak", "pine", "cedar", "willow", "vine", "leaf",
    "branch", "root", "seed", "bloom", "blossom", "garden", "grove", "forest", "hedge",
    
    # Creatures - Real
    "wolf", "eagle", "hawk", "owl", "raven", "crow", "swan", "lion", "tiger", "bear",
    "whale", "dolphin", "shark", "serpent", "viper", "cobra", "fox", "deer", "horse",
    
    # Creatures - Mythical
    "dragon", "phoenix", "griffin", "unicorn", "pegasus", "hydra", "sphinx", "kraken",
    "wyrm", "drake", "basilisk", "chimera", "manticore", "leviathan", "behemoth",
    
    # People & Roles
    "knight", "warrior", "mage", "wizard", "witch", "sorcerer", "king", "queen", "prince",
    "princess", "lord", "lady", "sage", "monk", "priest", "hunter", "seeker", "wanderer",
    "nomad", "guardian", "sentinel", "watcher", "keeper", "master", "apprentice",
    
    # Objects - Weapons
    "sword", "blade", "dagger", "spear", "axe", "bow", "arrow", "shield", "armor",
    "hammer", "mace", "staff", "wand", "orb", "scepter", "flail", "scythe",
    
    # Objects - Magical & Valuable
    "crystal", "gem", "jewel", "ruby", "emerald", "sapphire", "diamond", "pearl",
    "crown", "ring", "amulet", "pendant", "bracelet", "talisman", "relic", "artifact",
    
    # Objects - Tools & Items
    "book", "scroll", "tome", "grimoire", "map", "compass", "key", "lock", "chain",
    "mirror", "lens", "prism", "vessel", "chalice", "grail", "cauldron", "bell",
    
    # Places & Structures
    "castle", "tower", "temple", "shrine", "sanctuary", "haven", "refuge", "fortress",
    "palace", "cathedral", "citadel", "keep", "gate", "bridge", "arch", "door", "portal",
    
    # Abstract Concepts
    "dream", "nightmare", "vision", "illusion", "memory", "thought", "mind", "soul",
    "spirit", "heart", "destiny", "fate", "fortune", "chance", "luck", "hope", "wish",
    "prayer", "blessing", "curse", "prophecy", "omen", "sign", "portent", "truth", "lie",
    "secret", "mystery", "riddle", "puzzle", "enigma", "paradox", "wisdom", "knowledge",
    
    # Time & Space
    "moment", "instant", "eternity", "infinity", "beginning", "ending", "dawn", "dusk",
    "twilight", "midnight", "abyss", "void", "nexus", "crossroads", "path", "way", "road",
    "journey", "quest", "adventure", "voyage", "expedition"
]

verbs = [
    "seek", "chase", "dance", "dream", "fall", "flow", "fly", "gaze",
    "grow", "heal", "jump", "know", "make", "move", "ride", "rise",
    "see", "take", "walk", "watch", "become", "look", "speak", "run",
    "chasing", "whisper", "conquer", "wander", "embrace", "forge", "guide", "ignite",
]

prepositions = ["to", "beyond", "through", "under", "over", "before", "among"]
determiners = ["the", "a", "this", "that", "these", "those"]

patterns = [
    ("adjective", "noun"),
    ("verb", "preposition", "determiner", "noun"),
    ("adjective", "adjective", "noun"),
    ("verb", "noun"),
    ("verb", "determiner", "noun"),
    ("adjective", "noun", "noun"),
    ("verb", "adjective", "noun"),
    ("determiner", "adjective", "noun"),
    ("noun", "of", "noun"),
    ("verb", "preposition", "determiner", "noun"),
    ("adjective", "and", "adjective", "noun"),
    ("noun", "preposition", "adjective", "noun")
]
