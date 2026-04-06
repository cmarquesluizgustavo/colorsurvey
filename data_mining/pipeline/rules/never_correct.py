"""Words that should NEVER be fuzzy-corrected — they are valid as-is."""

NEVER_CORRECT = {
    # Valid color words that look like typos of other words
    "blush", "violent", "manila", "saturated", "desaturated",
    # Valid -y adjective modifiers
    "creamy", "cloudy", "fleshy", "steely", "earthy", "stormy",
    "grassy", "watery", "swampy", "mossy", "leafy", "minty",
    "smokey", "dusty", "rusty", "muddy", "sandy", "milky",
    "chalky", "silky", "peachy", "rosy", "inky", "smoky",
    "foresty",
    # Other valid words
    "burple", "barney", "brass", "bronze", "leather", "custard",
    "caramel", "parrot", "creme", "slime", "hazel", "irish",
    "vanilla", "plumb",
    # Colors that look like typos of other words
    "goldenrod", "buttercup", "candy", "amber", "orangered",
    "night", "russet", "spruce", "fresh", "bubblegum", "limey",
    "sickly", "gross",
}
