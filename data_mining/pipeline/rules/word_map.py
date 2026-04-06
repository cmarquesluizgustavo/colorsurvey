"""Deterministic word-level replacements (typos, British → American, etc.)."""

DETERMINISTIC_WORD_MAP: dict[str, str] = {
    # British → American
    "grey": "gray",
    "greyish": "grayish",
    "greyed": "grayed",
    "greyey": "grayish",
    "greay": "gray",
    "gery": "gray",
    "greyn": "green",  # typo, not gray

    # Green typos
    "gree": "green",
    "gren": "green",
    "greem": "green",
    "grenn": "green",
    "geen": "green",
    "greeb": "green",
    "grene": "green",
    "groen": "green",
    "grean": "green",
    "grren": "green",
    "geren": "green",
    "gween": "green",
    "greeeeen": "green",
    "greeeen": "green",
    "greeeeeeen": "green",
    "greeeeeen": "green",
    "greeeeeeeen": "green",

    # Purple typos
    "pruple": "purple",
    "pueple": "purple",
    "purpel": "purple",
    "purplr": "purple",
    "purble": "purple",
    "puprle": "purple",
    "poiple": "purple",
    "purlpe": "purple",
    "purplw": "purple",
    "putple": "purple",
    "pirple": "purple",
    "murple": "purple",
    "purpur": "purple",
    "purpke": "purple",
    "purpil": "purple",
    "purpul": "purple",
    "grurple": "purple",
    "porple": "purple",
    "ourple": "purple",
    "purkle": "purple",

    # Chartreuse typos
    "charteuse": "chartreuse",
    "chartruese": "chartreuse",
    "chartruce": "chartreuse",
    "chatreuse": "chartreuse",
    "chartreus": "chartreuse",
    "chatruse": "chartreuse",
    "shartruse": "chartreuse",

    # Fuchsia typos
    "fuschia": "fuchsia",
    "fucia": "fuchsia",
    "fusia": "fuchsia",
    "fusha": "fuchsia",
    "fuchisa": "fuchsia",
    "fucha": "fuchsia",

    # Mauve typos
    "muave": "mauve",
    "mouve": "mauve",
    "moave": "mauve",
    "mauv": "mauve",
    "maube": "mauve",
    "mauce": "mauve",
    "malve": "mauve",

    # Raspberry typos
    "rasberry": "raspberry",

    # Terracotta typos
    "terracota": "terracotta",
    "teracotta": "terracotta",

    # Khaki typos
    "kaki": "khaki",
    "kahki": "khaki",
    "kakhi": "khaki",
    "karki": "khaki",
    "kacki": "khaki",

    # Taupe typos
    "tope": "taupe",
    "toupe": "taupe",
    "taup": "taupe",
    "toap": "taupe",
    "toape": "taupe",

    # Ochre typos
    "ocre": "ochre",
    "ocker": "ochre",
    "ocra": "ochre",
    "oker": "ochre",
    "ochra": "ochre",
    "ockre": "ochre",

    # Lilac typos
    "lila": "lilac",
    "lilla": "lilac",
    "lylac": "lilac",
    "lilas": "lilac",
    "lilic": "lilac",
    "lilca": "lilac",
    "lylic": "lilac",
    "lilav": "lilac",

    # Seafoam typos
    "seafom": "seafoam",
    "seaform": "seafoam",

    # Blue typos
    "bluue": "blue",
    "bluw": "blue",

    # Avocado typos
    "avacado": "avocado",

    # Beige typos
    "beig": "beige",
    "baige": "beige",
    "biege": "beige",

    # Fuchsia typos (additional)
    "fuscia": "fuchsia",
    "fushia": "fuchsia",
    "fucshia": "fuchsia",
    "fuscha": "fuchsia",

    # Violet typos
    "violot": "violet",
    "voilet": "violet",
    "viloet": "violet",

    # Brown typos
    "brow": "brown",

    # Burgundy typos
    "burgendy": "burgundy",

    # Cream typos
    "creme": "cream",

    # Lavender typos
    "lavendar": "lavender",

    # Magenta typos
    "magneta": "magenta",
    "majenta": "magenta",

    # Puce typos
    "peuce": "puce",
    "puse": "puce",

    # Reddish typos
    "redish": "reddish",

    # Turquoise typos (additional)
    "terquoise": "turquoise",
    "torquise": "turquoise",
    "turqois": "turquoise",
    "turqiose": "turquoise",
    "tourquise": "turquoise",
    "turcoise": "turquoise",
    "tourqoise": "turquoise",
    "turqouis": "turquoise",
    "turqoiuse": "turquoise",
    "turquiose": "turquoise",
    "turqouse": "turquoise",
    "turquase": "turquoise",
    "turquese": "turquoise",
    "turquios": "turquoise",
    "turqiouse": "turquoise",
    "torquiose": "turquoise",
    "torqouise": "turquoise",
    "torqoise": "turquoise",
    "tirquoise": "turquoise",
    "turquis": "turquoise",
    "turqouise": "turquoise",

    # Blue typos (additional)
    "bllue": "blue",
    "blule": "blue",

    # Pink typos
    "pikn": "pink",

    # Orange typos
    "ornage": "orange",
    "oragne": "orange",
    "organe": "orange",
    "oraneg": "orange",
    "orance": "orange",
    "orenge": "orange",
    "oarnge": "orange",

    # Brown typos (additional)
    "bronw": "brown",
    "brwon": "brown",
    "borwn": "brown",
    "bown": "brown",
    "broen": "brown",
    "brwn": "brown",

    # Yellow typos
    "yelloe": "yellow",
    "yellpw": "yellow",
    "yeloow": "yellow",

    # Beige typos (additional)
    "bage": "beige",
    "bege": "beige",
    "beigh": "beige",
    "baise": "beige",
    "bez": "beige",

    # Fuchsia typos (more)
    "fusca": "fuchsia",

    # Cyan typos
    "cyab": "cyan",
    "cya": "cyan",

    # Aqua typos
    "aque": "aqua",
    "aqau": "aqua",
    "awua": "aqua",
    "auqa": "aqua",
    "auqua": "aqua",
    "auqamarine": "aquamarine",

    # Comparative modifiers → base (standalone they get stripped anyway)
    "greener": "green",
    "bluer": "blue",

    # Short fragments commonly mismatched by fuzzy
    "urple": "purple",
    "purpl": "purple",
    "marroon": "maroon",

    # Alternate spellings / additional typos
    "ocher": "ochre",
    "marron": "maroon",
    "blooo": "blue",
    "blud": "blue",
    "pinlk": "pink",
    "pinl": "pink",
    "turkiz": "turquoise",
    "terqoise": "turquoise",
    "forset": "forest",
    "oliv": "olive",
    "ligt": "light",
    "lght": "light",
    "bule": "blue",
    "bleu": "blue",
    "blu": "blue",
    "bue": "blue",
    "violte": "violet",
    "saumon": "salmon",
    "lavenda": "lavender",

    # Blue typos (more)
    "bloue": "blue",
    "bluje": "blue",
    "bluwe": "blue",

    # Magenta typos (more)
    "maginta": "magenta",

    # Cantaloupe / cerise / sherbet
    "cantelope": "cantaloupe",
    "cerice": "cerise",
    "sherbert": "sherbet",

    # Turquoise typos (more)
    "tourquiose": "turquoise",

    # Avocado typos (more)
    "avacodo": "avocado",

    # Pale / navy typos
    "plae": "pale",
    "navi": "navy",

    # Beige typos (more)
    "bauge": "beige",

    # Maroon alternate
    "marrone": "maroon",

    # Misc typos
    "breen": "green",
    "pinkle": "pink",
    "purblue": "purple",
    "limish": "lime",
    "birght": "bright",
    "lav": "lavender",
    "sku": "sky",

    # Onboarded from fuzzy review
    "florescent": "fluorescent",
    "chartruse": "chartreuse",
    "vermillion": "vermilion",
    "turquose": "turquoise",
    "orangeish": "orangish",
    "tuquoise": "turquoise",
    "violett": "violet",
    "orangy": "orangey",
    "purpler": "purple",
    "magent": "magenta",
    "forrest": "forest",
    "colour": "color",
    "flourescent": "fluorescent",
    "robins": "robin",
    "purpe": "purple",
    "purle": "purple",
    "lillac": "lilac",
    "fluro": "fluoro",
    "greeen": "green",
    "kelley": "kelly",
    "liliac": "lilac",
    "orang": "orange",
    "siena": "sienna",
    "yello": "yellow",
    "puple": "purple",
    "yucky": "yuck",
    "brighter": "bright",
}
