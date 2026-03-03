import numpy as np
from collections import Counter
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Union
import re


class BaseEmbedder(ABC):
    """Base class for text embedding methods."""

    @abstractmethod
    def fit(self, texts: List[str]) -> "BaseEmbedder":
        """Fit the embedder on a list of texts."""
        pass

    @abstractmethod
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts into embeddings."""
        pass


class BagOfWordsEmbedder(BaseEmbedder):
    """
    Bag of Words embedder for color names.
    Creates vectors based on word counts.
    """

    def __init__(self, top_n_words: int = 100):
        """
        Args:
            top_n_words: Number of most common words to include in vocabulary
        """
        super().__init__()
        self.top_n_words = top_n_words
        self.word_to_idx: Dict[str, int] = {}
        self.vocab_size = 0

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize a color name into words.
        Handles spaces, camelCase, and common separators.

        Examples:
            "pastel blue" -> ["pastel", "blue"]
            "skyBlue" -> ["sky", "blue"]
            "dark-red" -> ["dark", "red"]
        """
        # Convert to lowercase
        text = text.lower()

        # Split camelCase (e.g., "skyBlue" -> "sky blue")
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)

        # Extract all words (alphanumeric sequences)
        words = re.findall(r"\b[a-z]+\b", text)

        return words

    def fit(self, texts: List[str]) -> "BagOfWordsEmbedder":
        """
        Build vocabulary from texts.

        Args:
            texts: List of color names
        """
        # Tokenize all texts and count word frequencies
        word_counter = Counter()
        for text in texts:
            words = self._tokenize(text)
            word_counter.update(words)

        # Select top N most common words
        top_words = [word for word, _ in word_counter.most_common(self.top_n_words)]

        # Build word to index mapping
        self.word_to_idx = {word: idx for idx, word in enumerate(top_words)}
        self.vocab_size = len(self.word_to_idx)

        print(f"Built BoW vocabulary with {self.vocab_size} words")

        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts into BoW vectors.

        Args:
            texts: List of color names

        Returns:
            Array of shape (len(texts), vocab_size) with word counts
        """
        # Create output array
        vectors = np.zeros((len(texts), self.vocab_size), dtype=np.float32)

        # Fill in word counts for each text
        for i, text in enumerate(texts):
            words = self._tokenize(text)
            for word in words:
                if word in self.word_to_idx:
                    idx = self.word_to_idx[word]
                    vectors[i, idx] += 1

        return vectors

    def get_vocabulary(self) -> List[str]:
        """Get the list of words in vocabulary (in index order)."""
        return [
            word for word, _ in sorted(self.word_to_idx.items(), key=lambda x: x[1])
        ]


def rgb_to_oklch(
    colors: Union[np.ndarray, List[Tuple[int, int, int]]], normalize: bool = True
) -> np.ndarray:
    """
    Convert RGB colors to OKLCH color space.
    
    OKLCH is a perceptually uniform color space that better represents
    how humans perceive colors compared to RGB.
    
    Args:
        colors: RGB values (n_samples, 3) in [0, 255]
        normalize: If True, normalize to [0, 1] range
        
    Returns:
        OKLCH values (n_samples, 3):
        - L (Lightness): [0, 1]
        - C (Chroma): [0, ~0.4] or [0, 1] if normalized
        - H (Hue): [0, 360] or [0, 1] if normalized
    """
    colors = np.array(colors) if isinstance(colors, list) else colors
    
    # sRGB to linear RGB
    rgb = colors / 255.0
    rgb_linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    
    # Linear RGB to XYZ (D65)
    M_xyz = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = rgb_linear @ M_xyz.T
    
    # XYZ to OKLab
    M_lms = np.array(
        [
            [0.8189330101, 0.3618667424, -0.1288597137],
            [0.0329845436, 0.9293118715, 0.0361456387],
            [0.0482003018, 0.2643662691, 0.6338517070],
        ]
    )
    M_oklab = np.array(
        [
            [0.2104542553, 0.7936177850, -0.0040720468],
            [1.9779984951, -2.4285922050, 0.4505937099],
            [0.0259040371, 0.7827717662, -0.8086757660],
        ]
    )
    oklab = np.cbrt(xyz @ M_lms.T) @ M_oklab.T
    
    # OKLab to OKLCH
    L, a, b = oklab[..., 0], oklab[..., 1], oklab[..., 2]
    C = np.sqrt(a**2 + b**2)
    H = np.arctan2(b, a) * 180.0 / np.pi
    H = np.where(H < 0, H + 360, H)
    
    oklch = np.stack([L, C, H], axis=-1)
    
    if normalize:
        oklch[..., 1] /= 0.5  # Chroma: [0, ~0.4] -> [0, 1]
        oklch[..., 2] /= 360.0  # Hue: [0, 360] -> [0, 1]
    
    return oklch
