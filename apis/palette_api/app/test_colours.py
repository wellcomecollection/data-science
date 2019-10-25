import re

import numpy as np
from hypothesis import example, given
from hypothesis import strategies as st

from .colours import hex_to_rgb, random_hex, rgb_to_lab


@given(st.from_regex(r'[A-Fa-f0-9]{6}', fullmatch=True))
def test_hex_to_rgb(hex_code):
    rgb = hex_to_rgb(hex_code)
    assert isinstance(rgb, list)
    assert len(rgb) == 3
    assert all([x < 256 for x in rgb])
    assert all([x >= 0 for x in rgb])


@given(st.lists(st.lists(
    st.integers(min_value=0, max_value=255),
    min_size=3, max_size=3),
    min_size=5, max_size=5)
)
def test_rgb_to_lab(rgb_palette):
    lab = rgb_to_lab(np.array(rgb_palette))
    assert lab.shape == (5, 3)
    assert (lab < 128).all()
    assert (lab >= -127).all()


def test_random_hex():
    assert re.fullmatch(r'[A-Fa-f0-9]{6}', random_hex())
