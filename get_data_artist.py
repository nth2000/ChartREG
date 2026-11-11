import json
import pickle
import os
import matplotlib.pyplot as plt
from matplotlib.artist import Artist
import numpy as np
import sys

# Absolute path of this module.
current_file_path = os.path.abspath(__file__)


# TODO: Hide axes and spines on the relevant subplot when required.
# ax.axis('off')
def get_data_related_artists(all_artist):
    """Return data-related artists, excluding axes and layout objects."""
    filtered_artists = set()
    for artist in all_artist:
        # Exclude axes, spines, background rectangles, and layout elements.
        if not ('spine' in str(artist).lower() or 'axis' in str(artist).lower() or (
                'rectangle' in str(artist) and 'xy=(0, 0)' in str(artist)) or 'axes' in str(artist).lower()):
            filtered_artists.add(artist)
    return filtered_artists


def get_artist_and_properties(fig):
    # Snapshot artists and their properties before the call.
    artists = {artist: artist.properties() for artist in fig.findobj(Artist) if artist.figure is not None}
    return artists


def get_artist_and_visible(fig):
    # Snapshot artist visibility before the call.
    artists = {}
    artist_all = fig.findobj(Artist)
    for artist in artist_all:
        if hasattr(artist, 'get_visible'):
            artists[artist] = artist.get_visible()
        else:
            artists[artist] = None
    return artists


def get_artist_changes(before_artists, after_artists):
    modified_artists = []
    for artist in before_artists:
        if artist in after_artists and (
                before_artists[artist] == False and after_artists[artist] == True):
            modified_artists.append(artist)

    # Find newly added artists.
    added_artists = [artist for artist in after_artists if artist not in before_artists]

    return modified_artists, added_artists


at_least_set_one = False


def set_artists_visible_false(return_value):
    """
    Set set_visible(False) for all artists in the given return value from a matplotlib API call.

    Parameters:
    return_value : Any
        The return value from a matplotlib API call. This could be an Artist, a list/tuple/dict of Artists,
        a Collection, or None.
    """
    global at_least_set_one
    if return_value is None:
        return

    # Handle single Artist
    if isinstance(return_value, Artist):
        return_value.set_visible(False)
        at_least_set_one = True
        return

    # Handle sequences (list, tuple)
    if isinstance(return_value, (list, tuple)):
        for item in return_value:
            set_artists_visible_false(item)  # Recursively handle each item
        return

    # Handle dictionaries (check values)
    if isinstance(return_value, dict):
        for value in return_value.values():
            set_artists_visible_false(value)  # Recursively handle each value
        return

    # Handle collections (e.g., PathCollection, LineCollection)
    if hasattr(return_value, '__iter__') and not isinstance(return_value, (str, bytes)):
        try:
            for item in return_value:
                set_artists_visible_false(item)  # Recursively handle each item
            return
        except TypeError:
            pass  # Not iterable in the way we expect

    # If none of the above, assume it's not an Artist or container of Artists

    if hasattr(return_value, 'ax'):  # colorbar
        set_artists_visible_false(return_value.ax)

    return