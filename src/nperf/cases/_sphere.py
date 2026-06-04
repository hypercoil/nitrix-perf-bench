# -*- coding: utf-8 -*-
"""Shared helpers for the spherical-geometry family.

``nitrix.geometry.{latlong_to_cartesian, cartesian_to_latlong}`` are the
spherical<->Cartesian coordinate conversions (latitude measured from the
equator; longitude around it). They are pure elementwise trig with an
unambiguous closed form, so the reference IS that closed form (a numpy
reimplementation + fp64 oracle) -- no external library is more canonical than
the formula, and bridging one (astropy/healpy colatitude conventions) would
only add match-the-right-target risk. A CuPy GPU ref runs the same trig.

Verified in fp64: ``latlong_to_cartesian`` matches the closed form to 0.0 and
the round-trip ``cartesian_to_latlong(latlong_to_cartesian(.))`` recovers the
angles to ~1e-16.

(This module will grow with the geodesic-distance / spherical-conv ops.)
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np


def latlong_input(n: int, seed: int = 0) -> np.ndarray:
    '''Random ``(latitude, longitude)`` in radians, ``(n, 2)``; latitude in
    ``[-pi/2, pi/2]``, longitude in ``[-pi, pi]``.'''
    rng = np.random.default_rng(seed)
    lat = rng.uniform(-np.pi / 2, np.pi / 2, n).astype(np.float32)
    lon = rng.uniform(-np.pi, np.pi, n).astype(np.float32)
    return np.stack([lat, lon], axis=-1)


def xyz_input(n: int, seed: int = 0) -> np.ndarray:
    '''Random unit-sphere Cartesian points, ``(n, 3)``.'''
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, 3)).astype(np.float32)
    return (v / np.linalg.norm(v, axis=-1, keepdims=True)).astype(np.float32)


def _l2c(ll: Any, xp: Any, r: float) -> Any:
    lat = ll[..., 0]
    lon = ll[..., 1]
    cos_lat = xp.cos(lat)
    return xp.stack(
        (r * cos_lat * xp.cos(lon),
         r * cos_lat * xp.sin(lon),
         r * xp.sin(lat)),
        axis=-1,
    )


def _c2ll(xyz: Any, xp: Any) -> Any:
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]
    lat = xp.arctan2(z, xp.sqrt(x ** 2 + y ** 2))
    lon = xp.arctan2(y, x)
    return xp.stack((lat, lon), axis=-1)


def np_latlong_to_cartesian(ll: Any, r: float = 1.0) -> np.ndarray:
    '''numpy closed-form lat/long -> Cartesian (floor + fp64 oracle).'''
    return _l2c(np.asarray(ll), np, r)


def np_cartesian_to_latlong(xyz: Any) -> np.ndarray:
    '''numpy closed-form Cartesian -> lat/long (floor + fp64 oracle).'''
    return _c2ll(np.asarray(xyz), np)


def cupy_latlong_to_cartesian(r: float = 1.0) -> Callable[[Any], Any]:
    '''GPU lat/long -> Cartesian (same trig); cupy lazy.'''

    def run(ll: Any) -> Any:
        import cupy as cp

        return _l2c(ll, cp, r)

    return run


def cupy_cartesian_to_latlong() -> Callable[[Any], Any]:
    '''GPU Cartesian -> lat/long (same trig); cupy lazy.'''

    def run(xyz: Any) -> Any:
        import cupy as cp

        return _c2ll(xyz, cp)

    return run


# ---- spherical geodesic distance -----------------------------------------


def _geodesic(x: Any, xp: Any, r: float = 1.0) -> Any:
    '''All-pairs great-circle distance within ``x`` (``(n, 3)``) via the
    robust ``r·atan2(|X×Y|, X·Y)`` formula (nitrix's convention). Returns
    ``(n, n)``.'''
    xb = x[..., :, None, :]
    yb = x[..., None, :, :]
    cross = xp.cross(xb, yb, axis=-1)
    num = xp.sqrt((cross ** 2).sum(-1))
    den = (xb * yb).sum(-1)
    return r * xp.arctan2(num, den)


def np_geodesic(x: Any, r: float = 1.0) -> np.ndarray:
    '''numpy all-pairs geodesic (fp64 oracle).'''
    return _geodesic(np.asarray(x), np, r)


def sklearn_haversine() -> Callable[[Any], Any]:
    '''``sklearn.metrics.pairwise.haversine_distances`` on the lat/long of the
    points -- the canonical domain-tool great-circle distance (for the unit
    sphere the angular distance equals the geodesic; verified ~2e-15 in fp64).
    sklearn imported lazily (only the numpy worker).'''

    def run(x: Any) -> Any:
        from sklearn.metrics.pairwise import haversine_distances

        ll = np_cartesian_to_latlong(x)
        return haversine_distances(ll, ll)

    return run


def cupy_geodesic(r: float = 1.0) -> Callable[[Any], Any]:
    '''GPU all-pairs geodesic (same atan2 formula); cupy lazy.'''

    def run(x: Any) -> Any:
        import cupy as cp

        return _geodesic(x, cp, r)

    return run
