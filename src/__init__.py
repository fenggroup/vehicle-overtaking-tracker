"""
VehiclePassTracker - A package for detecting and analyzing vehicle overtaking events
using bicycle-mounted IR cameras.
"""

from .tracker import VehiclePassTracker

__version__ = '1.0.0'
__author__ = 'Gandhimathi (Mathi) Padmanaban'

# Export main class
__all__ = ['VehiclePassTracker']