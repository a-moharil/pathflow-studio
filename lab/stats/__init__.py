"""Efficient realisation layer: indexed observables, run navigation and statistics.

Everything in this package reads recorded results. It never trains, never mutates a
checkpoint and never queues numerical work. Statistics are computed from the
observable index, which is synchronised incrementally from stored measurements and
completed jobs.
"""
