import os

"""
Common functionality for data science applications in the Wellcome Collection
platform
"""

# We're using a combination of github releases and actions to publish this
# package to pypi.
# We have a workflow set up to publish this package when a new github release
# is created. Github actions creates a GITHUB_REF environment variable for the
# build, where GITHUB_REF = the tag ref that triggered the workflow (see https://docs.github.com/en/actions/configuring-and-managing-workflows/using-environment-variables)
# eg. if a release is tagged 0.1.2, we'll have a GITHUB_REF="0.1.2" env var.
# Flit then uses this __version__ variable to tag the pypi release.
__version__ = os.environ["GITHUB_REF"]
