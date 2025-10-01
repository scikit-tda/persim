# -*- coding: utf-8 -*-
import re
import os
import sys

sys.path.insert(0, os.path.abspath("."))
from sktda_docs_config import *


def get_version():
    VERSIONFILE = "persim/_version.py"
    verstrline = open(VERSIONFILE, "rt").read()
    VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
    mo = re.search(VSRE, verstrline, re.M)
    if mo:
        return mo.group(1)
    else:
        raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))


project = "Persim"
copyright = "2019, Nathaniel Saul"
author = "Nathaniel Saul"

version = get_version()
release = get_version()

language = "en"

html_theme_options.update(
    {
        "collapse_naviation": False,
        # Google Analytics info
        "ga_ua": "UA-124965309-3",
        "ga_domain": "",
        "gh_url": "scikit-tda/persim",
    }
)

html_short_title = project
htmlhelp_basename = "Persimdoc"

autodoc_default_options = {"members": False, "maxdepth": 1}

autodoc_member_order = "groupwise"


# Set canonical URL from the Read the Docs Domain
html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "")

# Tell Jinja2 templates the build is running on Read the Docs
if os.environ.get("READTHEDOCS", "") == "True":
    if "html_context" not in globals():
        html_context = {}
    html_context["READTHEDOCS"] = True
