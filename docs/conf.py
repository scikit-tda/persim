# -*- coding: utf-8 -*-
import os
import sys
from importlib import metadata

sys.path.insert(0, os.path.abspath("."))
from sktda_docs_config import *


project = "Persim"
copyright = "2019, Nathaniel Saul"
author = "Nathaniel Saul"

version = metadata.version("persim")
release = metadata.version("persim")

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
