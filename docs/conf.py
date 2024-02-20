# -*- coding: utf-8 -*-
import os
import sys

sys.path.insert(0, os.path.abspath("."))
from sktda_docs_config import *

from persim import __version__

project = "Persim"
copyright = "2019, Nathaniel Saul"
author = "Nathaniel Saul"

version = __version__
release = __version__

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
