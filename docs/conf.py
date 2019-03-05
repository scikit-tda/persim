# -*- coding: utf-8 -*-
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
from persim import __version__
from sktda_docs_config import *

project = u'Persim'
copyright = u'2019, Nathaniel Saul'
author = u'Nathaniel Saul'

version = __version__
release = __version__

html_theme_options.update({
  # Google Analytics info
  'ga_ua': 'UA-124965309-3',
  'ga_domain': '',
  'gh_url': 'scikit-tda/persim'
})

html_short_title = project
htmlhelp_basename = 'Persimdoc'
