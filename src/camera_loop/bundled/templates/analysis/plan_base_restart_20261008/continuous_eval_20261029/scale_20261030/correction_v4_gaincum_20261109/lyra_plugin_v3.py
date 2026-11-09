import json
from pathlib import Path
R=Path(__file__).resolve().parent
import sys as _s;_s.path.append(str(R))  # gaussverse 2026-11-09: gain_rule.py (shared gain rule)
from checked_common import install
install()
meta=json.loads((R/'source_manifest.json').read_text())['lyra_plugin.py']
exec(compile((R/'snapshots/lyra_plugin.py').read_text(),meta['original'],'exec'),dict(__name__='__main__',__file__=meta['original']))
