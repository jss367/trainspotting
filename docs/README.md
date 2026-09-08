# The site

`index.html` plus `js/` and `data/`, served as-is from this directory by GitHub
Pages. No build step.

The script is an ES module (`js/main.js` boots `js/app.js`), and browsers refuse
to load modules over `file://`, so preview it through a local server:

```bash
python -m http.server -d docs 8000
# then open http://localhost:8000/
```

`data/` is written by `scripts/export_site_data.py`; do not edit it by hand.
