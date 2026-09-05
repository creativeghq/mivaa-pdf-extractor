"""Evaluation of what the pipeline extracted, against what a person says is on the page.

`extraction_eval` is pure stdlib on purpose: CI installs pytest alone, and the unit
test loads it by path. Nothing in here touches the database; the internal route in
`app/api/extraction_eval_routes.py` does the reading and writing.
"""
