"""Ask a spreadsheet a question — validated read-only SQL over an in-memory DuckDB.

Adopted 2026-09-05 from the GAIK toolkit's `TabularAgent`, security model included:
generated SQL is parsed and allow-listed (`sql_guard`), the engine is locked before the
first generated query runs (`loader.lock_down`), and there is no Python execution
anywhere — a pandas-style agent that `exec`s model output turns a hostile spreadsheet
cell into remote code execution (CVE-2024-12366).
"""
