"""
Middleware package for the PDF2Markdown microservice.

The middleware stack is registered in ``app.main`` and is, in order:
CORS, ErrorLogging, JSONSerialization, JWTAuth, Performance, Logging.

Import middleware from its own module rather than re-exporting here — a
re-export in this file is what kept ``ValidationMiddleware`` looking wired up
for as long as it did (audit #12, finding B): 1,563 lines of size caps,
content-type checks and path-traversal guards that were never in the stack and
so never ran on a single request. Unreachable protection is worse than none,
because it reads as protection. Deleted along with the equally unreferenced
``app/core/validation`` package (2,437 lines).
"""
