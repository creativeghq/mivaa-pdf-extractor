"""Authentication and tenancy primitives.

Deliberately free of service imports (Supabase, RAG, settings) so the security
logic in here can be unit-tested with nothing installed but fastapi + pytest.
"""
