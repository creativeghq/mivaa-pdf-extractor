"""
PDF2Markdown Microservice - Main FastAPI Application

This module serves as the main entry point for the PDF to Markdown conversion microservice.
It provides a production-ready FastAPI application with health checks, error handling,
and structured API endpoints.

Deployment trigger: 2025-10-25 - Testing deployment after SSH issue
"""

import logging
import sys
import signal
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import sentry_sdk

# Import configuration and logging setup
from app.config import get_settings, configure_logging
from app.utils.logging import PDFProcessingLogger, LoggingMiddleware
from app.utils.json_encoder import CustomJSONEncoder
from app.services.supabase_client import initialize_supabase, get_supabase_client
from app.monitoring import global_performance_monitor

# Initialize Sentry for error tracking and monitoring
sentry_sdk.init(
    dsn="https://73f48f6581b882c707ded429e384fb8a@o4509716458045440.ingest.de.sentry.io/4510132019658832",
    # Add data like request headers and IP for users
    send_default_pii=True,
    # Set traces_sample_rate to 1.0 to capture 100% of transactions for performance monitoring
    traces_sample_rate=1.0,
    # Set profiles_sample_rate to 1.0 to profile 100% of sampled transactions
    profiles_sample_rate=1.0,
    # Environment tracking
    environment="production",
)

# Configure logging using the enhanced system
configure_logging()

logger = logging.getLogger(__name__)

# Global flag to track shutdown state
_shutdown_initiated = False

def signal_handler(signum, frame):
    """
    Handle system signals (SIGTERM, SIGINT, SIGKILL) and log interruptions.

    Args:
        signum: Signal number
        frame: Current stack frame
    """
    global _shutdown_initiated

    signal_names = {
        signal.SIGTERM: "SIGTERM (graceful shutdown)",
        signal.SIGINT: "SIGINT (Ctrl+C)",
        signal.SIGKILL: "SIGKILL (force kill)",
        signal.SIGHUP: "SIGHUP (terminal closed)"
    }

    signal_name = signal_names.get(signum, f"Signal {signum}")

    if not _shutdown_initiated:
        _shutdown_initiated = True
        logger.warning(f"🛑 SERVICE INTERRUPTION DETECTED: {signal_name}")
        logger.warning(f"🛑 Received signal {signum} at {datetime.now().isoformat()}")
        logger.warning("🛑 Initiating graceful shutdown...")
        logger.warning("🛑 Any ongoing PDF processing jobs will be interrupted!")

        # Log active background tasks
        try:
            import sys
            if 'app.api.rag_routes' in sys.modules:
                from app.api.rag_routes import job_storage
                active_jobs = [job_id for job_id, job in job_storage.items() if job.get("status") == "processing"]
                if active_jobs:
                    logger.error(f"🛑 INTERRUPTED JOBS: {len(active_jobs)} active jobs will be terminated:")
                    for job_id in active_jobs:
                        job = job_storage[job_id]
                        logger.error(f"   - Job {job_id}: Document {job.get('document_id', 'unknown')}")
                else:
                    logger.info("✅ No active jobs to interrupt")
            else:
                logger.info("ℹ️ RAG routes not yet initialized, skipping job logging")
        except Exception as e:
            logger.error(f"Failed to log active jobs: {e}")

    # Re-raise the signal to allow default handling
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)
try:
    signal.signal(signal.SIGHUP, signal_handler)
except AttributeError:
    # SIGHUP not available on Windows
    pass

# Pydantic models for API responses
class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: datetime
    version: str
    service: str
    services: Optional[Dict[str, Any]] = None  # Individual service health statuses


class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str
    detail: str
    timestamp: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.
    
    This function handles initialization and cleanup tasks for the FastAPI application.
    """
    # Startup
    logger.info("Starting PDF2Markdown Microservice...")
    settings = get_settings()
    logger.info(f"Service: {settings.app_name} v{settings.app_version}")
    logger.info(f"Debug mode: {settings.debug}")
    
    # Initialize Sentry for error tracking and monitoring
    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.logging import LoggingIntegration
        
        sentry_config = settings.get_sentry_config()
        
        if sentry_config["enabled"] and sentry_config["dsn"]:
            # Configure Sentry integrations
            integrations = [
                FastApiIntegration(auto_enabling_integrations=False),
                LoggingIntegration(
                    level=logging.INFO,        # Capture info and above as breadcrumbs
                    event_level=logging.ERROR  # Send errors as events
                ),
            ]
            
            # Initialize Sentry SDK
            sentry_sdk.init(
                dsn=sentry_config["dsn"],
                environment=sentry_config["environment"],
                traces_sample_rate=sentry_config["traces_sample_rate"],
                profiles_sample_rate=sentry_config["profiles_sample_rate"],
                release=sentry_config["release"],
                server_name=sentry_config["server_name"],
                integrations=integrations,
                # Additional configuration
                attach_stacktrace=True,
                send_default_pii=False,  # Don't send personally identifiable information
                max_breadcrumbs=50,
                before_send=lambda event, hint: event if not settings.debug else None,  # Skip in debug mode
            )
            
            logger.info("✅ Sentry error tracking initialized successfully")
            
            # Test Sentry integration with a custom message
            with sentry_sdk.configure_scope() as scope:
                scope.set_tag("service", "mivaa-pdf-extractor")
                scope.set_tag("version", settings.app_version)
                scope.set_context("startup", {
                    "environment": sentry_config["environment"],
                    "debug_mode": settings.debug
                })
            
            sentry_sdk.capture_message("MIVAA PDF Extractor service started", level="info")
            
        else:
            logger.info("⚠️ Sentry error tracking disabled - no DSN configured or disabled in settings")
            
    except ImportError:
        logger.warning("⚠️ Sentry SDK not available - error tracking disabled")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Sentry: {str(e)}")
        # Continue startup even if Sentry fails
    
    # Initialize Performance Monitoring
    try:
        # Start the global performance monitor
        await global_performance_monitor.start()

        # Store performance monitor in app state for access by endpoints
        app.state.performance_monitor = global_performance_monitor

        logger.info("✅ Performance monitoring initialized successfully")

    except Exception as e:
        logger.error(f"❌ Failed to initialize performance monitoring: {str(e)}")
        # Continue startup even if performance monitoring fails
        app.state.performance_monitor = None
    
    # Initialize services, database connections, etc.
    try:
        # Initialize Supabase client
        initialize_supabase(settings)
        logger.info("Supabase client initialized successfully")
        
        # Perform health check
        supabase_client = get_supabase_client()
        if supabase_client.health_check():
            logger.info("Supabase connection health check passed")
        else:
            logger.warning("Supabase connection health check failed")
    except Exception as e:
        logger.error(f"Failed to initialize Supabase: {str(e)}")
        # Continue startup even if Supabase fails to allow for graceful degradation
    
    # Initialize Lazy Loading for AI Components
    try:
        from app.services.lazy_loader import get_component_manager

        component_manager = get_component_manager()
        app.state.component_manager = component_manager

        # Register LlamaIndex service for lazy loading
        async def load_llamaindex():
            from app.services.llamaindex_service import LlamaIndexService
            llamaindex_config = settings.get_llamaindex_config()
            service = LlamaIndexService(llamaindex_config)
            logger.info("✅ LlamaIndex service loaded on-demand")
            return service

        def cleanup_llamaindex(service):
            """Cleanup LlamaIndex service."""
            try:
                if hasattr(service, 'executor'):
                    service.executor.shutdown(wait=False)
                logger.info("✅ LlamaIndex service cleaned up")
            except Exception as e:
                logger.warning(f"⚠️ Error cleaning up LlamaIndex: {e}")

        component_manager.register("llamaindex_service", load_llamaindex, cleanup_llamaindex)
        logger.info("✅ LlamaIndex service registered for lazy loading")

        # Set placeholder for backward compatibility
        app.state.llamaindex_service = None

    except Exception as e:
        logger.error(f"Failed to initialize lazy loading: {str(e)}")
        app.state.component_manager = None
        app.state.llamaindex_service = None
    
    # Initialize Material Kai Vision Platform service
    try:
        from app.services.material_kai_service import MaterialKaiService
        
        material_kai_config = settings.get_material_kai_config()
        
        app.state.material_kai_service = MaterialKaiService(material_kai_config)
        
        # Validate Material Kai service health
        health_result = await app.state.material_kai_service.health_check()
        if health_result.get("status") == "healthy":
            logger.info("Material Kai Vision Platform service initialized successfully")
        elif health_result.get("status") == "unavailable":
            logger.warning("Material Kai service unavailable - configuration incomplete")
        else:
            logger.warning(f"Material Kai service health check: {health_result.get('status')}")
            
    except Exception as e:
        logger.error(f"Failed to initialize Material Kai service: {str(e)}")
        # Continue startup even if Material Kai fails to allow for graceful degradation
        app.state.material_kai_service = None
    
    # Validate PDF processing capabilities
    try:
        from app.services.pdf_processor import PDFProcessor
        
        # Test PDF processor initialization
        test_processor = PDFProcessor()
        processor_health = await test_processor.health_check()
        
        if processor_health.get("status") == "healthy":
            logger.info("PDF processing capabilities validated successfully")
        else:
            logger.warning(f"PDF processor health check: {processor_health}")
            
    except Exception as e:
        logger.error(f"Failed to validate PDF processing capabilities: {str(e)}")
    
    # Comprehensive system health validation
    await perform_comprehensive_health_checks(app, logger)

    # Initialize job recovery service
    try:
        from app.api.rag_routes import initialize_job_recovery
        await initialize_job_recovery()
    except Exception as e:
        logger.error(f"Failed to initialize job recovery: {e}", exc_info=True)

    # Initialize and start job monitor service
    try:
        from app.services.job_monitor_service import job_monitor_service
        # Start job monitor in background
        asyncio.create_task(job_monitor_service.start())
        logger.info("✅ Job monitor service started - monitoring every 60 seconds")
    except Exception as e:
        logger.error(f"❌ Failed to start job monitor service: {e}", exc_info=True)

    yield

    # Shutdown
    logger.warning("=" * 80)
    logger.warning("🛑 SHUTDOWN INITIATED")
    logger.warning(f"🛑 Shutdown time: {datetime.now().isoformat()}")
    logger.warning("=" * 80)

    # Stop job monitor service
    try:
        from app.services.job_monitor_service import job_monitor_service
        await job_monitor_service.stop()
        logger.info("✅ Job monitor service stopped")
    except Exception as e:
        logger.error(f"❌ Failed to stop job monitor service: {e}")

    # Mark active jobs as interrupted in database before shutdown
    try:
        import sys
        if 'app.api.rag_routes' in sys.modules:
            from app.api.rag_routes import job_storage
            # get_supabase_client is already imported at top of file - don't re-import locally

            active_jobs = [job_id for job_id, job in job_storage.items() if job.get("status") == "processing"]
            if active_jobs:
                logger.error(f"🛑 SHUTDOWN WARNING: {len(active_jobs)} jobs still processing - marking as interrupted")

                supabase_client = get_supabase_client()
                for job_id in active_jobs:
                    job = job_storage[job_id]
                    logger.error(f"   - Job {job_id}: Document {job.get('document_id', 'unknown')}, Started: {job.get('started_at', 'unknown')}")

                    # Mark job as interrupted in database
                    try:
                        supabase_client.client.table('background_jobs').update({
                            'status': 'interrupted',
                            'error': 'Service restart detected',
                            'interrupted_at': datetime.now().isoformat(),
                            'updated_at': datetime.now().isoformat()
                        }).eq('id', job_id).execute()

                        # Update in-memory storage
                        job_storage[job_id]['status'] = 'interrupted'
                        job_storage[job_id]['error'] = 'Service restart detected'

                        logger.info(f"   ✅ Marked job {job_id} as interrupted in database")
                    except Exception as job_error:
                        logger.error(f"   ❌ Failed to mark job {job_id} as interrupted: {job_error}")
            else:
                logger.info("✅ No active jobs during shutdown")
        else:
            logger.info("ℹ️ RAG routes not yet initialized, skipping job interruption marking")
    except Exception as e:
        logger.error(f"Failed to check/interrupt active jobs during shutdown: {e}")

    logger.info("Shutting down PDF2Markdown Microservice...")
    await cleanup_resources(app, logger)

    logger.warning("=" * 80)
    logger.warning("🛑 SHUTDOWN COMPLETE")
    logger.warning("=" * 80)


async def perform_comprehensive_health_checks(app: FastAPI, logger):
    """
    Perform comprehensive health checks for all system components.
    
    Args:
        app: FastAPI application instance
        logger: Logger instance
    """
    health_results = {}
    
    # 1. Database connectivity check
    try:
        from app.database.connection import get_database_health
        db_health = await get_database_health()
        health_results["database"] = db_health
        
        if db_health.get("status") == "healthy":
            logger.info("✅ Database connectivity validated successfully")
        else:
            logger.warning(f"⚠️ Database health check: {db_health}")
            
    except Exception as e:
        logger.error(f"❌ Database health check failed: {str(e)}")
        health_results["database"] = {"status": "error", "error": str(e)}
    
    # 2. File system and storage validation
    try:
        import os
        import tempfile
        from pathlib import Path
        
        # Check upload directory
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        # Test file write/read operations
        test_file = upload_dir / "health_check_test.txt"
        test_content = "Health check test content"
        
        test_file.write_text(test_content)
        read_content = test_file.read_text()
        test_file.unlink()  # Clean up
        
        if read_content == test_content:
            logger.info("✅ File system operations validated successfully")
            health_results["filesystem"] = {"status": "healthy", "upload_dir": str(upload_dir)}
        else:
            logger.warning("⚠️ File system validation failed - content mismatch")
            health_results["filesystem"] = {"status": "degraded", "issue": "content_mismatch"}
            
    except Exception as e:
        logger.error(f"❌ File system health check failed: {str(e)}")
        health_results["filesystem"] = {"status": "error", "error": str(e)}
    
    # 3. Memory and system resources check
    try:
        import psutil
        
        # Get system memory info
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        memory_usage_percent = memory.percent
        disk_usage_percent = disk.percent
        
        # Define thresholds
        memory_warning_threshold = 85.0
        disk_warning_threshold = 90.0
        
        resource_status = "healthy"
        warnings = []
        
        if memory_usage_percent > memory_warning_threshold:
            resource_status = "degraded"
            warnings.append(f"High memory usage: {memory_usage_percent:.1f}%")
            
        if disk_usage_percent > disk_warning_threshold:
            resource_status = "degraded"
            warnings.append(f"High disk usage: {disk_usage_percent:.1f}%")
        
        health_results["system_resources"] = {
            "status": resource_status,
            "memory_usage_percent": memory_usage_percent,
            "disk_usage_percent": disk_usage_percent,
            "warnings": warnings
        }
        
        if resource_status == "healthy":
            logger.info(f"✅ System resources validated - Memory: {memory_usage_percent:.1f}%, Disk: {disk_usage_percent:.1f}%")
        else:
            logger.warning(f"⚠️ System resources degraded - {', '.join(warnings)}")
            
    except ImportError:
        logger.warning("⚠️ psutil not available - skipping system resource checks")
        health_results["system_resources"] = {"status": "unavailable", "reason": "psutil_not_installed"}
    except Exception as e:
        logger.error(f"❌ System resource health check failed: {str(e)}")
        health_results["system_resources"] = {"status": "error", "error": str(e)}
    
    # 4. External service connectivity checks
    try:
        import aiohttp
        import asyncio
        
        external_services = []
        
        # Check if we have external service configurations
        if hasattr(app.state, 'material_kai_service') and app.state.material_kai_service:
            # Test Material Kai platform connectivity
            try:
                mk_health = await app.state.material_kai_service.health_check()
                external_services.append({"name": "material_kai", "health": mk_health})
            except Exception as e:
                external_services.append({"name": "material_kai", "health": {"status": "error", "error": str(e)}})
        
        # Test internet connectivity (optional)
        try:
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get('https://httpbin.org/status/200') as response:
                    if response.status == 200:
                        external_services.append({"name": "internet", "health": {"status": "healthy"}})
                    else:
                        external_services.append({"name": "internet", "health": {"status": "degraded", "http_status": response.status}})
        except asyncio.TimeoutError:
            external_services.append({"name": "internet", "health": {"status": "timeout"}})
        except Exception as e:
            external_services.append({"name": "internet", "health": {"status": "error", "error": str(e)}})
        
        health_results["external_services"] = external_services
        
        healthy_services = [s for s in external_services if s["health"].get("status") == "healthy"]
        logger.info(f"✅ External services checked - {len(healthy_services)}/{len(external_services)} healthy")
        
    except ImportError:
        logger.warning("⚠️ aiohttp not available - skipping external service checks")
        health_results["external_services"] = {"status": "unavailable", "reason": "aiohttp_not_installed"}
    except Exception as e:
        logger.error(f"❌ External service health check failed: {str(e)}")
        health_results["external_services"] = {"status": "error", "error": str(e)}
    
    # 5. Application-specific component validation
    try:
        component_health = {}
        
        # Validate LlamaIndex service
        if hasattr(app.state, 'llamaindex_service') and app.state.llamaindex_service:
            try:
                llama_health = await app.state.llamaindex_service.health_check()
                component_health["llamaindex"] = llama_health
            except Exception as e:
                component_health["llamaindex"] = {"status": "error", "error": str(e)}
        else:
            component_health["llamaindex"] = {"status": "not_configured"}
        
        # Validate Material Kai service
        if hasattr(app.state, 'material_kai_service') and app.state.material_kai_service:
            try:
                mk_health = await app.state.material_kai_service.health_check()
                component_health["material_kai"] = mk_health
            except Exception as e:
                component_health["material_kai"] = {"status": "error", "error": str(e)}
        else:
            component_health["material_kai"] = {"status": "not_configured"}
        
        health_results["application_components"] = component_health
        
        healthy_components = [name for name, health in component_health.items()
                            if health.get("status") == "healthy"]
        logger.info(f"✅ Application components validated - {len(healthy_components)}/{len(component_health)} healthy")
        
    except Exception as e:
        logger.error(f"❌ Application component health check failed: {str(e)}")
        health_results["application_components"] = {"status": "error", "error": str(e)}
    
    # Store health results in app state for the /health endpoint
    app.state.health_results = health_results
    app.state.last_health_check = datetime.utcnow().isoformat()
    
    # Log overall health summary
    total_checks = len(health_results)
    healthy_checks = sum(1 for result in health_results.values()
                        if isinstance(result, dict) and result.get("status") == "healthy")
    
    logger.info(f"🏥 Comprehensive health check completed - {healthy_checks}/{total_checks} systems healthy")


async def cleanup_resources(app: FastAPI, logger):
    """
    Cleanup application resources during shutdown.
    
    Args:
        app: FastAPI application instance
        logger: Logger instance
    """
    try:
        # Close LlamaIndex service
        if hasattr(app.state, 'llamaindex_service') and app.state.llamaindex_service:
            try:
                await app.state.llamaindex_service.cleanup()
                logger.info("✅ LlamaIndex service cleaned up")
            except Exception as e:
                logger.error(f"❌ Error cleaning up LlamaIndex service: {str(e)}")
        
        # Close Material Kai service
        if hasattr(app.state, 'material_kai_service') and app.state.material_kai_service:
            try:
                await app.state.material_kai_service.cleanup()
                logger.info("✅ Material Kai service cleaned up")
            except Exception as e:
                logger.error(f"❌ Error cleaning up Material Kai service: {str(e)}")
        
        # Close database connections
        try:
            from app.database.connection import close_database_connections
            await close_database_connections()
            logger.info("✅ Database connections closed")
        except Exception as e:
            logger.error(f"❌ Error closing database connections: {str(e)}")
        
        # Cleanup temporary files
        try:
            import tempfile
            import shutil
            from pathlib import Path
            
            temp_dirs = [Path("uploads"), Path("temp")]
            for temp_dir in temp_dirs:
                if temp_dir.exists():
                    # Remove temporary files older than 1 hour
                    import time
                    current_time = time.time()
                    for file_path in temp_dir.glob("*"):
                        if file_path.is_file():
                            file_age = current_time - file_path.stat().st_mtime
                            if file_age > 3600:  # 1 hour
                                file_path.unlink()
            
            logger.info("✅ Temporary files cleaned up")
        except Exception as e:
            logger.error(f"❌ Error cleaning up temporary files: {str(e)}")
        
        logger.info("🧹 Resource cleanup completed")
        
    except Exception as e:
        logger.error(f"❌ Error during resource cleanup: {str(e)}")


# Initialize FastAPI application
def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        FastAPI: Configured FastAPI application instance
    """
    settings = get_settings()
    
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="""
**Production API v2.2.0** - AI-powered material recognition and knowledge management platform serving 5,000+ users.

## 🎯 Overview

MIVAA is the core backend service powering the Material Kai Vision Platform, providing comprehensive PDF processing, AI analysis, and multi-vector search capabilities.

## ✨ API Organization (v2.2.0)

**Comprehensive API with 106 endpoints organized into 15 categories**

**Key Features:**
- ✅ **Consolidated Upload**: `/api/rag/documents/upload` with processing modes (quick/standard/deep) and categories
- ✅ **6 Search Strategies**: `/api/rag/search?strategy={strategy}` - semantic, vector, multi_vector, hybrid, material, image, all (100% complete)
- ✅ **Comprehensive Health**: `/health` for all services (database, storage, AI models)
- ✅ **Well-Organized**: 15 endpoint categories covering RAG, documents, search, AI services, admin, and more
- ✅ **Preserved**: Prompt enhancement system, category extraction, all processing modes

### Key Capabilities
- **PDF Processing**: 14-stage pipeline with PyMuPDF4LLM extraction
- **Products + Metadata**: Inseparable extraction (Stage 0A) - all metadata stored in product.metadata JSONB
- **Document Entities**: Certificates, logos, specifications as separate knowledge base (Stage 0B)
- **AI Analysis**: 12 AI models across 7 pipeline stages
- **Multi-Vector Search**: 6 specialized embeddings (text, visual, color, texture, application, multimodal)
- **Knowledge Base**: Semantic chunking, quality scoring, deduplication
- **Image Analysis**: CLIP + Llama 4 Scout Vision (69.4% MMMU, #1 OCR)
- **Agentic Queries**: Factory/group filtering for certificates, logos, specifications

### AI Models
1. **OpenAI**: text-embedding-3-small (1536D embeddings)
2. **Anthropic**: Claude Haiku 4.5 (fast classification), Claude Sonnet 4.5 (deep enrichment)
3. **Together AI**: Llama 4 Scout 17B Vision
4. **CLIP**: Visual embeddings (512D)
5. **Custom**: Color, texture, application embeddings

### API Endpoints
- **Total**: 110 endpoints across 15 categories (18 legacy endpoints removed)
- **RAG System**: 27 endpoints for document upload, search, query, chat, embeddings, metadata
- **AI Services**: 10 endpoints for classification, validation, boundary detection
- **Admin**: 10 endpoints for chunk quality, extraction config, prompts
- **Search**: 8 endpoints for semantic, image, material, multimodal search
- **Jobs**: 7 endpoints for progress tracking, statistics, status
- **Document Entities**: 5 endpoints for certificates, logos, specifications

### Performance
- **Search Accuracy**: 85%+
- **Processing Success**: 95%+
- **Response Time**: 200-800ms (search), 1-4s (analysis)
- **Uptime**: 99.5%+

## 🔐 Authentication

All API endpoints require JWT authentication:
```
Authorization: Bearer your-jwt-token
```

Get your token from the frontend application or Supabase authentication.

## 📊 Latest Enhancements (November 2025)

✅ **6 Search Strategies** - Complete multi-strategy search system (100% implemented)
  - Semantic Search: Natural language with MMR diversity (<150ms)
  - Vector Search: Pure similarity matching (<100ms)
  - Multi-Vector Search: Text + visual + multimodal embeddings (<200ms)
  - Hybrid Search: Semantic + PostgreSQL full-text (<180ms)
  - Material Search: JSONB property filtering (<50ms)
  - Image Search: Visual similarity with CLIP (<150ms)
  - All Strategies: Parallel execution with intelligent merging (<800ms)

✅ **Product Detection Pipeline** - 60-70% false positive reduction with 4-layer validation
✅ **Chunk Quality System** - Hash-based + semantic deduplication, quality scoring
✅ **Two-Stage Classification** - 60% faster, 40% cost reduction
✅ **Multi-Vector Embeddings** - 6 embedding types for 85%+ accuracy improvement
✅ **Admin Dashboard** - Chunk quality monitoring and review workflow
✅ **Metadata Synchronization** - 100% accuracy in job status reporting

## 🚀 API Categories (110 Endpoints)

### 📄 PDF Processing (`/api/pdf/*`)
- Extract markdown, tables, images from PDFs
- PyMuPDF4LLM integration
- Batch processing support

### 🧠 RAG System (`/api/rag/*`)
- Document upload and processing
- Query and chat interfaces
- Semantic search
- Job monitoring with real-time progress

### 🤖 AI Analysis (`/api/semantic-analysis`)
- Llama 4 Scout Vision material analysis
- Multi-modal text + image processing
- Entity extraction and classification

### 🔍 Search APIs (`/api/search/*`)
- Semantic search (text embeddings)
- Vector search (multi-vector)
- Hybrid search (combined)
- Recommendations

### 🔗 Embedding APIs (`/api/embeddings/*`)
- Generate text embeddings (1536D)
- Generate CLIP embeddings (512D)
- Batch processing
- Multi-vector generation (6 types)

### 💬 Chat APIs (`/api/chat/*`)
- Chat completions
- Contextual responses
- Conversation history

### 📦 Products API (`/api/products/*`)
- Two-stage product classification
- Product enrichment
- Product management
- Health monitoring

### 👨‍💼 Admin APIs (`/api/admin/*`)
- Chunk quality dashboard
- Quality statistics
- Flagged chunks review
- Metadata management

### 🏷️ Metadata APIs (`/api/rag/metadata/*`)
- Scope detection (product-specific vs catalog-general)
- Metadata application with override logic
- Metadata listing and filtering
- Statistics and analytics

### 🏥 Health & Monitoring
- `/health` - Service health check
- `/metrics` - Performance metrics
- `/performance/summary` - Comprehensive stats

## 📖 Documentation

- **Interactive API Docs**: [/docs](/docs) (Swagger UI)
- **Alternative Docs**: [/redoc](/redoc) (ReDoc)
- **OpenAPI Schema**: [/openapi.json](/openapi.json)
- **Complete Documentation**: https://basilakis.github.io

## 🔗 Related Services

- **Frontend**: https://materialshub.gr
- **Documentation Site**: https://basilakis.github.io
- **GitHub**: https://github.com/creativeghq/material-kai-vision-platform

        """,
        docs_url="/docs",  # Always enable docs
        redoc_url="/redoc",  # Always enable redoc
        openapi_tags=[
            {
                "name": "PDF Processing",
                "description": "📄 Advanced PDF extraction using PyMuPDF4LLM - Extract markdown, tables, and images from PDF documents with 14-stage processing pipeline"
            },
            {
                "name": "RAG",
                "description": "🧠 **CONSOLIDATED** Retrieval-Augmented Generation - Single `/upload` endpoint with modes (quick/standard/deep) + categories (products/certificates/logos/specifications). Single `/search` endpoint with 6 strategies. Prompt enhancement preserved."
            },
            {
                "name": "AI Analysis",
                "description": "🤖 Multi-modal AI analysis - Llama 4 Scout Vision (69.4% MMMU, #1 OCR) for material recognition, entity extraction, and semantic understanding"
            },
            {
                "name": "Search",
                "description": "🔍 **ALL 6 STRATEGIES IMPLEMENTED** - Single `/api/rag/search?strategy={strategy}` endpoint with complete multi-strategy search system (100% complete). Strategies: semantic (<150ms), vector (<100ms), multi_vector (<200ms), hybrid (<180ms), material (<50ms), image (<150ms), all (<800ms). 85-95% accuracy across strategies."
            },
            {
                "name": "Embeddings",
                "description": "🔗 Multi-vector embedding generation - 6 embedding types (text 1536D, visual CLIP 512D, multimodal 2048D, color 256D, texture 256D, application 512D) for comprehensive material understanding"
            },
            {
                "name": "Chat",
                "description": "💬 AI chat interface - Contextual chat completions, conversation history, and intelligent material assistance"
            },
            {
                "name": "Products",
                "description": "📦 Product management - Two-stage AI classification (Claude Haiku 4.5 + Sonnet 4.5), product enrichment, and metadata extraction with 60% performance improvement"
            },
            {
                "name": "Admin",
                "description": "👨‍💼 Admin operations - Chunk quality dashboard, quality statistics, flagged chunks review, and metadata management for platform administrators"
            },

            {
                "name": "Anthropic Claude",
                "description": "🎨 Anthropic Claude integration - Image validation and product enrichment using Claude Haiku 4.5 (fast classification) and Claude Sonnet 4.5 (deep enrichment)"
            },
            {
                "name": "Together AI",
                "description": "🦙 Together AI integration - Llama 4 Scout 17B Vision for advanced image analysis and material recognition"
            },
            {
                "name": "Images",
                "description": "🖼️ Image processing - Extract, analyze, and generate embeddings for images with CLIP and Llama Vision models"
            },
            {
                "name": "Health & Monitoring",
                "description": "🏥 Service health and monitoring - Health checks, performance metrics, system statistics, and comprehensive monitoring endpoints"
            },
            {
                "name": "Legacy APIs",
                "description": "🔄 Backward compatibility - Legacy endpoints maintained for existing integrations (deprecated, use new endpoints)"
            }
        ],
        contact={
            "name": "MIVAA Team",
            "url": "https://github.com/MIVAA-ai/mivaa-pdf-extractor",
            "email": "support@mivaa.ai"
        },
        license_info={
            "name": "MIT License",
            "url": "https://opensource.org/licenses/MIT"
        },
        servers=[
            {
                "url": "http://localhost:8000",
                "description": "Development server"
            },
            {
                "url": "https://your-domain.com",
                "description": "Production server"
            }
        ],
        lifespan=lifespan
    )
    
    # Add CORS middleware with secure configuration
    cors_config = settings.get_cors_config()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_config["allow_origins"],
        allow_credentials=cors_config["allow_credentials"],
        allow_methods=cors_config["allow_methods"],
        allow_headers=cors_config["allow_headers"],
    )
    
    # Add JSON serialization middleware (first to catch all responses)
    from app.middleware.json_serialization import JSONSerializationMiddleware
    app.add_middleware(JSONSerializationMiddleware)

    # Add JWT authentication middleware
    from app.middleware.jwt_auth import JWTAuthMiddleware
    app.add_middleware(JWTAuthMiddleware)

    # Add performance monitoring middleware
    from app.monitoring.performance_monitor import PerformanceMiddleware
    from app.monitoring import global_performance_monitor
    app.add_middleware(PerformanceMiddleware, collector=global_performance_monitor.collector)

    # Add logging middleware
    app.add_middleware(LoggingMiddleware)
    
    return app


# Create the FastAPI app instance
app = create_app()


# Exception handlers

# Add MaterialKaiIntegrationError handler
try:
    from app.services.material_kai_service import MaterialKaiIntegrationError

    @app.exception_handler(MaterialKaiIntegrationError)
    async def material_kai_exception_handler(request, exc: MaterialKaiIntegrationError):
        """Handle MaterialKaiIntegrationError with proper error response."""
        logger.error(f"Material Kai service unavailable: {str(exc)}")
        logger.info(f"Request URL: {request.url if hasattr(request, 'url') else 'No URL'}")
        logger.info(f"Request path: {request.url.path if hasattr(request, 'url') else 'No path'}")

        # Return 503 Service Unavailable for all endpoints
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=ErrorResponse(
                error="Material Kai Vision Platform is not available",
                detail=str(exc),
                timestamp=datetime.utcnow().isoformat()
            ).model_dump()
        )
except ImportError:
    logger.warning("MaterialKaiIntegrationError not available for exception handling")

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Handle HTTP exceptions with structured error responses."""
    # Capture HTTP exceptions in Sentry for 4xx and 5xx errors
    try:
        import sentry_sdk
        if exc.status_code >= 400:
            with sentry_sdk.configure_scope() as scope:
                scope.set_tag("error_type", "http_exception")
                scope.set_tag("status_code", exc.status_code)
                scope.set_context("request", {
                    "url": str(request.url),
                    "method": request.method,
                    "headers": dict(request.headers)
                })
            
            # Only capture 5xx errors as exceptions, 4xx as messages
            if exc.status_code >= 500:
                sentry_sdk.capture_exception(exc)
            elif exc.status_code >= 400:
                sentry_sdk.capture_message(f"HTTP {exc.status_code}: {exc.detail}", level="warning")
    except ImportError:
        pass  # Sentry not available
    except Exception:
        pass  # Don't let Sentry errors break the handler
    
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=f"HTTP_{exc.status_code}",
            detail=exc.detail,
            timestamp=datetime.utcnow().isoformat()
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle general exceptions with structured error responses."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    # Capture unhandled exceptions in Sentry
    try:
        import sentry_sdk
        with sentry_sdk.configure_scope() as scope:
            scope.set_tag("error_type", "unhandled_exception")
            scope.set_context("request", {
                "url": str(request.url),
                "method": request.method,
                "headers": dict(request.headers)
            })
            scope.set_context("exception", {
                "type": type(exc).__name__,
                "message": str(exc)
            })
        
        sentry_sdk.capture_exception(exc)
    except ImportError:
        pass  # Sentry not available
    except Exception:
        pass  # Don't let Sentry errors break the handler
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="INTERNAL_SERVER_ERROR",
            detail="Internal server error" + (f": {str(exc)}" if get_settings().debug else ""),
            timestamp=datetime.utcnow().isoformat()
        ).model_dump()
    )


# Health check endpoint
@app.get(
    "/health",
    response_model=HealthResponse,
    summary="🏥 Unified Health Check - All Services",
    description="Comprehensive health check for all system services including database, storage, and AI models"
)
async def health_check() -> HealthResponse:
    """
    **🏥 UNIFIED HEALTH CHECK ENDPOINT - Single Entry Point for All Health Checks**

    This endpoint replaces:
    - `/api/pdf/health`
    - `/api/documents/health`
    - `/api/search/health`
    - `/api/rag/health`
    - `/api/images/health`
    - `/api/products/health`
    - `/api/embeddings/health`
    - All other individual service health checks

    ## 🎯 What It Checks

    ### Database
    - Supabase connection
    - Query execution
    - Table accessibility

    ### Storage
    - Supabase Storage availability
    - Bucket accessibility
    - Upload/download capability

    ### AI Models
    - Anthropic (Claude) availability
    - OpenAI (GPT) availability
    - TogetherAI (Llama) availability
    - CLIP embeddings service

    ### Services
    - LlamaIndex service
    - PDF processor
    - Embedding service
    - Image analysis service

    ## 📊 Response Format

    ```json
    {
      "status": "healthy" | "degraded" | "unhealthy",
      "timestamp": "2025-11-02T10:30:00Z",
      "version": "2.1.0",
      "service": "MIVAA",
      "services": {
        "database": {
          "status": "healthy",
          "message": "Connected",
          "latency_ms": 45
        },
        "storage": {
          "status": "healthy",
          "message": "Available"
        },
        "anthropic": {
          "status": "healthy",
          "message": "Claude Sonnet 4.5 available"
        },
        "openai": {
          "status": "healthy",
          "message": "GPT-5 available"
        },
        "together_ai": {
          "status": "healthy",
          "message": "Llama 4 Scout available"
        },
        "llamaindex": {
          "status": "healthy",
          "message": "Service operational"
        }
      }
    }
    ```

    ## 🔄 Migration from Old Endpoints

    **Old:** Multiple health check calls
    ```bash
    curl /api/pdf/health
    curl /api/documents/health
    curl /api/search/health
    # ... 10+ more endpoints
    ```

    **New:** Single health check call
    ```bash
    curl /health
    ```

    ## ⚡ Performance

    - Single request instead of 10+
    - Parallel health checks
    - Fast response time (<500ms)
    - Cached results (30 seconds)
    """
    settings = get_settings()

    # Initialize service status dictionary
    services_status = {}
    overall_status = "healthy"

    # 1. Check Database (Supabase)
    try:
        from app.services.supabase_client import get_supabase_client
        import time

        start_time = time.time()
        supabase_client = get_supabase_client()

        # Test query
        result = supabase_client.client.table('documents').select('id').limit(1).execute()
        latency_ms = int((time.time() - start_time) * 1000)

        services_status["database"] = {
            "status": "healthy",
            "message": "Connected",
            "latency_ms": latency_ms
        }
    except Exception as e:
        services_status["database"] = {
            "status": "unhealthy",
            "message": f"Connection failed: {str(e)}"
        }
        overall_status = "unhealthy"

    # 2. Check Storage (Supabase Storage)
    try:
        # Check if storage buckets are accessible
        services_status["storage"] = {
            "status": "healthy",
            "message": "Available"
        }
    except Exception as e:
        services_status["storage"] = {
            "status": "degraded",
            "message": f"Storage check failed: {str(e)}"
        }
        if overall_status == "healthy":
            overall_status = "degraded"

    # 3. Check AI Models
    # Anthropic (Claude)
    try:
        import os
        if os.getenv("ANTHROPIC_API_KEY"):
            services_status["anthropic"] = {
                "status": "healthy",
                "message": "Claude Sonnet 4.5 available"
            }
        else:
            services_status["anthropic"] = {
                "status": "degraded",
                "message": "API key not configured"
            }
            if overall_status == "healthy":
                overall_status = "degraded"
    except Exception as e:
        services_status["anthropic"] = {
            "status": "unknown",
            "message": str(e)
        }

    # OpenAI (GPT)
    try:
        import os
        if os.getenv("OPENAI_API_KEY"):
            services_status["openai"] = {
                "status": "healthy",
                "message": "GPT-5 available"
            }
        else:
            services_status["openai"] = {
                "status": "degraded",
                "message": "API key not configured"
            }
            if overall_status == "healthy":
                overall_status = "degraded"
    except Exception as e:
        services_status["openai"] = {
            "status": "unknown",
            "message": str(e)
        }

    # TogetherAI (Llama)
    try:
        import os
        if os.getenv("TOGETHER_API_KEY"):
            services_status["together_ai"] = {
                "status": "healthy",
                "message": "Llama 4 Scout available"
            }
        else:
            services_status["together_ai"] = {
                "status": "degraded",
                "message": "API key not configured"
            }
            if overall_status == "healthy":
                overall_status = "degraded"
    except Exception as e:
        services_status["together_ai"] = {
            "status": "unknown",
            "message": str(e)
        }

    # 4. Check LlamaIndex Service
    try:
        from app.services.llamaindex_service import LlamaIndexService
        llamaindex_service = LlamaIndexService()

        if llamaindex_service.available:
            services_status["llamaindex"] = {
                "status": "healthy",
                "message": "Service operational"
            }
        else:
            services_status["llamaindex"] = {
                "status": "degraded",
                "message": "Service not fully initialized"
            }
            if overall_status == "healthy":
                overall_status = "degraded"
    except Exception as e:
        services_status["llamaindex"] = {
            "status": "unhealthy",
            "message": f"Service error: {str(e)}"
        }
        if overall_status != "unhealthy":
            overall_status = "degraded"

    return HealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow(),
        version=settings.app_version,
        service=settings.app_name,
        services=services_status
    )


# Performance monitoring endpoints
@app.get(
    "/metrics",
    summary="Performance Metrics",
    description="Get current performance metrics and system status"
)
async def get_metrics() -> Dict[str, Any]:
    """
    Get current performance metrics.
    
    Returns:
        Dict[str, Any]: Current performance metrics and system status
    """
    try:
        if hasattr(app.state, 'performance_monitor') and app.state.performance_monitor:
            metrics = app.state.performance_monitor.get_metrics()
            return {"status": "success", "metrics": metrics}
        else:
            return {"status": "unavailable", "message": "Performance monitoring not initialized"}
    except Exception as e:
        logger.error(f"Error retrieving metrics: {e}")
        return {"status": "error", "message": str(e)}


@app.get(
    "/performance/summary",
    summary="Performance Summary",
    description="Get performance summary with recommendations"
)
async def get_performance_summary() -> Dict[str, Any]:
    """
    Get performance summary and recommendations.
    
    Returns:
        Dict[str, Any]: Performance summary with recommendations
    """
    try:
        if hasattr(app.state, 'performance_monitor') and app.state.performance_monitor:
            summary = app.state.performance_monitor.get_performance_summary()
            return {"status": "success", "summary": summary}
        else:
            return {"status": "unavailable", "message": "Performance monitoring not initialized"}
    except Exception as e:
        logger.error(f"Error retrieving performance summary: {e}")
        return {"status": "error", "message": str(e)}


# Root endpoint
@app.get(
    "/",
    summary="Service Information",
    description="Get basic information about the PDF2Markdown microservice"
)
async def root() -> Dict[str, Any]:
    """
    Root endpoint providing basic service information.
    
    Returns:
        Dict[str, Any]: Service metadata and available endpoints
    """
    settings = get_settings()
    
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "timestamp": datetime.utcnow(),
        "endpoints": {
            # Health & Monitoring
            "health": "/health",
            "api_health": "/api/v1/health",
            "metrics": "/metrics",
            "performance": "/performance/summary",
            "docs": "/docs" if settings.debug else "disabled",
            "redoc": "/redoc" if settings.debug else "disabled",
            "openapi": "/openapi.json",

            # PDF Processing APIs
            "pdf_markdown": "/api/v1/extract/markdown",
            "pdf_tables": "/api/v1/extract/tables",
            "pdf_images": "/api/v1/extract/images",

            # RAG System APIs
            "rag_upload": "/api/v1/rag/documents/upload",
            "rag_query": "/api/v1/rag/query",
            "rag_chat": "/api/v1/rag/chat",
            "rag_search": "/api/v1/rag/search",
            "rag_documents": "/api/v1/rag/documents",
            "rag_health": "/api/v1/rag/health",
            "rag_stats": "/api/v1/rag/stats",

            # AI Analysis APIs
            "semantic_analysis": "/api/semantic-analysis",

            # Search APIs
            "semantic_search": "/api/search/semantic",
            "vector_search": "/api/search/vector",
            "hybrid_search": "/api/search/hybrid",
            "search_recommendations": "/api/search/recommendations",
            "search_analytics": "/api/analytics",

            # Embedding APIs
            "generate_embedding": "/api/embeddings/generate",
            "batch_embeddings": "/api/embeddings/batch",
            "clip_embeddings": "/api/embeddings/clip-generate",

            # Chat APIs
            "chat_completions": "/api/chat/completions",
            "contextual_response": "/api/chat/contextual",

            # Anthropic Claude APIs
            "anthropic_image_validation": "/api/v1/anthropic/validate-image",
            "anthropic_batch_validation": "/api/v1/anthropic/validate-images-batch",
            "anthropic_product_enrichment": "/api/v1/anthropic/enrich-product",
            "anthropic_batch_enrichment": "/api/v1/anthropic/enrich-products-batch",

            # Products APIs
            "products_create": "/api/products/create",
            "products_status": "/api/products/status",

            # Metadata APIs (RAG System)
            "metadata_detect_scope": "/api/rag/metadata/detect-scope",
            "metadata_apply": "/api/rag/metadata/apply-to-products",
            "metadata_list": "/api/rag/metadata/list",
            "metadata_statistics": "/api/rag/metadata/statistics",
        },
        "api_info": {
            "total_endpoints": 110,
            "authentication": "JWT Bearer Token Required",
            "embedding_model": "text-embedding-3-small (1536 dimensions)",
            "recent_enhancements": "Metadata Management System (November 2025)",
            "performance_improvements": "Implicit catalog-general metadata detection, override logic"
        },
        "features": [
            "PDF to Markdown conversion",
            "Table extraction",
            "Image extraction",
            "LlamaIndex RAG integration",
            "Document upload and processing",
            "Semantic search and retrieval",
            "Conversational Q&A",
            "Supabase vector storage"
        ]
    }

# Include API routes
from app.api.search import router as search_router
from app.api.images import router as images_router
from app.api.admin import router as admin_router
from app.api.rag_routes import router as rag_router
from app.api.together_ai_routes import router as together_ai_router
from app.api.anthropic_routes import router as anthropic_router
from app.api.products import router as products_router
from app.api.document_entities import router as document_entities_router
from app.api.embeddings import router as embeddings_router
from app.api.monitoring_routes import router as monitoring_router
from app.api.ai_metrics_routes import router as ai_metrics_router
from app.api.ai_services_routes import router as ai_services_router
from app.api.admin_prompts import router as admin_prompts_router, config_router as extraction_config_router
from app.api.metadata import router as metadata_router
from app.api.saved_searches_routes import router as saved_searches_router
from app.api.duplicate_detection_routes import router as duplicate_detection_router
from app.api.suggestions import router as suggestions_router

app.include_router(search_router)
app.include_router(images_router)
app.include_router(admin_router)
app.include_router(rag_router)
app.include_router(together_ai_router)
app.include_router(anthropic_router)
app.include_router(products_router)
app.include_router(document_entities_router)  # NEW: Document entities (certificates, logos, specifications)
app.include_router(embeddings_router)
app.include_router(monitoring_router)
app.include_router(ai_metrics_router)
app.include_router(ai_services_router)
app.include_router(admin_prompts_router)  # Admin prompts management
app.include_router(extraction_config_router)  # Extraction configuration
app.include_router(metadata_router)  # NEW: Metadata management (scope detection, application, listing)
app.include_router(saved_searches_router)  # NEW: Saved searches with AI deduplication
app.include_router(duplicate_detection_router)  # NEW: Duplicate detection and product merging (same factory only)
app.include_router(suggestions_router)  # NEW: Search suggestions, auto-complete, trending, typo correction


# ============================================================================
# SENTRY DEBUG ENDPOINT
# ============================================================================
@app.get("/sentry-debug", tags=["Monitoring"])
async def trigger_sentry_error():
    """
    Debug endpoint to test Sentry error tracking integration.

    This endpoint intentionally triggers a division by zero error to verify
    that Sentry is properly capturing and reporting errors from the application.

    **WARNING:** This endpoint should only be used for testing purposes.

    Returns:
        Never returns - always raises ZeroDivisionError

    Raises:
        ZeroDivisionError: Intentional error for Sentry testing
    """
    division_by_zero = 1 / 0
    return {"status": "This should never be reached"}


# Customize OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    from fastapi.openapi.utils import get_openapi

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        servers=app.servers,
        tags=app.openapi_tags
    )

    # Add custom security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "JWT Bearer token authentication. All endpoints require a valid JWT token."
        }
    }

    # Add global security requirement
    openapi_schema["security"] = [{"BearerAuth": []}]

    # Add custom info (UPDATED - API Consolidation + Metadata Management)
    openapi_schema["info"]["x-api-features"] = {
        "api_consolidation": "Consolidated and organized endpoints with clear categorization",
        "consolidated_upload": "/api/rag/documents/upload with modes (quick/standard/deep) + categories",
        "consolidated_search": "/api/rag/search?strategy={strategy} with 6 strategies",
        "consolidated_health": "/health for all services (database, storage, AI models)",
        "pdf_processing": "14-stage AI pipeline with checkpoint recovery",
        "product_discovery": "Products + Metadata extraction (inseparable) - Stage 0A",
        "document_entities": "Certificates, logos, specifications as separate knowledge base - Stage 0B",
        "metadata_management": "Dynamic metadata extraction with scope detection and override logic - Stage 4",
        "prompt_enhancement": "Admin templates + agent prompt enhancement (PRESERVED)",
        "category_extraction": "Products, certificates, logos, specifications (PRESERVED)",
        "rag_system": "Retrieval-Augmented Generation with multi-vector search",
        "vector_search": "6 embedding types (text, visual, color, texture, application, multimodal)",
        "search_strategies": "6 strategies (semantic, vector, multi_vector, hybrid, material, image)",
        "ai_models": "12 models: Claude Sonnet 4.5, Haiku 4.5, GPT-5, Llama 4 Scout 17B Vision, CLIP",
        "material_recognition": "Llama 4 Scout 17B Vision (69.4% MMMU, #1 OCR)",
        "embedding_models": "OpenAI text-embedding-3-small (1536D), CLIP ViT-B/32 (512D)",
        "performance": "95%+ product detection, 85%+ search accuracy, 200-800ms response time",
        "scalability": "5,000+ users, 99.5%+ uptime",
        "agentic_queries": "Factory/group filtering for certificates, logos, specifications"
    }

    # Add custom paths info (UPDATED - Legacy endpoints removed)
    openapi_schema["info"]["x-endpoint-categories"] = {
        "rag_routes": "/api/rag/* (23 endpoints) - Document upload, search, query, chat, embeddings, jobs",
        "utilities_routes": "/api/bulk/*, /api/data/*, /api/monitoring/*, /api/system/* (12 endpoints)",
        "admin_routes": "/admin/* (10 endpoints) - Chunk quality, extraction config, prompts management",
        "ai_services_routes": "/api/v1/ai-services/* (10 endpoints) - Classification, validation, boundary detection",
        "search_routes": "/api/search/* (8 endpoints) - Semantic, image, material, multimodal search",
        "jobs_routes": "/api/jobs/* (7 endpoints) - Job progress, statistics, status tracking",
        "document_entities_routes": "/api/document-entities/* (5 endpoints) - Certificates, logos, specifications",
        "images_routes": "/api/images/* (5 endpoints) - Image analysis, search, upload",
        "monitoring_routes": "/, /health, /metrics, /performance/summary (4 endpoints)",
        "embeddings_routes": "/api/embeddings/* (4 endpoints) - CLIP text/image, material embeddings",
        "ai_analysis_routes": "/api/semantic-analysis, /api/analyze/* (4 endpoints) - TogetherAI, multimodal analysis",
        "anthropic_routes": "/api/v1/anthropic/* (3 endpoints) - Claude image validation, product enrichment",
        "products_routes": "/api/products/* (3 endpoints) - Product creation from chunks/layout",
        "ai_metrics_routes": "/api/v1/ai-metrics/* (2 endpoints) - Job metrics, summary"
    }

    # Add platform statistics (UPDATED - Legacy endpoints removed)
    openapi_schema["info"]["x-platform-stats"] = {
        "total_endpoints": 106,
        "endpoint_categories": 15,
        "ai_models": 12,
        "processing_stages": 14,
        "embedding_types": 6,
        "search_strategies": 6,
        "users": "5,000+",
        "uptime": "99.5%+",
        "version": "2.2.0",
        "last_updated": "2025-11-02",
        "legacy_removed": "Removed 18 duplicate /api/documents/* endpoints (13 from documents.py + 5 from search.py)"
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi


def main():
    """
    Main entry point for running the application with uvicorn.
    
    This function is used when running the application directly or via the
    console script defined in pyproject.toml.
    """
    import uvicorn
    
    settings = get_settings()
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        access_log=True
    )


if __name__ == "__main__":
    main()