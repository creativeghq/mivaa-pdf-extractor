"""
Lazy Loading Service for AI Components

This module provides lazy loading functionality for heavy AI components
to reduce memory usage during startup and pipeline execution.

Components are loaded only when needed and can be unloaded after use.
"""

import logging
import gc
import inspect
from typing import Any, Optional, Callable, Dict
from functools import wraps
from datetime import datetime

logger = logging.getLogger(__name__)


class LazyComponent:
    """Wrapper for lazy-loaded components."""
    
    def __init__(self, name: str, loader_func: Callable, cleanup_func: Optional[Callable] = None):
        """
        Initialize lazy component.
        
        Args:
            name: Component name
            loader_func: Function to load the component
            cleanup_func: Optional function to cleanup the component
        """
        self.name = name
        self.loader_func = loader_func
        self.cleanup_func = cleanup_func
        self._instance = None
        self._loaded = False
        self._load_time = None
        
    async def load(self):
        """Load the component if not already loaded."""
        if self._loaded and self._instance is not None:
            logger.debug(f"✅ Component '{self.name}' already loaded")
            return self._instance

        try:
            logger.info(f"📦 Loading component: {self.name}...")
            self._load_time = datetime.utcnow()
            # Check if loader_func is an async function (coroutine function)
            if inspect.iscoroutinefunction(self.loader_func):
                self._instance = await self.loader_func()
            else:
                self._instance = self.loader_func()
            self._loaded = True
            logger.info(f"✅ Component '{self.name}' loaded successfully")
            return self._instance
        except Exception as e:
            logger.error(f"❌ Failed to load component '{self.name}': {e}")
            self._loaded = False
            self._instance = None
            raise
    
    async def unload(self):
        """Unload the component and free memory."""
        if not self._loaded or self._instance is None:
            return
        
        try:
            logger.info(f"🧹 Unloading component: {self.name}...")
            
            if self.cleanup_func:
                cleanup_result = self.cleanup_func(self._instance)
                if hasattr(cleanup_result, '__await__'):
                    await cleanup_result
            
            self._instance = None
            self._loaded = False
            
            # Force garbage collection
            gc.collect()
            logger.info(f"✅ Component '{self.name}' unloaded and memory freed")
        except Exception as e:
            logger.error(f"❌ Failed to unload component '{self.name}': {e}")
    
    async def get(self):
        """Get the component, loading if necessary."""
        if not self._loaded:
            await self.load()
        return self._instance
    
    def is_loaded(self) -> bool:
        """Check if component is loaded."""
        return self._loaded and self._instance is not None


class LazyComponentManager:
    """Manages lazy-loaded components."""
    
    def __init__(self):
        """Initialize component manager."""
        self.components: Dict[str, LazyComponent] = {}
        logger.info("✅ Lazy Component Manager initialized")
    
    def register(self, name: str, loader_func: Callable, cleanup_func: Optional[Callable] = None):
        """Register a lazy component."""
        self.components[name] = LazyComponent(name, loader_func, cleanup_func)
        logger.info(f"📝 Registered lazy component: {name}")
    
    async def load(self, name: str):
        """Load a specific component."""
        if name not in self.components:
            raise ValueError(f"Component '{name}' not registered")
        return await self.components[name].load()
    
    async def unload(self, name: str):
        """Unload a specific component."""
        if name not in self.components:
            raise ValueError(f"Component '{name}' not registered")
        await self.components[name].unload()
    
    async def get(self, name: str):
        """Get a component, loading if necessary."""
        if name not in self.components:
            raise ValueError(f"Component '{name}' not registered")
        return await self.components[name].get()
    
    async def unload_all(self):
        """Unload all components."""
        logger.info("🧹 Unloading all components...")
        for name, component in self.components.items():
            if component.is_loaded():
                await component.unload()
        logger.info("✅ All components unloaded")
    
    def get_status(self) -> Dict[str, bool]:
        """Get status of all components."""
        return {name: component.is_loaded() for name, component in self.components.items()}


# Global component manager
_component_manager = LazyComponentManager()


def get_component_manager() -> LazyComponentManager:
    """Get the global component manager."""
    return _component_manager


def lazy_load_component(name: str, cleanup_func: Optional[Callable] = None):
    """
    Decorator to make a component loader function lazy.
    
    Usage:
        @lazy_load_component("my_component")
        def load_my_component():
            return MyComponent()
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            component = await _component_manager.get(name)
            if component is None:
                component = await _component_manager.load(name)
            return component
        
        # Register the component
        _component_manager.register(name, func, cleanup_func)
        return wrapper
    
    return decorator


async def load_component_for_stage(stage_name: str, components_needed: list):
    """
    Load components needed for a specific pipeline stage.
    
    Args:
        stage_name: Name of the pipeline stage
        components_needed: List of component names to load
    """
    logger.info(f"📦 Loading components for stage: {stage_name}")
    loaded = []
    
    for component_name in components_needed:
        try:
            await _component_manager.load(component_name)
            loaded.append(component_name)
        except Exception as e:
            logger.error(f"❌ Failed to load {component_name}: {e}")
    
    logger.info(f"✅ Loaded {len(loaded)}/{len(components_needed)} components for {stage_name}")
    return loaded


async def unload_components_after_stage(stage_name: str, components_to_unload: list):
    """
    Unload components after a pipeline stage completes.
    
    Args:
        stage_name: Name of the pipeline stage
        components_to_unload: List of component names to unload
    """
    logger.info(f"🧹 Unloading components after stage: {stage_name}")
    unloaded = []
    
    for component_name in components_to_unload:
        try:
            await _component_manager.unload(component_name)
            unloaded.append(component_name)
        except Exception as e:
            logger.error(f"❌ Failed to unload {component_name}: {e}")
    
    logger.info(f"✅ Unloaded {len(unloaded)}/{len(components_to_unload)} components after {stage_name}")
    return unloaded

