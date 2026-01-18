"""
Auto KB Document Service

Automatically creates knowledge base documents from extracted product metadata.
This service is called after metadata extraction to create KB docs for:
- Packaging information
- Compliance/safety data
- Care/maintenance instructions
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from app.services.core.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)


class AutoKBDocumentService:
    """
    Service to auto-create KB documents from extracted metadata.

    Creates structured KB documents from:
    - packaging metadata
    - compliance/safety metadata
    - care/maintenance instructions
    """

    def __init__(self):
        self.supabase = get_supabase_client()

    async def create_kb_documents_from_metadata(
        self,
        product_id: str,
        product_name: str,
        workspace_id: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create KB docs from packaging/compliance/care metadata.

        Args:
            product_id: Database product ID
            product_name: Product name for document titles
            workspace_id: Workspace ID
            metadata: Extracted metadata dict

        Returns:
            Stats dict with documents_created count and any errors
        """
        stats = {'documents_created': 0, 'errors': []}

        try:
            # 1. Packaging information
            packaging = metadata.get('packaging', {})
            if packaging and any(self._has_value(v) for v in packaging.values()):
                doc_id = await self._create_kb_doc(
                    workspace_id, product_id,
                    f"{product_name} - Packaging",
                    self._format_section("Packaging Information", packaging),
                    'packaging'
                )
                if doc_id:
                    stats['documents_created'] += 1

            # 2. Compliance/safety information
            compliance = metadata.get('compliance', {})
            if compliance and any(self._has_value(v) for v in compliance.values()):
                doc_id = await self._create_kb_doc(
                    workspace_id, product_id,
                    f"{product_name} - Compliance & Safety",
                    self._format_section("Compliance & Safety", compliance),
                    'compliance'
                )
                if doc_id:
                    stats['documents_created'] += 1

            # 3. Care/maintenance instructions (from application metadata)
            application = metadata.get('application', {})
            care_instructions = application.get('care_instructions') or application.get('maintenance')
            if care_instructions:
                care_value = self._extract_value(care_instructions)
                if care_value:
                    doc_id = await self._create_kb_doc(
                        workspace_id, product_id,
                        f"{product_name} - Care Instructions",
                        f"## Care & Maintenance\n\n{care_value}",
                        'care'
                    )
                    if doc_id:
                        stats['documents_created'] += 1

            # 4. Certifications (from compliance or standalone)
            certifications = metadata.get('certifications', []) or compliance.get('certifications', [])
            if certifications:
                certs_value = self._extract_value(certifications)
                if certs_value:
                    cert_list = certs_value if isinstance(certs_value, list) else [certs_value]
                    if cert_list:
                        cert_content = "## Certifications\n\n" + "\n".join(f"- {c}" for c in cert_list)
                        doc_id = await self._create_kb_doc(
                            workspace_id, product_id,
                            f"{product_name} - Certifications",
                            cert_content,
                            'certification'
                        )
                        if doc_id:
                            stats['documents_created'] += 1

            if stats['documents_created'] > 0:
                logger.info(f"✅ Created {stats['documents_created']} KB docs for {product_name}")
            else:
                logger.debug(f"ℹ️ No KB-eligible metadata found for {product_name}")

        except Exception as e:
            logger.error(f"❌ KB creation failed for {product_name}: {e}")
            stats['errors'].append(str(e))

        return stats

    def _has_value(self, val) -> bool:
        """Check if a value is non-empty."""
        if val is None:
            return False
        if isinstance(val, dict):
            if 'value' in val:
                return bool(val['value'])
            return any(self._has_value(v) for v in val.values())
        if isinstance(val, list):
            return len(val) > 0
        return bool(val)

    def _extract_value(self, val):
        """Extract value from {value, confidence} format or plain value."""
        if isinstance(val, dict) and 'value' in val:
            return val['value']
        return val

    def _format_section(self, title: str, data: Dict) -> str:
        """Format metadata dict as markdown section."""
        lines = [f"## {title}\n"]
        for key, value in data.items():
            extracted = self._extract_value(value)
            if isinstance(extracted, list):
                extracted = ", ".join(str(x) for x in extracted)
            if extracted:
                formatted_key = key.replace('_', ' ').title()
                lines.append(f"- **{formatted_key}**: {extracted}")
        return "\n".join(lines)

    async def _create_kb_doc(
        self,
        workspace_id: str,
        product_id: str,
        title: str,
        content: str,
        category: str
    ) -> Optional[str]:
        """
        Create KB document and link to product.

        Args:
            workspace_id: Workspace ID
            product_id: Product ID to link
            title: Document title
            content: Document content (markdown)
            category: Document category (packaging, compliance, care, certification)

        Returns:
            Document ID if created, None otherwise
        """
        try:
            # Create KB document
            result = self.supabase.client.table("kb_docs").insert({
                "workspace_id": workspace_id,
                "title": title,
                "content": content,
                "status": "published",
                "metadata": {
                    "auto_generated": True,
                    "category": category,
                    "product_id": product_id,
                    "generated_at": datetime.utcnow().isoformat()
                }
            }).execute()

            if result.data:
                doc_id = result.data[0]['id']

                # Link document to product
                self.supabase.client.table("kb_doc_attachments").insert({
                    "workspace_id": workspace_id,
                    "document_id": doc_id,
                    "product_id": product_id,
                    "relationship_type": category
                }).execute()

                logger.info(f"   ✅ KB doc created: {title}")
                return doc_id

            return None

        except Exception as e:
            logger.error(f"   ❌ KB doc creation failed for '{title}': {e}")
            return None
