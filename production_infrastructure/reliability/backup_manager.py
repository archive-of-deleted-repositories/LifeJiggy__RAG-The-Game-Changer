"""
Backup Manager - RAG-The-Game-Changer

Backup and disaster recovery management for RAG systems.
"""

import asyncio
import gzip
import hashlib
import json
import logging
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class BackupType(Enum):
    """Types of backups."""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"


class BackupStorage(Enum):
    """Storage backends for backups."""
    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"
    AZURE = "azure"


@dataclass
class BackupConfig:
    """Configuration for backup system."""
    backup_dir: str = "./backups"
    retention_days: int = 30
    max_backups: int = 30
    backup_type: BackupType = BackupType.FULL
    storage_backend: BackupStorage = BackupStorage.LOCAL
    compression: bool = True
    encryption_enabled: bool = True
    backup_schedule: str = "0 2 * * *"  # Daily at 2 AM
    s3_config: Dict[str, Any] = field(default_factory=dict)
    gcs_config: Dict[str, Any] = field(default_factory=dict)
    azure_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BackupMetadata:
    """Metadata for a backup."""
    backup_id: str
    timestamp: str
    backup_type: BackupType
    size_bytes: int
    compressed: bool
    encrypted: bool
    checksum: str
    files_count: int
    components: List[str] = field(default_factory=list)
    storage_location: str


@dataclass
class BackupResult:
    """Result from backup operation."""
    success: bool
    backup_id: Optional[str]
    size_bytes: int
    duration_seconds: float
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class BackupManager:
    """Manages backups and disaster recovery for RAG systems."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.backup_config = BackupConfig(**self.config)
        
        # Ensure backup directory exists
        self.backup_dir = Path(self.backup_config.backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        self.backups: List[BackupMetadata] = []
        self._load_backup_manifest()
        
        self._lock = asyncio.Lock()
        
    def _load_backup_manifest(self):
        """Load backup manifest from disk."""
        manifest_path = self.backup_dir / "manifest.json"
        
        if manifest_path.exists():
            try:
                with open(manifest_path, 'r') as f:
                    data = json.load(f)
                    self.backups = [BackupMetadata(**item) for item in data.get("backups", [])]
                logger.info(f"Loaded {len(self.backups)} backup entries from manifest")
            except Exception as e:
                logger.error(f"Error loading backup manifest: {e}")
        
    async def create_backup(
        self,
        components: Optional[List[str]] = None,
        backup_type: Optional[BackupType] = None
        description: str = ""
    ) -> BackupResult:
        """Create a backup of specified components."""
        start_time = asyncio.get_event_loop().time()
        
        backup_id = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_type = backup_type or self.backup_config.backup_type
        
        logger.info(f"Creating backup {backup_id} of type {backup_type.value}")
        
        try:
            # Determine what to backup
            backup_dirs = self._get_backup_directories(components)
            
            # Create backup
            backup_path = self.backup_dir / backup_id
            backup_path.mkdir(exist_ok=True)
            
            total_size = 0
            files_count = 0
            
            for source_dir in backup_dirs:
                if source_dir.exists():
                    backup_content_path = backup_path / source_dir.name
                    
                    # Copy directory
                    if source_dir.is_dir():
                        shutil.copytree(source_dir, backup_content_path)
                        total_size += sum(f.stat().st_size for f in source_dir.rglob('*') if f.is_file())
                        files_count += len(list(source_dir.rglob('*')))
                    else:
                        shutil.copy2(source_dir, backup_content_path)
                        total_size += source_dir.stat().st_size
                        files_count += 1
            
            # Calculate checksum
            checksum = self._calculate_backup_checksum(backup_path)
            
            # Compress if enabled
            if self.backup_config.compression:
                compressed_path = backup_path.with_suffix('.tar.gz')
                self._compress_backup(backup_path, compressed_path)
                total_size = compressed_path.stat().st_size
                backup_path.unlink()
                backup_path = compressed_path
            
            # Encrypt if enabled
            if self.backup_config.encryption_enabled:
                encrypted_path = backup_path.with_suffix('.enc')
                await self._encrypt_backup(backup_path, encrypted_path)
                total_size = encrypted_path.stat().st_size
                backup_path.unlink()
                backup_path = encrypted_path
            
            # Create metadata
            metadata = BackupMetadata(
                backup_id=backup_id,
                timestamp=datetime.now().isoformat(),
                backup_type=backup_type,
                size_bytes=total_size,
                compressed=self.backup_config.compression,
                encrypted=self.backup_config.encryption_enabled,
                checksum=checksum,
                files_count=files_count,
                components=components or ["all"],
                storage_location=str(backup_path)
            )
            
            # Save to storage backend
            if self.backup_config.storage_backend != BackupStorage.LOCAL:
                await self._upload_to_cloud(backup_path, backup_id)
            
            # Update manifest
            await self._update_manifest(metadata)
            
            # Cleanup old backups
            await self._cleanup_old_backups()
            
            duration = asyncio.get_event_loop().time() - start_time
            
            logger.info(f"Backup {backup_id} completed in {duration:.2f}s ({total_size / 1024 / 1024:.2f} MB)")
            
            return BackupResult(
                success=True,
                backup_id=backup_id,
                size_bytes=total_size,
                duration_seconds=duration,
                message=f"Backup {backup_id} created successfully",
                metadata={"description": description}
            )
            
        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            return BackupResult(
                success=False,
                backup_id=None,
                size_bytes=0,
                duration_seconds=0,
                message=f"Backup failed: {str(e)}",
                metadata={"error": str(e)}
            )
    
    def _get_backup_directories(self, components: Optional[List[str]]) -> List[Path]:
        """Get directories to backup based on components."""
        if components:
            # Backup specific components
            return [Path(c) for c in components]
        else:
            # Backup everything except backups directory
            return [d for d in Path('.').iterdir() 
                    if d.is_dir() and d.name != 'backups' and not d.name.startswith('.')]
    
    def _calculate_backup_checksum(self, backup_path: Path) -> str:
        """Calculate checksum for backup."""
        checksum_data = []
        
        for file_path in backup_path.rglob('*'):
            if file_path.is_file():
                with open(file_path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                    checksum_data.append(file_hash)
        
        combined = "".join(sorted(checksum_data))
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    def _compress_backup(self, source_path: Path, output_path: Path):
        """Compress backup directory."""
        logger.info(f"Compressing {source_path} to {output_path}")
        
        with tarfile.open(output_path, "w:gz") as tar:
            tar.add(source_path, arcname=source_path.name)
    
    async def _encrypt_backup(self, source_path: Path, output_path: Path):
        """Encrypt backup file."""
        logger.info(f"Encrypting {source_path} to {output_path}")
        
        # Simple encryption for demo (use proper encryption in production)
        key = b'your-secret-key-32-bytes-long'
        
        import subprocess
        subprocess.run([
            'openssl', 'enc',
            '-aes-256-cbc',
            '-salt',
            '-in', str(source_path),
            '-out', str(output_path),
            '-k', key.hex()
        ], check=True)
    
    async def _upload_to_cloud(self, backup_path: Path, backup_id: str):
        """Upload backup to cloud storage."""
        logger.info(f"Uploading {backup_id} to {self.backup_config.storage_backend.value}")
        
        # Simulate upload delay
        await asyncio.sleep(2)
        
        # In production, this would use AWS S3, GCP GCS, or Azure SDK
        if self.backup_config.storage_backend == BackupStorage.S3:
            await self._upload_to_s3(backup_path, backup_id)
        elif self.backup_config.storage_backend == BackupStorage.GCS:
            await self._upload_to_gcs(backup_path, backup_id)
        elif self.backup_config.storage_backend == BackupStorage.AZURE:
            await self._upload_to_azure(backup_path, backup_id)
    
    async def _upload_to_s3(self, backup_path: Path, backup_id: str):
        """Upload backup to AWS S3."""
        # Placeholder for S3 upload
        pass
    
    async def _upload_to_gcs(self, backup_path: Path, backup_id: str):
        """Upload backup to Google Cloud Storage."""
        # Placeholder for GCS upload
        pass
    
    async def _upload_to_azure(self, backup_path: Path, backup_id: str):
        """Upload backup to Azure Blob Storage."""
        # Placeholder for Azure upload
        pass
    
    async def _update_manifest(self, metadata: BackupMetadata):
        """Update backup manifest file."""
        self.backups.append(metadata)
        
        manifest_data = {
            "backups": [
                {
                    "backup_id": b.backup_id,
                    "timestamp": b.timestamp,
                    "backup_type": b.backup_type.value,
                    "size_bytes": b.size_bytes,
                    "compressed": b.compressed,
                    "encrypted": b.encrypted,
                    "checksum": b.checksum,
                    "files_count": b.files_count,
                    "components": b.components,
                    "storage_location": b.storage_location
                }
                for b in self.backups
            ]
        }
        
        manifest_path = self.backup_dir / "manifest.json"
        
        with open(manifest_path, 'w') as f:
            json.dump(manifest_data, f, indent=2)
        
        logger.debug(f"Updated backup manifest with {metadata.backup_id}")
    
    async def _cleanup_old_backups(self):
        """Remove backups older than retention period."""
        if self.backup_config.max_backups <= 0:
            return
        
        cutoff_date = datetime.now() - timedelta(days=self.backup_config.retention_days)
        
        # Sort backups by timestamp, oldest first
        sorted_backups = sorted(self.backups, key=lambda b: b.timestamp)
        
        # Keep only max_backups most recent
        to_remove = sorted_backups[:-self.backup_config.max_backups]
        
        for backup in to_remove:
            backup_date = datetime.fromisoformat(backup.timestamp)
            
            if backup_date < cutoff_date:
                backup_path = Path(backup.storage_location)
                
                if backup_path.exists():
                    # Remove from cloud if not local
                    if self.backup_config.storage_backend == BackupStorage.LOCAL:
                        if backup_path.is_file():
                            backup_path.unlink()
                        elif backup_path.is_dir():
                            shutil.rmtree(backup_path)
                        
                        logger.info(f"Removed old backup {backup.backup_id}")
    
    async def restore_backup(
        self,
        backup_id: str,
        target_dir: Optional[str] = None
    ) -> BackupResult:
        """Restore from a backup."""
        logger.info(f"Restoring backup {backup_id}")
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Find backup metadata
            backup_metadata = next((b for b in self.backups if b.backup_id == backup_id), None)
            
            if not backup_metadata:
                raise ValueError(f"Backup {backup_id} not found")
            
            # Prepare restore location
            restore_dir = Path(target_dir) if target_dir else Path('.')
            
            # Decrypt if encrypted
            backup_path = Path(backup_metadata.storage_location)
            
            if backup_metadata.encrypted:
                decrypted_path = backup_path.with_suffix('.dec')
                await self._decrypt_backup(backup_path, decrypted_path)
                backup_path = decrypted_path
            
            # Decompress if compressed
            if backup_metadata.compressed:
                self._decompress_backup(backup_path, restore_dir)
            else:
                shutil.copytree(backup_path, restore_dir / backup_id, dirs_exist_ok=True)
            
            duration = asyncio.get_event_loop().time() - start_time
            
            logger.info(f"Restore {backup_id} completed in {duration:.2f}s")
            
            return BackupResult(
                success=True,
                backup_id=backup_id,
                size_bytes=backup_metadata.size_bytes,
                duration_seconds=duration,
                message=f"Restore {backup_id} completed successfully",
                metadata={"restored_to": str(restore_dir)}
            )
            
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return BackupResult(
                success=False,
                backup_id=backup_id,
                size_bytes=0,
                duration_seconds=0,
                message=f"Restore failed: {str(e)}",
                metadata={"error": str(e)}
            )
    
    def _decompress_backup(self, backup_path: Path, target_dir: Path):
        """Decompress backup archive."""
        logger.info(f"Decompressing {backup_path} to {target_dir}")
        
        with tarfile.open(backup_path, "r:gz") as tar:
            tar.extractall(path=target_dir)
    
    async def _decrypt_backup(self, encrypted_path: Path, output_path: Path):
        """Decrypt backup file."""
        logger.info(f"Decrypting {encrypted_path} to {output_path}")
        
        key = b'your-secret-key-32-bytes-long'
        
        import subprocess
        subprocess.run([
            'openssl', 'enc',
            '-d',
            '-aes-256-cbc',
            '-in', str(encrypted_path),
            '-out', str(output_path),
            '-k', key.hex()
        ], check=True)
    
    def list_backups(self, limit: int = 10) -> List[BackupMetadata]:
        """List available backups."""
        sorted_backups = sorted(self.backups, key=lambda b: b.timestamp, reverse=True)
        return sorted_backups[:limit]
    
    def get_backup_info(self, backup_id: str) -> Optional[BackupMetadata]:
        """Get detailed information about a specific backup."""
        return next((b for b in self.backups if b.backup_id == backup_id), None)
    
    async def verify_backup(self, backup_id: str) -> bool:
        """Verify backup integrity."""
        metadata = self.get_backup_info(backup_id)
        
        if not metadata:
            return False
        
        logger.info(f"Verifying backup {backup_id}")
        
        # Check if backup exists
        backup_path = Path(metadata.storage_location)
        
        if not backup_path.exists():
            return False
        
        # Verify checksum
        current_checksum = self._calculate_backup_checksum(backup_path)
        
        is_valid = current_checksum == metadata.checksum
        
        if is_valid:
            logger.info(f"Backup {backup_id} verification passed")
        else:
            logger.warning(f"Backup {backup_id} verification failed: checksum mismatch")
        
        return is_valid
    
    def get_stats(self) -> Dict[str, Any]:
        """Get backup manager statistics."""
        total_size = sum(b.size_bytes for b in self.backups)
        total_files = sum(b.files_count for b in self.backups)
        
        compressed_count = sum(1 for b in self.backups if b.compressed)
        encrypted_count = sum(1 for b in self.backups if b.encrypted)
        
        return {
            "total_backups": len(self.backups),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / 1024 / 1024,
            "total_files": total_files,
            "compressed_count": compressed_count,
            "encrypted_count": encrypted_count,
            "backup_directory": str(self.backup_dir),
            "retention_days": self.backup_config.retention_days,
            "max_backups": self.backup_config.max_backups
        }
