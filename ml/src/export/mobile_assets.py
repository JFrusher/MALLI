"""Mobile app asset management and automatic synchronization.

Handles copying exported models to mobile app asset directories
and updating asset configurations.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class MobileAssetManager:
    """Manages model files in mobile app asset directories."""
    
    # Asset directory structure
    ASSET_DIRS = {
        "flutter": {
            "android": "assets/models/android",
            "ios": "assets/models/ios",
            "shared": "assets/models",
        },
        "swiftui": {
            "ios": "assets/models/ios",
        },
        "react-native": {
            "android": "assets/models/android",
            "ios": "assets/models/ios",
        },
    }
    
    def __init__(
        self,
        mobile_root: Path,
        app_framework: str = "flutter",
    ):
        """Initialize mobile asset manager.
        
        Args:
            mobile_root: Root directory of mobile app (e.g., mobile/)
            app_framework: Mobile framework ("flutter", "react-native", "swiftui")
        """
        self.mobile_root = Path(mobile_root)
        self.app_framework = app_framework
        
        if app_framework not in self.ASSET_DIRS:
            raise ValueError(
                f"Unknown app framework: {app_framework}. "
                f"Supported: {list(self.ASSET_DIRS.keys())}"
            )
    
    def get_asset_path(
        self,
        platform: str = "shared",
    ) -> Path:
        """Get asset directory for platform.
        
        Args:
            platform: Target platform ("shared", "android", "ios")
            
        Returns:
            Full path to asset directory
        """
        rel_path = self.ASSET_DIRS[self.app_framework].get(platform, "assets/models")
        return self.mobile_root / rel_path
    
    def sync_model(
        self,
        source_model_path: Path,
        platform: str = "shared",
        model_name: str = "malaria_detector.tflite",
    ) -> bool:
        """Sync model to mobile app assets.
        
        Args:
            source_model_path: Path to exported model file
            platform: Target platform ("shared", "android", "ios")
            model_name: Name to save as in assets
            
        Returns:
            True if successful, False otherwise
        """
        if not source_model_path.exists():
            logger.error("Source model not found: %s", source_model_path)
            return False
        
        asset_dir = self.get_asset_path(platform)
        asset_dir.mkdir(parents=True, exist_ok=True)
        
        target_path = asset_dir / model_name
        
        try:
            logger.info("Syncing model: %s -> %s", source_model_path, target_path)
            shutil.copy2(source_model_path, target_path)
            logger.info("✓ Model synced successfully")
            return True
        except Exception as e:
            logger.error("✗ Failed to sync model: %s", e)
            return False
    
    def sync_models_by_format(
        self,
        export_dir: Path,
        model_formats: dict[str, Path],
    ) -> dict[str, bool]:
        """Sync models to appropriate platforms based on format.
        
        Args:
            export_dir: Root export directory
            model_formats: Dict mapping format -> model path
                          e.g., {"tflite": Path(...), "coreml": Path(...)}
        
        Returns:
            Dict mapping format -> success boolean
        """
        results = {}
        
        for fmt, model_path in model_formats.items():
            if fmt == "tflite":
                # Copy to shared and android
                results[f"{fmt}_shared"] = self.sync_model(model_path, platform="shared")
                results[f"{fmt}_android"] = self.sync_model(model_path, platform="android")
            elif fmt == "coreml":
                # Copy to iOS
                results[f"{fmt}_ios"] = self.sync_model(model_path, platform="ios")
            elif fmt == "onnx":
                # ONNX can go to shared for universal access
                results[f"{fmt}_shared"] = self.sync_model(model_path, platform="shared")
        
        return results
    
    def update_pubspec_assets(
        self,
        pubspec_path: Path,
        model_name: str = "malaria_detector.tflite",
    ) -> bool:
        """Update Flutter pubspec.yaml with asset paths (Flutter only).
        
        Args:
            pubspec_path: Path to pubspec.yaml
            model_name: Name of model file
            
        Returns:
            True if successful
        """
        if self.app_framework != "flutter":
            logger.debug("Skipping pubspec update (not Flutter)")
            return True
        
        if not pubspec_path.exists():
            logger.warning("pubspec.yaml not found: %s", pubspec_path)
            return False
        
        try:
            content = pubspec_path.read_text(encoding="utf-8")
            
            # Check if assets section exists
            if "flutter:" not in content:
                logger.warning("No flutter: section found in pubspec.yaml")
                return False
            
            # Add asset declarations if not present
            model_asset = f"assets/models/{model_name}"
            
            if model_asset not in content:
                # Simple update (in production, use YAML library)
                if "assets:" in content:
                    logger.info("Asset section exists, manually verify: %s", model_asset)
                else:
                    logger.info("Consider adding asset: %s", model_asset)
            else:
                logger.info("✓ Asset already declared in pubspec.yaml")
            
            return True
        except Exception as e:
            logger.error("Failed to update pubspec.yaml: %s", e)
            return False
    
    def generate_asset_manifest(
        self,
        output_path: Path,
        models: dict[str, Path],
    ) -> bool:
        """Generate manifest of synced assets for app reference.
        
        Args:
            output_path: Where to save manifest JSON
            models: Dict mapping format -> model path
        
        Returns:
            True if successful
        """
        manifest = {
            "framework": self.app_framework,
            "generated_at": str(Path(__file__).parent),
            "models": {},
        }
        
        try:
            for fmt, path in models.items():
                if path.exists():
                    file_size = path.stat().st_size
                    
                    # Compute checksum
                    checksum = self._compute_checksum(path)
                    
                    manifest["models"][fmt] = {
                        "path": str(path),
                        "size_bytes": file_size,
                        "checksum_sha256": checksum,
                    }
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(manifest, indent=2),
                encoding="utf-8"
            )
            logger.info("✓ Asset manifest saved to %s", output_path)
            return True
        except Exception as e:
            logger.error("Failed to generate asset manifest: %s", e)
            return False
    
    @staticmethod
    def _compute_checksum(file_path: Path, algorithm: str = "sha256") -> str:
        """Compute file checksum.
        
        Args:
            file_path: Path to file
            algorithm: Hash algorithm ("sha256", "md5")
            
        Returns:
            Hex digest of file
        """
        hasher = hashlib.new(algorithm)
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def verify_asset_sync(
        self,
        source_path: Path,
        asset_path: Path,
    ) -> bool:
        """Verify that asset was synced correctly via checksum.
        
        Args:
            source_path: Original model file
            asset_path: Synced asset file
            
        Returns:
            True if checksums match
        """
        if not asset_path.exists():
            logger.error("Asset not found: %s", asset_path)
            return False
        
        source_checksum = self._compute_checksum(source_path)
        asset_checksum = self._compute_checksum(asset_path)
        
        match = source_checksum == asset_checksum
        status = "✓" if match else "✗"
        logger.info(
            "%s Checksum verification: source=%s, asset=%s",
            status,
            source_checksum[:8],
            asset_checksum[:8],
        )
        
        return match


class MobileAssetSyncPipeline:
    """Complete pipeline for exporting and syncing models to mobile apps."""
    
    def __init__(
        self,
        mobile_root: Path,
        export_root: Path,
        app_framework: str = "flutter",
    ):
        """Initialize sync pipeline.
        
        Args:
            mobile_root: Root of mobile app directory
            export_root: Root of ML export directory
            app_framework: Mobile framework type
        """
        self.asset_manager = MobileAssetManager(mobile_root, app_framework)
        self.export_root = Path(export_root)
        self.mobile_root = Path(mobile_root)
        self.app_framework = app_framework
    
    def sync_exports(
        self,
        models: dict[str, Path],
        verify: bool = True,
    ) -> dict[str, dict]:
        """Sync all exported models to mobile app.
        
        Args:
            models: Dict mapping format -> exported model path
            verify: Whether to verify synced files via checksum
            
        Returns:
            Dict with sync results and verification status
        """
        logger.info("Starting mobile asset sync pipeline...")
        results = {
            "synced": {},
            "verified": {},
            "manifest": None,
        }
        
        # Sync by format
        for fmt, source_path in models.items():
            if fmt == "tflite":
                platforms = ["shared", "android"]
            elif fmt == "coreml":
                platforms = ["ios"]
            else:
                platforms = ["shared"]
            
            for platform in platforms:
                asset_path = self.asset_manager.get_asset_path(platform) / f"malaria_detector.{fmt}"
                success = self.asset_manager.sync_model(source_path, platform=platform)
                results["synced"][f"{fmt}_{platform}"] = success
                
                if verify and success:
                    verified = self.asset_manager.verify_asset_sync(source_path, asset_path)
                    results["verified"][f"{fmt}_{platform}"] = verified
        
        # Generate manifest
        manifest_path = self.mobile_root / "asset_manifest.json"
        manifest_success = self.asset_manager.generate_asset_manifest(manifest_path, models)
        results["manifest"] = {
            "path": str(manifest_path),
            "success": manifest_success,
        }
        
        logger.info("Asset sync complete: %d synced, %d verified", 
                   sum(results["synced"].values()),
                   sum(results["verified"].values()))
        
        return results
