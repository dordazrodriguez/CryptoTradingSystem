"""Persistent storage for RL experiences to enable long-term memory."""

import pickle
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime


class PersistentExperienceStorage:
    """Storage for experiences that persists across sessions."""
    
    def __init__(self, storage_dir: Optional[Path] = None):
        """
        Initialize persistent experience storage.
        
        Args:
            storage_dir: Directory to store experiences. Defaults to 'data/experiences'
        """
        if storage_dir is None:
            storage_dir = Path("data/experiences")
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata file to track stored experiences
        self.metadata_file = self.storage_dir / "metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata about stored experiences."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {"files": [], "total_experiences": 0}
        return {"files": [], "total_experiences": 0}
    
    def _save_metadata(self):
        """Save metadata about stored experiences."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save metadata: {e}")
    
    def save_experiences(
        self,
        experiences: List[Dict[str, Any]],
        tag: Optional[str] = None,
        max_experiences_per_file: int = 10000
    ) -> str:
        """
        Save experiences to disk.
        
        Args:
            experiences: List of experience dictionaries
            tag: Optional tag to identify this batch (e.g., "2025-11-02")
            max_experiences_per_file: Maximum experiences per file
            
        Returns:
            Path to saved file
        """
        if not experiences:
            return ""
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag_str = f"_{tag}" if tag else ""
        filename = f"experiences{tag_str}_{timestamp}.pkl"
        filepath = self.storage_dir / filename
        
        # Save experiences with original numpy arrays
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(experiences, f)
            
            # Update metadata
            file_info = {
                "filename": filename,
                "timestamp": timestamp,
                "tag": tag,
                "num_experiences": len(experiences),
                "filepath": str(filepath)
            }
            self.metadata["files"].append(file_info)
            self.metadata["total_experiences"] += len(experiences)
            
            # Keep only recent metadata (last 100 files)
            if len(self.metadata["files"]) > 100:
                # Remove oldest files from metadata (but keep files on disk)
                self.metadata["files"] = self.metadata["files"][-100:]
            
            self._save_metadata()
            
            return str(filepath)
        except Exception as e:
            print(f"Error saving experiences: {e}")
            return ""
    
    def load_experiences(
        self,
        filepath: Optional[str] = None,
        num_files: Optional[int] = None,
        tag: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Load experiences from disk.
        
        Args:
            filepath: Specific file to load. If None, loads most recent files
            num_files: Number of most recent files to load (if filepath is None)
            tag: Filter by tag (if filepath is None)
            
        Returns:
            List of experiences
        """
        if filepath:
            # Load specific file
            try:
                with open(filepath, 'rb') as f:
                    experiences = pickle.load(f)
                return experiences
            except Exception as e:
                print(f"Error loading experiences from {filepath}: {e}")
                return []
        
        # Load from metadata
        files = self.metadata.get("files", [])
        
        # Filter by tag if specified
        if tag:
            files = [f for f in files if f.get("tag") == tag]
        
        # Sort by timestamp (most recent first)
        files.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        # Limit number of files
        if num_files:
            files = files[:num_files]
        
        # Load all experiences
        all_experiences = []
        for file_info in files:
            filepath = file_info.get("filepath")
            if filepath and Path(filepath).exists():
                try:
                    with open(filepath, 'rb') as f:
                        experiences = pickle.load(f)
                        all_experiences.extend(experiences)
                except Exception as e:
                    print(f"Warning: Failed to load {filepath}: {e}")
        
        return all_experiences
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get statistics about stored experiences."""
        files = self.metadata.get("files", [])
        total_files = len(files)
        total_experiences = self.metadata.get("total_experiences", 0)
        
        # Calculate actual disk usage
        total_size = 0
        for file_info in files:
            filepath = file_info.get("filepath")
            if filepath and Path(filepath).exists():
                total_size += Path(filepath).stat().st_size
        
        return {
            "total_files": total_files,
            "total_experiences": total_experiences,
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "files": files[-10:] if files else []  # Last 10 files
        }

