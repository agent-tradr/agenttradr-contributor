"""
BuildSession model for managing code generation and build sessions.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
from dataclasses import dataclass, field
import uuid


class BuildStatus(Enum):
    """Enum for build session status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BuildType(Enum):
    """Enum for build types."""
    FULL_BUILD = "full_build"
    INCREMENTAL = "incremental"
    HOT_RELOAD = "hot_reload"
    TEST_BUILD = "test_build"
    PRODUCTION = "production"


@dataclass
class BuildMetrics:
    """Metrics for a build session."""
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    files_processed: int = 0
    files_generated: int = 0
    errors_count: int = 0
    warnings_count: int = 0
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None


@dataclass
class BuildArtifact:
    """Represents an artifact produced by a build session."""
    artifact_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    file_path: str = ""
    file_type: str = ""
    size_bytes: int = 0
    checksum: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class BuildSession:
    """
    Model for managing build sessions in the AgentCODR system.
    
    A build session represents a single code generation or build process,
    tracking its status, metrics, and artifacts.
    """
    
    # Core fields
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str = ""
    user_id: str = ""
    build_type: BuildType = BuildType.FULL_BUILD
    status: BuildStatus = BuildStatus.PENDING
    
    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)
    
    # Build details
    source_directory: str = ""
    output_directory: str = ""
    target_platform: Optional[str] = None
    build_command: Optional[str] = None
    
    # Metrics and tracking
    metrics: Optional[BuildMetrics] = None
    artifacts: List[BuildArtifact] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)
    error_messages: List[str] = field(default_factory=list)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    # Version control
    git_commit: Optional[str] = None
    git_branch: Optional[str] = None
    version: Optional[str] = None
    
    def start_build(self) -> None:
        """Start the build session."""
        self.status = BuildStatus.IN_PROGRESS
        self.metrics = BuildMetrics(start_time=datetime.utcnow())
        self.updated_at = datetime.utcnow()
        self.add_log(f"Build session {self.session_id} started")
    
    def complete_build(self, success: bool = True) -> None:
        """Complete the build session."""
        self.status = BuildStatus.COMPLETED if success else BuildStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        
        if self.metrics:
            self.metrics.end_time = datetime.utcnow()
            duration = (self.metrics.end_time - self.metrics.start_time).total_seconds()
            self.metrics.duration_seconds = duration
        
        status_msg = "completed successfully" if success else "failed"
        self.add_log(f"Build session {self.session_id} {status_msg}")
    
    def cancel_build(self) -> None:
        """Cancel the build session."""
        self.status = BuildStatus.CANCELLED
        self.updated_at = datetime.utcnow()
        self.add_log(f"Build session {self.session_id} cancelled")
    
    def add_artifact(self, file_path: str, file_type: str, size_bytes: int, 
                    checksum: Optional[str] = None) -> BuildArtifact:
        """Add an artifact to the build session."""
        artifact = BuildArtifact(
            file_path=file_path,
            file_type=file_type,
            size_bytes=size_bytes,
            checksum=checksum
        )
        self.artifacts.append(artifact)
        
        if self.metrics:
            self.metrics.files_generated += 1
        
        self.add_log(f"Artifact added: {file_path}")
        return artifact
    
    def add_log(self, message: str) -> None:
        """Add a log message to the build session."""
        timestamp = datetime.utcnow().isoformat()
        log_entry = f"[{timestamp}] {message}"
        self.logs.append(log_entry)
    
    def add_error(self, error_message: str) -> None:
        """Add an error message to the build session."""
        self.error_messages.append(error_message)
        
        if self.metrics:
            self.metrics.errors_count += 1
        
        self.add_log(f"ERROR: {error_message}")
    
    def add_warning(self, warning_message: str) -> None:
        """Add a warning message to the build session."""
        if self.metrics:
            self.metrics.warnings_count += 1
        
        self.add_log(f"WARNING: {warning_message}")
    
    def update_metrics(self, **kwargs) -> None:
        """Update build metrics with provided values."""
        if not self.metrics:
            self.metrics = BuildMetrics(start_time=datetime.utcnow())
        
        for key, value in kwargs.items():
            if hasattr(self.metrics, key):
                setattr(self.metrics, key, value)
    
    def is_active(self) -> bool:
        """Check if the build session is currently active."""
        return self.status == BuildStatus.IN_PROGRESS
    
    def is_completed(self) -> bool:
        """Check if the build session has completed."""
        return self.status in [BuildStatus.COMPLETED, BuildStatus.FAILED, BuildStatus.CANCELLED]
    
    def get_duration(self) -> Optional[float]:
        """Get the duration of the build session in seconds."""
        if self.metrics and self.metrics.duration_seconds:
            return self.metrics.duration_seconds
        
        if self.metrics and self.metrics.start_time:
            if self.metrics.end_time:
                return (self.metrics.end_time - self.metrics.start_time).total_seconds()
            else:
                return (datetime.utcnow() - self.metrics.start_time).total_seconds()
        
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the build session to a dictionary."""
        return {
            "session_id": self.session_id,
            "project_id": self.project_id,
            "user_id": self.user_id,
            "build_type": self.build_type.value,
            "status": self.status.value,
            "config": self.config,
            "environment": self.environment,
            "source_directory": self.source_directory,
            "output_directory": self.output_directory,
            "target_platform": self.target_platform,
            "build_command": self.build_command,
            "metrics": {
                "start_time": self.metrics.start_time.isoformat() if self.metrics else None,
                "end_time": self.metrics.end_time.isoformat() if self.metrics and self.metrics.end_time else None,
                "duration_seconds": self.metrics.duration_seconds if self.metrics else None,
                "files_processed": self.metrics.files_processed if self.metrics else 0,
                "files_generated": self.metrics.files_generated if self.metrics else 0,
                "errors_count": self.metrics.errors_count if self.metrics else 0,
                "warnings_count": self.metrics.warnings_count if self.metrics else 0,
                "memory_usage_mb": self.metrics.memory_usage_mb if self.metrics else None,
                "cpu_usage_percent": self.metrics.cpu_usage_percent if self.metrics else None,
            },
            "artifacts": [
                {
                    "artifact_id": a.artifact_id,
                    "file_path": a.file_path,
                    "file_type": a.file_type,
                    "size_bytes": a.size_bytes,
                    "checksum": a.checksum,
                    "created_at": a.created_at.isoformat()
                }
                for a in self.artifacts
            ],
            "logs": self.logs,
            "error_messages": self.error_messages,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "git_commit": self.git_commit,
            "git_branch": self.git_branch,
            "version": self.version
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BuildSession":
        """Create a BuildSession from a dictionary."""
        session = cls(
            session_id=data.get("session_id", str(uuid.uuid4())),
            project_id=data.get("project_id", ""),
            user_id=data.get("user_id", ""),
            build_type=BuildType(data.get("build_type", BuildType.FULL_BUILD.value)),
            status=BuildStatus(data.get("status", BuildStatus.PENDING.value)),
            config=data.get("config", {}),
            environment=data.get("environment", {}),
            source_directory=data.get("source_directory", ""),
            output_directory=data.get("output_directory", ""),
            target_platform=data.get("target_platform"),
            build_command=data.get("build_command"),
            logs=data.get("logs", []),
            error_messages=data.get("error_messages", []),
            git_commit=data.get("git_commit"),
            git_branch=data.get("git_branch"),
            version=data.get("version")
        )
        
        # Parse timestamps
        if data.get("created_at"):
            session.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("updated_at"):
            session.updated_at = datetime.fromisoformat(data["updated_at"])
        if data.get("completed_at"):
            session.completed_at = datetime.fromisoformat(data["completed_at"])
        
        # Parse metrics
        metrics_data = data.get("metrics", {})
        if metrics_data and metrics_data.get("start_time"):
            session.metrics = BuildMetrics(
                start_time=datetime.fromisoformat(metrics_data["start_time"]),
                end_time=datetime.fromisoformat(metrics_data["end_time"]) if metrics_data.get("end_time") else None,
                duration_seconds=metrics_data.get("duration_seconds"),
                files_processed=metrics_data.get("files_processed", 0),
                files_generated=metrics_data.get("files_generated", 0),
                errors_count=metrics_data.get("errors_count", 0),
                warnings_count=metrics_data.get("warnings_count", 0),
                memory_usage_mb=metrics_data.get("memory_usage_mb"),
                cpu_usage_percent=metrics_data.get("cpu_usage_percent")
            )
        
        # Parse artifacts
        for artifact_data in data.get("artifacts", []):
            artifact = BuildArtifact(
                artifact_id=artifact_data.get("artifact_id", str(uuid.uuid4())),
                file_path=artifact_data.get("file_path", ""),
                file_type=artifact_data.get("file_type", ""),
                size_bytes=artifact_data.get("size_bytes", 0),
                checksum=artifact_data.get("checksum")
            )
            if artifact_data.get("created_at"):
                artifact.created_at = datetime.fromisoformat(artifact_data["created_at"])
            session.artifacts.append(artifact)
        
        return session


def create_build_session(project_id: str, user_id: str, 
                        build_type: BuildType = BuildType.FULL_BUILD,
                        **kwargs) -> BuildSession:
    """
    Factory function to create a new build session.
    
    Args:
        project_id: The project identifier
        user_id: The user identifier
        build_type: The type of build
        **kwargs: Additional optional parameters
    
    Returns:
        A new BuildSession instance
    """
    return BuildSession(
        project_id=project_id,
        user_id=user_id,
        build_type=build_type,
        source_directory=kwargs.get("source_directory", ""),
        output_directory=kwargs.get("output_directory", ""),
        target_platform=kwargs.get("target_platform"),
        build_command=kwargs.get("build_command"),
        config=kwargs.get("config", {}),
        environment=kwargs.get("environment", {}),
        git_commit=kwargs.get("git_commit"),
        git_branch=kwargs.get("git_branch"),
        version=kwargs.get("version")
    )