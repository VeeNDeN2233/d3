
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class AnalysisStep(Enum):
    LOGIN = "login"
    UPLOAD = "upload"
    PARAMETERS = "parameters"
    ANALYSIS = "analysis"
    RESULTS = "results"


@dataclass
class UserState:
    is_authenticated: bool = False
    session_token: Optional[str] = None
    email: Optional[str] = None
    username: Optional[str] = None
    full_name: Optional[str] = None
    role: str = "user"


@dataclass
class VideoState:
    file_path: Optional[str] = None
    file_name: Optional[str] = None
    file_size: int = 0
    is_uploaded: bool = False
    is_valid: bool = False
    validation_error: Optional[str] = None


@dataclass
class AnalysisParameters:
    patient_age_weeks: int = 12
    gestational_age: int = 40


@dataclass
class AnalysisState:
    is_running: bool = False
    is_cancelled: bool = False
    progress: float = 0.0
    current_step: str = ""
    error: Optional[str] = None
    results: Optional[Dict[str, Any]] = None


@dataclass
class ModelState:
    is_loaded: bool = False
    loading_error: Optional[str] = None
    status_message: str = "Модели не загружены"


@dataclass
class AppState:
    user: UserState = field(default_factory=UserState)
    video: VideoState = field(default_factory=VideoState)
    parameters: AnalysisParameters = field(default_factory=AnalysisParameters)
    analysis: AnalysisState = field(default_factory=AnalysisState)
    models: ModelState = field(default_factory=ModelState)
    current_step: AnalysisStep = AnalysisStep.LOGIN
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'user': asdict(self.user),
            'video': asdict(self.video),
            'parameters': asdict(self.parameters),
            'analysis': asdict(self.analysis),
            'models': asdict(self.models),
            'current_step': self.current_step.value,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AppState':
        state = cls()
        if 'user' in data:
            state.user = UserState(**data['user'])
        if 'video' in data:
            state.video = VideoState(**data['video'])
        if 'parameters' in data:
            state.parameters = AnalysisParameters(**data['parameters'])
        if 'analysis' in data:
            state.analysis = AnalysisState(**data['analysis'])
        if 'models' in data:
            state.models = ModelState(**data['models'])
        if 'current_step' in data:
            state.current_step = AnalysisStep(data['current_step'])
        return state
    
    def save_to_file(self, file_path: Path) -> bool:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"Состояние сохранено: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Ошибка сохранения состояния: {e}")
            return False
    
    @classmethod
    def load_from_file(cls, file_path: Path) -> Optional['AppState']:
        try:
            if not file_path.exists():
                return None
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return cls.from_dict(data)
        except Exception as e:
            logger.error(f"Ошибка загрузки состояния: {e}")
            return None
    
    def reset_analysis(self):
        self.analysis = AnalysisState()
        self.video = VideoState()
        self.current_step = AnalysisStep.UPLOAD if self.user.is_authenticated else AnalysisStep.LOGIN
    
    def reset_all(self):
        self.user = UserState()
        self.video = VideoState()
        self.parameters = AnalysisParameters()
        self.analysis = AnalysisState()
        self.current_step = AnalysisStep.LOGIN


class StateManager:
    
    def __init__(self):
        self.state = AppState()
        self._listeners: List[callable] = []
    
    def get_state(self) -> AppState:
        return self.state
    
    def update_state(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)
            else:
                logger.warning(f"Неизвестное поле состояния: {key}")
        self._notify_listeners()
    
    def update_user(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self.state.user, key):
                setattr(self.state.user, key, value)
        self._notify_listeners()
    
    def update_video(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self.state.video, key):
                setattr(self.state.video, key, value)
        self._notify_listeners()
    
    def update_analysis(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self.state.analysis, key):
                setattr(self.state.analysis, key, value)
        self._notify_listeners()
    
    def update_models(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self.state.models, key):
                setattr(self.state.models, key, value)
        self._notify_listeners()
    
    def set_step(self, step: AnalysisStep):
        self.state.current_step = step
        self._notify_listeners()
    
    def add_listener(self, callback: callable):
        self._listeners.append(callback)
    
    def _notify_listeners(self):
        for listener in self._listeners:
            try:
                listener(self.state)
            except Exception as e:
                logger.error(f"Ошибка в слушателе состояния: {e}")

