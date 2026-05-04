import json
import re
from dataclasses import dataclass
from typing import Literal
from urllib import error, request
from .local_judge import LocalTeacherJudge as LLMJudge