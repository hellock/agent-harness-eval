"""Grader specifications and grader result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

GraderKind = Literal[
    "test_pass",
    "file_exists",
    "regex",
    "json_schema",
    "trajectory",
    "rubric_judge",
    "test_suite",
]

TrajectoryRuleKind = Literal[
    "tool_called",
    "no_dangerous_commands",
    "no_loop",
    "read_before_answer",
    "boundary_respected",
]


@dataclass
class TrajectoryRule:
    kind: TrajectoryRuleKind
    tool_pattern: str | None = None
    min: int | None = None
    max: int | None = None
    forbidden_patterns: list[str] | None = None
    max_consecutive_identical: int | None = None
    constraint: Literal["internet", "shell", "file_write"] | None = None


@dataclass
class RubricDimensionDef:
    name: str
    required: bool = True
    weight: float = 1.0


@dataclass
class RubricDimensionResult:
    name: str
    passed: bool
    score: float
    reason: str | None = None
    required: bool = True
    weight: float = 1.0


@dataclass
class TestPassGrader:
    type: GraderKind = field(default="test_pass", init=False)
    command: str = ""
    cwd: str | None = None


@dataclass
class FileExistsGrader:
    type: GraderKind = field(default="file_exists", init=False)
    paths: list[str] = field(default_factory=list)


@dataclass
class RegexGrader:
    type: GraderKind = field(default="regex", init=False)
    target: Literal["final_text", "artifact"] = "final_text"
    pattern: str = ""
    should_match: bool = True
    artifact_path: str | None = None
    case_insensitive: bool | None = None


@dataclass
class JsonSchemaGrader:
    type: GraderKind = field(default="json_schema", init=False)
    target: Literal["final_text", "artifact"] = "final_text"
    artifact_path: str = ""
    schema: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrajectoryGrader:
    type: GraderKind = field(default="trajectory", init=False)
    rule: TrajectoryRule = field(default_factory=lambda: TrajectoryRule(kind="tool_called"))


@dataclass
class RubricJudgeGrader:
    type: GraderKind = field(default="rubric_judge", init=False)
    rubric: str = ""
    dimensions: list[str | RubricDimensionDef] | None = None
    snapshot_paths: list[str] | None = None
    hard_grader_override: bool | None = None


@dataclass
class TestSuiteCase:
    name: str
    command: str


@dataclass
class TestSuiteGrader:
    type: GraderKind = field(default="test_suite", init=False)
    cases: list[TestSuiteCase] = field(default_factory=list)
    runner: str | None = None
    working_dir: str | None = None
    pass_threshold: float = 1.0
    setup_commands: list[str] = field(default_factory=list)


GraderSpec = (
    TestPassGrader
    | FileExistsGrader
    | RegexGrader
    | JsonSchemaGrader
    | TrajectoryGrader
    | RubricJudgeGrader
    | TestSuiteGrader
)


@dataclass
class GraderResult:
    grader_type: str
    name: str
    passed: bool
    score: float | None = None
    details: str | None = None
    dimensions: list[RubricDimensionResult] | None = None


def parse_grader_spec(data: dict[str, Any]) -> GraderSpec:
    """Deserialize a grader spec from a task.yaml dict entry."""
    type_str = data.get("type", "")
    if type_str == "test_suite":
        return _parse_test_suite_spec(data)

    cls_map: dict[str, type] = {
        "test_pass": TestPassGrader,
        "file_exists": FileExistsGrader,
        "regex": RegexGrader,
        "json_schema": JsonSchemaGrader,
        "trajectory": TrajectoryGrader,
        "rubric_judge": RubricJudgeGrader,
    }
    cls = cls_map.get(type_str)
    if cls is None:
        raise ValueError(f"Unknown grader type: {type_str!r}")

    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        if key == "type":
            continue
        if key == "rule" and isinstance(value, dict):
            kwargs["rule"] = TrajectoryRule(**value)
        elif key == "dimensions" and isinstance(value, list):
            dims: list[str | RubricDimensionDef] = []
            for item in value:
                if isinstance(item, str):
                    dims.append(item)
                elif isinstance(item, dict):
                    dims.append(RubricDimensionDef(**item))
            kwargs["dimensions"] = dims
        else:
            kwargs[key] = value

    return cls(**kwargs)  # type: ignore[return-value]


def _parse_test_suite_spec(data: dict[str, Any]) -> TestSuiteGrader:
    """Parse a test_suite grader spec, handling both dict and shorthand case formats."""
    runner = data.get("runner")
    working_dir = data.get("working_dir")
    pass_threshold = data.get("pass_threshold", 1.0)
    raw_cases = data.get("cases", [])

    cases: list[TestSuiteCase] = []
    for item in raw_cases:
        if isinstance(item, str):
            if runner == "pytest":
                cases.append(TestSuiteCase(name=item, command=f"uv run python -m pytest {item} -x --tb=short"))
            else:
                cases.append(TestSuiteCase(name=item, command=item))
        elif isinstance(item, dict):
            name = item.get("name", item.get("command", "unnamed"))
            command = item.get("command", "")
            if not command:
                raise ValueError(f"test_suite case missing 'command': {item!r}")
            cases.append(TestSuiteCase(name=name, command=command))

    setup_commands = data.get("setup_commands", [])

    grader = TestSuiteGrader(
        cases=cases,
        runner=runner,
        working_dir=working_dir,
        pass_threshold=pass_threshold,
        setup_commands=setup_commands,
    )
    return grader


def grader_result_from_dict(data: dict[str, Any]) -> GraderResult:
    dimensions = None
    if data.get("dimensions"):
        dimensions = [
            RubricDimensionResult(
                name=dim["name"],
                passed=dim["pass"],
                score=dim["score"],
                reason=dim.get("reason"),
                required=dim.get("required", True),
                weight=dim.get("weight", 1.0),
            )
            for dim in data["dimensions"]
        ]

    return GraderResult(
        grader_type=data["grader_type"],
        name=data["name"],
        passed=data["pass"],
        score=data.get("score"),
        details=data.get("details"),
        dimensions=dimensions,
    )
