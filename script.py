import os
from typing import Dict, List, Set

from fastapi import FastAPI, Form, HTTPException
from google import genai
from pydantic import BaseModel, Field

# -------------------------------------------------------------------
# FastAPI App Initialization
# -------------------------------------------------------------------
app = FastAPI(
    title="Personalized Learning Path Service",
    description="Feature #2: Intelligent Daily Learning Path Scheduler with Gemini Rationale",
    version="1.0.0",
)


# -------------------------------------------------------------------
# Helper: Gemini Client Setup
# -------------------------------------------------------------------
def get_gemini_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY environment variable is not set.",
        )
    return genai.Client(api_key=api_key)


# -------------------------------------------------------------------
# Data Schemas & Models
# -------------------------------------------------------------------
class Node(BaseModel):
    id: str
    subject_id: str
    title_ar: str
    title_en: str
    difficulty: int = Field(default=3, ge=1, le=5)
    estimated_minutes: int = Field(default=15, ge=5, le=60)


class Edge(BaseModel):
    from_node_id: str
    to_node_id: str


class StudentNodeState(BaseModel):
    student_id: str
    node_id: str
    mastery_score: float = Field(default=0.0, ge=0.0, le=1.0)
    is_exam_flagged: bool = False
    days_since_last_studied: int = 0


class ScheduledLessonResponse(BaseModel):
    node_id: str
    title_ar: str
    subject_id: str
    difficulty: int
    estimated_minutes: int
    priority_score: float
    reason_flag: str
    ai_rationale: str


class LearningPathResponse(BaseModel):
    student_id: str
    total_scheduled_minutes: int
    daily_budget_minutes: int
    lessons: List[ScheduledLessonResponse]


# -------------------------------------------------------------------
# Engine 1: Graph Traversal & Prerequisite Engine
# -------------------------------------------------------------------
class GraphEngine:

    def __init__(self, nodes: List[Node], edges: List[Edge]):
        self.nodes_map: Dict[str, Node] = {node.id: node for node in nodes}
        self.downstream_map: Dict[str, List[str]] = {
            node.id: [] for node in nodes
        }
        self.prerequisites_map: Dict[str, List[str]] = {
            node.id: [] for node in nodes
        }

        for edge in edges:
            if edge.from_node_id in self.downstream_map:
                self.downstream_map[edge.from_node_id].append(edge.to_node_id)
            if edge.to_node_id in self.prerequisites_map:
                self.prerequisites_map[edge.to_node_id].append(
                    edge.from_node_id
                )

    def is_eligible_to_study(
        self,
        node_id: str,
        student_states: Dict[str, StudentNodeState],
        passing_threshold: float = 0.7,
    ) -> bool:
        prereqs = self.prerequisites_map.get(node_id, [])
        for prereq_id in prereqs:
            state = student_states.get(prereq_id)
            if not state or state.mastery_score < passing_threshold:
                return False
        return True

    def calculate_unlock_value(self, node_id: str) -> int:
        visited: Set[str] = set()
        queue: List[str] = list(self.downstream_map.get(node_id, []))
        while queue:
            current = queue.pop(0)
            if current not in visited:
                visited.add(current)
                queue.extend(self.downstream_map.get(current, []))
        return max(1, len(visited))


# -------------------------------------------------------------------
# Engine 2: Priority Scoring Engine
# -------------------------------------------------------------------
class PriorityScorer:

    def __init__(self, graph_engine: GraphEngine):
        self.graph_engine = graph_engine

    def calculate_node_score(
        self, node_id: str, state: StudentNodeState
    ) -> float:
        mastery_gap = 1.0 - state.mastery_score
        if mastery_gap <= 0.05:
            return 0.0

        urgency_weight = 1.0
        if state.is_exam_flagged:
            urgency_weight += 0.5
        if state.days_since_last_studied > 14:
            urgency_weight += 0.3

        unlock_value = self.graph_engine.calculate_unlock_value(node_id)
        return round(mastery_gap * urgency_weight * unlock_value, 3)


# -------------------------------------------------------------------
# Curriculum Graph Data Setup
# -------------------------------------------------------------------
CURRICULUM_NODES = [
    Node(
        id="m_frac",
        subject_id="Mathematics",
        title_ar="جمع وطرح الكسور",
        title_en="Fractions Addition",
        difficulty=2,
        estimated_minutes=15,
    ),
    Node(
        id="m_quad",
        subject_id="Mathematics",
        title_ar="حل المعادلات التربيعية",
        title_en="Quadratics",
        difficulty=4,
        estimated_minutes=20,
    ),
    Node(
        id="m_calc",
        subject_id="Mathematics",
        title_ar="تفاضل وتكامل الدوال",
        title_en="Calculus",
        difficulty=5,
        estimated_minutes=25,
    ),
    Node(
        id="s_cell",
        subject_id="Science",
        title_ar="تركيب الخلية الحية",
        title_en="Cell Structure",
        difficulty=2,
        estimated_minutes=15,
    ),
    Node(
        id="e_grammar",
        subject_id="English",
        title_ar="قواعد الأزمنة الماضية",
        title_en="Past Tenses",
        difficulty=3,
        estimated_minutes=15,
    ),
    Node(
        id="a_grammar",
        subject_id="Arabic",
        title_ar="المبتدأ والخبر وعلامات الإعراب",
        title_en="Subject and Predicate",
        difficulty=2,
        estimated_minutes=15,
    ),
]

CURRICULUM_EDGES = [
    Edge(from_node_id="m_frac", to_node_id="m_quad"),
    Edge(from_node_id="m_quad", to_node_id="m_calc"),
]

CURRICULUM_GRAPH = GraphEngine(nodes=CURRICULUM_NODES, edges=CURRICULUM_EDGES)
PRIORITY_SCORER = PriorityScorer(graph_engine=CURRICULUM_GRAPH)


# -------------------------------------------------------------------
# Engine 3: AI Rationale Generator (Gemini LLM)
# -------------------------------------------------------------------
def generate_ai_rationale(
    client: genai.Client,
    lesson_title: str,
    subject: str,
    reason_flag: str,
    mastery_percent: int,
) -> str:
    prompt = f"""
    Write a single, highly encouraging, one-line explanation in Arabic (under 20 words) for a student explaining why this lesson was chosen for them today.
    Lesson: {lesson_title} ({subject})
    Context: Current Mastery is {mastery_percent}%, Reason: {reason_flag}.
    Style: Warm, motivating, plain text only.
    """
    try:
        res = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt,
        )
        if res.text:
            return res.text.strip().replace("\n", " ")
        return f"تم اختيار درس {lesson_title} لتقوية مهاراتك وسد الثغرات التعليمية."
    except Exception:
        return f"تم اختيار درس {lesson_title} لتقوية مهاراتك وسد الثغرات التعليمية الأساسية."


# -------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------
@app.get("/", tags=["General"])
def home():
    return {
        "message": "Personalized Learning Path Service is running. Visit /docs to test the API."
    }


@app.post(
    "/learning_path/generate",
    tags=["Feature #2: Personalized Learning Path"],
    response_model=LearningPathResponse,
)
async def generate_personalized_learning_path(
    student_id: str = Form("std_101", description="Student unique ID"),
    daily_time_minutes: int = Form(
        45, description="Available study time for today (e.g. 30, 45, 60)"
    ),
    include_math: bool = Form(True, description="Include Mathematics"),
    include_science: bool = Form(True, description="Include Science"),
    include_arabic: bool = Form(True, description="Include Arabic"),
    include_english: bool = Form(True, description="Include English"),
):
    """
    Builds an optimized daily learning path based on the student's mastery profile and prerequisite graph.
    Uses AI strictly to generate a one-line motivating rationale for each scheduled topic.
    """
    # 1. فلترة المواد المشترك بها الطالب
    subscribed_subjects = []
    if include_math:
        subscribed_subjects.append("Mathematics")
    if include_science:
        subscribed_subjects.append("Science")
    if include_arabic:
        subscribed_subjects.append("Arabic")
    if include_english:
        subscribed_subjects.append("English")

    if not subscribed_subjects:
        raise HTTPException(
            status_code=400, detail="Please select at least one subject."
        )

    # 2. سجل تقييم الطالب ومستوى إتقانه (Knowledge Profile)
    student_states: Dict[str, StudentNodeState] = {
        "m_frac": StudentNodeState(
            student_id=student_id,
            node_id="m_frac",
            mastery_score=0.35,
            is_exam_flagged=True,
            days_since_last_studied=4,
        ),
        "m_quad": StudentNodeState(
            student_id=student_id,
            node_id="m_quad",
            mastery_score=0.20,
            is_exam_flagged=False,
            days_since_last_studied=1,
        ),
        "m_calc": StudentNodeState(
            student_id=student_id,
            node_id="m_calc",
            mastery_score=0.10,
            is_exam_flagged=False,
            days_since_last_studied=0,
        ),
        "s_cell": StudentNodeState(
            student_id=student_id,
            node_id="s_cell",
            mastery_score=0.40,
            is_exam_flagged=False,
            days_since_last_studied=18,
        ),
        "e_grammar": StudentNodeState(
            student_id=student_id,
            node_id="e_grammar",
            mastery_score=0.50,
            is_exam_flagged=True,
            days_since_last_studied=6,
        ),
        "a_grammar": StudentNodeState(
            student_id=student_id,
            node_id="a_grammar",
            mastery_score=0.85,
            is_exam_flagged=False,
            days_since_last_studied=3,
        ),
    }

    # 3. حساب نقاط الأولوية وتصفية الدروس المؤهلة
    candidates = []
    for node_id, node in CURRICULUM_GRAPH.nodes_map.items():
        if node.subject_id not in subscribed_subjects:
            continue
        if not CURRICULUM_GRAPH.is_eligible_to_study(node_id, student_states):
            continue

        state = student_states.get(
            node_id,
            StudentNodeState(student_id=student_id, node_id=node_id),
        )
        score = PRIORITY_SCORER.calculate_node_score(node_id, state)

        if score > 0:
            flag = (
                "خطأ في الامتحان الأخير"
                if state.is_exam_flagged
                else (
                    "تنشيط موضوع غير مراجع"
                    if state.days_since_last_studied > 14
                    else "سد ثغرة تعليمية"
                )
            )
            candidates.append((node, score, flag, state.mastery_score))

    # الترتيب التنازلي حسب الأولوية
    candidates.sort(key=lambda x: x[1], reverse=True)

    # 4. خوارزمية التعبئة الذكية (Greedy Packing)
    scheduled_lessons = []
    time_used = 0
    last_was_hard = False
    client = get_gemini_client()

    for node, score, flag, mastery in candidates:
        if time_used + node.estimated_minutes <= daily_time_minutes:
            is_hard = node.difficulty >= 4
            if last_was_hard and is_hard:
                continue

            # استدعاء الذكاء الاصطناعي لكتابة تعليل الدرس فقط
            rationale = generate_ai_rationale(
                client=client,
                lesson_title=node.title_ar,
                subject=node.subject_id,
                reason_flag=flag,
                mastery_percent=int(mastery * 100),
            )

            scheduled_lessons.append(
                ScheduledLessonResponse(
                    node_id=node.id,
                    title_ar=node.title_ar,
                    subject_id=node.subject_id,
                    difficulty=node.difficulty,
                    estimated_minutes=node.estimated_minutes,
                    priority_score=score,
                    reason_flag=flag,
                    ai_rationale=rationale,
                )
            )

            time_used += node.estimated_minutes
            last_was_hard = is_hard

    return LearningPathResponse(
        student_id=student_id,
        total_scheduled_minutes=time_used,
        daily_budget_minutes=daily_time_minutes,
        lessons=scheduled_lessons,
    )
