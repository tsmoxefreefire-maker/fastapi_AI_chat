import io
import mimetypes
import os
import xml.etree.ElementTree as ET
import zipfile
from enum import Enum
from typing import Dict, List, Set

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

app = FastAPI(
    title="AI Developer & Learning Toolkit API",
    description="A multi-purpose AI service with Intelligent Learning Path & Gemini 3.5 Flash",
    version="1.0.0",
)


# -------------------------------------------------------------------
# Helper: Get Gemini Client
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
# Part 2: Feature #2 (Personalized Learning Path Generator) Engine
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


MOCK_CURRICULUM_NODES = [
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

MOCK_CURRICULUM_EDGES = [
    Edge(from_node_id="m_frac", to_node_id="m_quad"),
    Edge(from_node_id="m_quad", to_node_id="m_calc"),
]

CURRICULUM_GRAPH = GraphEngine(
    nodes=MOCK_CURRICULUM_NODES, edges=MOCK_CURRICULUM_EDGES
)
PRIORITY_SCORER = PriorityScorer(graph_engine=CURRICULUM_GRAPH)


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
    except (RuntimeError, ValueError, Exception):
        return f"تم اختيار درس {lesson_title} لتقوية مهاراتك وسد الثغرات التعليمية الأساسية."


# -------------------------------------------------------------------
# Helper: Universal File Handler
# -------------------------------------------------------------------
def extract_text_from_docx(file_bytes: bytes) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
            xml_content = z.read("word/document.xml")
            tree = ET.fromstring(xml_content)
            texts: List[str] = [
                str(node.text)
                for node in tree.iter()
                if node.tag.endswith("}t") and node.text is not None
            ]
            return "\n".join(texts)
    except (zipfile.BadZipFile, ET.ParseError, KeyError, Exception):
        return ""


def process_uploaded_file(file_bytes: bytes, filename: str, content_type: str):
    if not file_bytes or not file_bytes.strip():
        raise HTTPException(
            status_code=400,
            detail=f"The uploaded file '{filename}' is empty. Please provide a file with content.",
        )

    ext = os.path.splitext(filename)[1].lower()
    guessed_mime, _ = mimetypes.guess_type(filename)
    mime_type = (
        content_type
        if (content_type and content_type != "application/octet-stream")
        else (guessed_mime or "application/octet-stream")
    )

    if ext == ".docx":
        extracted_text = extract_text_from_docx(file_bytes)
        if extracted_text.strip():
            return extracted_text

    arabic_and_text_encodings = [
        "utf-8",
        "utf-8-sig",
        "cp1256",
        "iso-8859-6",
        "utf-16",
        "latin_1",
    ]
    text_extensions = {
        ".txt",
        ".md",
        ".csv",
        ".json",
        ".py",
        ".html",
        ".htm",
        ".xml",
        ".js",
        ".ts",
        ".css",
        ".java",
        ".c",
        ".cpp",
        ".cs",
        ".php",
        ".rb",
        ".sql",
        ".sh",
        ".yaml",
        ".yml",
        ".ini",
        ".log",
    }

    if (
        ext in text_extensions
        or mime_type.startswith("text/")
        or mime_type
        in ["application/json", "application/javascript", "application/xml"]
    ):
        for enc in arabic_and_text_encodings:
            try:
                decoded_text = file_bytes.decode(enc)
                if decoded_text and decoded_text.strip():
                    return decoded_text
            except (UnicodeDecodeError, LookupError):
                continue

    return types.Part.from_bytes(data=file_bytes, mime_type=mime_type)


@app.get("/", tags=["General"])
def home():
    return {"message": "API is online. Go to /docs to use the interactive UI."}


# -------------------------------------------------------------------
# Endpoint: Feature #2 - Personalized Learning Path Generator
# -------------------------------------------------------------------
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

    candidates.sort(key=lambda x: x[1], reverse=True)

    scheduled_lessons = []
    time_used = 0
    last_was_hard = False
    client = get_gemini_client()

    for node, score, flag, mastery in candidates:
        if time_used + node.estimated_minutes <= daily_time_minutes:
            is_hard = node.difficulty >= 4
            if last_was_hard and is_hard:
                continue

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


# -------------------------------------------------------------------
# Part 1: Programming-Only AI Assistant
# -------------------------------------------------------------------
@app.post("/chat/programming", tags=["Part 1: Programming Chat"])
async def programming_chat(
    question: str = Form(
        ..., description="Type your programming question here"
    ),
):
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    client = get_gemini_client()
    system_instruction = """
    You are a strict programming assistant.
    Answer ONLY questions related to programming, software engineering, databases, algorithms, web development, and computer science.
    Formatting: Clean plain text without markdown symbols (no ###, **, $).
    """
    prompt = f"{system_instruction}\n\nUser Question: {question}"

    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash", contents=prompt
        )
        answer_text = response.text or ""
        return {"answer": answer_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------------------------------------------
# Part 2: Document Summarizer (.txt File Export)
# -------------------------------------------------------------------
@app.post("/summarize", tags=["Part 2: Document Summarizer"])
async def summarize_document(file: UploadFile = File(...)):
    client = get_gemini_client()
    try:
        file_bytes = await file.read()
        filename = file.filename or "file.txt"
        content_type = file.content_type or "text/plain"

        file_input = process_uploaded_file(file_bytes, filename, content_type)
        prompt_rules = "Extract the most important points in clean plain text. Match the language of the document."

        contents = (
            [
                f"DOCUMENT CONTENT ({filename}):\n\n{file_input}\n\nINSTRUCTIONS:\n{prompt_rules}"
            ]
            if isinstance(file_input, str)
            else [file_input, prompt_rules]
        )

        response = client.models.generate_content(
            model="gemini-3.5-flash", contents=contents
        )
        txt_data = response.text or ""
        file_stream = io.BytesIO(txt_data.encode("utf-8"))

        base_name = os.path.splitext(filename)[0]
        new_filename = f"summary_{base_name}.txt"

        return StreamingResponse(
            file_stream,
            media_type="text/plain; charset=utf-8",
            headers={
                "Content-Disposition": f"attachment; filename={new_filename}"
            },
        )
    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------------------------------------------
# Part 3: Custom Question Generator (.txt File Export)
# -------------------------------------------------------------------
@app.post("/generate_questions", tags=["Part 3: Question Generator"])
async def generate_questions_file(
    num_questions: int = Form(5, description="Total number of questions"),
    mcq_percent: int = Form(
        0, description="Multiple choice percentage (0-100)"
    ),
    tf_percent: int = Form(100, description="True/False percentage (0-100)"),
    essay_percent: int = Form(0, description="Short answer percentage (0-100)"),
    file: UploadFile = File(...),
):
    if num_questions < 1 or num_questions > 20:
        raise HTTPException(
            status_code=400,
            detail="Please enter a question count between 1 and 20.",
        )

    total_percentage = mcq_percent + tf_percent + essay_percent
    if total_percentage != 100:
        raise HTTPException(
            status_code=400,
            detail=f"Percentages must sum to 100%. Current sum: {total_percentage}%",
        )

    count_mcq = round((mcq_percent / 100) * num_questions)
    count_tf = round((tf_percent / 100) * num_questions)
    count_essay = num_questions - (count_mcq + count_tf)

    client = get_gemini_client()
    try:
        file_bytes = await file.read()
        filename = file.filename or "file.txt"
        content_type = file.content_type or "text/plain"

        file_input = process_uploaded_file(file_bytes, filename, content_type)
        prompt_rules = f"""
        Generate exactly {num_questions} questions:
        - MCQ: {count_mcq}
        - True/False: {count_tf}
        - Short Answer: {count_essay}
        Format in clean plain text with an Answer Key at the end. Match the document language.
        """

        contents = (
            [
                f"DOCUMENT CONTENT ({filename}):\n\n{file_input}\n\nINSTRUCTIONS:\n{prompt_rules}"
            ]
            if isinstance(file_input, str)
            else [file_input, prompt_rules]
        )

        response = client.models.generate_content(
            model="gemini-3.5-flash", contents=contents
        )
        txt_data = response.text or ""
        file_stream = io.BytesIO(txt_data.encode("utf-8"))

        base_name = os.path.splitext(filename)[0]
        new_filename = f"questions_{base_name}.txt"

        return StreamingResponse(
            file_stream,
            media_type="text/plain; charset=utf-8",
            headers={
                "Content-Disposition": f"attachment; filename={new_filename}"
            },
        )
    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------------------------------------------
# Part 4: Core School Subjects Tutor
# -------------------------------------------------------------------
class SubjectName(str, Enum):
    MATHEMATICS = "Mathematics"
    SCIENCE = "Science"
    ARABIC = "Arabic"
    ENGLISH = "English"


@app.post("/chat/subjects", tags=["Part 4: Core Subjects Tutor"])
async def subjects_chat(
    subject: SubjectName = Form(..., description="Select the subject"),
    question: str = Form(..., description="Type your question or request here"),
):
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    client = get_gemini_client()
    system_instruction = f"""
    You are an expert tutor dedicated EXCLUSIVELY to teaching: {subject.value}.
    Format: Clean plain text without markdown symbols (no ###, **, $).
    """
    prompt = f"{system_instruction}\n\nStudent Question: {question}"

    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash", contents=prompt
        )
        answer_text = response.text or ""
        return {"selected_subject": subject.value, "answer": answer_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
