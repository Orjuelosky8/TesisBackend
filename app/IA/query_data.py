# app/query_data.py
from __future__ import annotations

import os
import json
import logging
import re
from functools import lru_cache
from typing import Any, Dict, List, Tuple, Optional

from sqlalchemy import text

from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableWithMessageHistory

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

from db.deps import engine
from IA.memory import get_history, EPHEMERAL_SESSION_ID

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if "GEMINI_API_KEY" not in os.environ:
    raise ValueError("La variable de entorno GEMINI_API_KEY no está configurada.")

DEFAULT_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))

EXPOSED_TABLES = [
    "licitacion",
    "licitacion_chunk",
    "flags",
    "flags_licitaciones",
    "flags_log",
    "chunks",
    "documents",
    "doc_section_hits",
    "document_text_samples",
    "licitacion_keymap",
]

BUSINESS_CONTEXT = """
Contexto de negocio (auditoría analítica de licitaciones):

- Cada licitación puede tener múltiples FLAGS y logs de evaluación.
- Se quiere poder describir una licitación en lenguaje natural.
"""

AUDIT_MODE_HINT = "Modo auditoría: no inventes datos, si no hay filas dilo."
NARRATIVE_STYLE_HINT = "Responde en español natural, máximo 2-3 párrafos, sin SQL ni tablas."

SYSTEM_PROMPT_TEMPLATE = """
Eres un asistente experto en PostgreSQL trabajando con LangChain.

OBJETIVO (OBLIGATORIO):
- Para cada pregunta del usuario, DEBES generar y ejecutar UNA única consulta SQL válida.
- La respuesta final debe estar basada ESTRICTAMENTE en los resultados de esa consulta.
- La respuesta final DEBE ser en lenguaje natural.
- NO muestres la consulta SQL, ni DDL, ni el esquema, ni tablas Markdown.

CONTEXTO:
{business_context}

ESQUEMA (solo para USO INTERNO, NO LO MUESTRES):
{table_info}

{audit_hint}

{narrative_hint}
"""

TOPIC_KEYWORDS = {
    "educacion": ["educacion", "educación", "colegio", "universidad", "escuela", "formación", "capacitació"],
    "construccion": ["construccion", "construcción", "obra", "infraestructura", "vía", "edificación"],
    "salud": ["salud", "hospital", "clínica", "clínico"],
    "tecnologia": ["tecnologia", "tecnología", "software", "licencias", "ti"],
    "seguridad": ["seguridad", "cctv", "vigilancia", "camaras"],
}


def _to_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        parts: List[str] = []
        for p in x:
            if isinstance(p, dict):
                if "text" in p:
                    parts.append(str(p["text"]))
                elif p.get("type") == "text" and "text" in p:
                    parts.append(str(p["text"]))
            else:
                parts.append(str(p))
        return " ".join(s for s in parts if s).strip()
    if isinstance(x, dict):
        if x.get("type") == "text" and "text" in x:
            return str(x["text"])
        if "output" in x:
            return _to_text(x["output"])
        if "text" in x:
            return str(x["text"])
    return str(x)


def _naturalize_answer(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"```[^`]*```", "", text, flags=re.DOTALL)
    text = re.sub(
        r"El esquema de la tabla.*?(?:\n\s*\n|$)",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(
        r"^(?:CREATE|ALTER|DROP)\s+TABLE.*?$",
        "",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    cleaned_lines: List[str] = []
    for line in text.splitlines():
        l = line.strip()
        if l.startswith("|") and l.endswith("|"):
            continue
        if re.match(r"^\|?[-:]+(\|[-:]+)+$", l):
            continue
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text or len(text) < 20:
        return "La base de datos no devolvió suficiente detalle para mostrarlo en forma narrativa."
    return text


def _extract_sql_steps(result: Any) -> List[str]:
    sqls: List[str] = []
    if not isinstance(result, dict):
        return sqls
    steps = result.get("intermediate_steps")
    if not steps:
        return sqls
    for step in steps:
        try:
            action, _obs = step
            ti = getattr(action, "tool_input", None)
            if ti is None and isinstance(action, dict):
                ti = action.get("tool_input") or action.get("input")
            if ti:
                sqls.append(str(ti))
                continue
        except Exception:
            pass
        try:
            s = json.dumps(step, ensure_ascii=False)
        except Exception:
            s = str(step)
        sqls.append(s)
    return sqls


def _looks_count_by_topic(q: str) -> Optional[str]:
    ql = q.lower()
    if "cuantas" in ql or "cuántas" in ql or "cantidad" in ql or "número" in ql or "numero" in ql:
        for topic, words in TOPIC_KEYWORDS.items():
            if any(w in ql for w in words):
                return topic
    return None


def _looks_licitacion_by_id(q: str) -> Optional[int]:
    # el usuario mete saltos de línea, por eso DOTALL
    m = re.search(r"licitaci[oó]n\s*#?\s*(\d+)", q, flags=re.IGNORECASE | re.DOTALL)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def _run_sql(engine, sql: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    with engine.begin() as conn:
        rows = conn.execute(text(sql), params).mappings().all()
    return [dict(r) for r in rows]


def _handle_count_by_topic(engine, topic: str) -> Tuple[str, str]:
    words = TOPIC_KEYWORDS.get(topic, [])
    like_clauses = []
    params: Dict[str, Any] = {}
    for i, w in enumerate(words):
        like_clauses.append(f"(objeto ILIKE :w{i} OR texto_indexado ILIKE :w{i})")
        params[f"w{i}"] = f"%{w}%"
    where_sql = " OR ".join(like_clauses) if like_clauses else "TRUE"

    sql = f"""
        SELECT COUNT(*) AS cnt
        FROM public.licitacion
        WHERE {where_sql}
    """
    rows = _run_sql(engine, sql, params)
    cnt = rows[0]["cnt"] if rows else 0

    if cnt == 0:
        answer = f"No encontré licitaciones que parezcan relacionadas con {topic}."
    elif cnt == 1:
        answer = f"Encontré 1 licitación que parece relacionada con {topic}."
    else:
        answer = f"Encontré {cnt:,} licitaciones relacionadas con {topic}.".replace(",", ".")
    return answer, sql


def _handle_licitacion_by_id(engine, lic_id: int) -> Tuple[str, str]:
    sql = """
        SELECT id, entidad, objeto, cuantia, modalidad, numero, estado,
               fecha_public, ubicacion
        FROM public.licitacion
        WHERE id = :id
        LIMIT 1
    """
    rows = _run_sql(engine, sql, {"id": lic_id})
    if not rows:
        return f"No encontré una licitación con ID {lic_id}.", sql

    r = rows[0]
    entidad = r.get("entidad") or "una entidad pública"
    objeto = r.get("objeto") or "sin descripción de objeto"
    cuantia = r.get("cuantia")
    modalidad = r.get("modalidad") or "sin modalidad registrada"
    estado = r.get("estado") or "sin estado"
    fecha = r.get("fecha_public")
    ubic = r.get("ubicacion") or "sin ubicación"

    if cuantia is not None:
        try:
            cuantia_txt = f"${float(cuantia):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        except Exception:
            cuantia_txt = str(cuantia)
    else:
        cuantia_txt = "no se especifica la cuantía"

    answer = (
        f"La licitación con ID {lic_id} fue registrada por {entidad}. "
        f"El objeto es: {objeto}. "
        f"La cuantía es {cuantia_txt} y la modalidad es {modalidad}. "
        f"El estado actual es {estado} y fue publicada en {ubic}"
    )
    if fecha:
        answer += f" el {fecha}."
    else:
        answer += "."
    return answer, sql


@lru_cache(maxsize=1)
def _get_table_info_string(db: SQLDatabase) -> str:
    return db.get_table_info(EXPOSED_TABLES)


def _create_sql_agent_executor() -> Any:
    db_engine = engine
    db = SQLDatabase(engine=db_engine, include_tables=EXPOSED_TABLES)

    table_info = _get_table_info_string(db)

    llm = ChatGoogleGenerativeAI(
        model=DEFAULT_MODEL,
        temperature=0,
        convert_system_message_to_human=True,
    )

    sys_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        business_context=BUSINESS_CONTEXT,
        table_info=table_info,
        audit_hint=AUDIT_MODE_HINT,
        narrative_hint=NARRATIVE_STYLE_HINT,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=sys_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content="{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    try:
        agent = create_sql_agent(
            llm=llm,
            db=db,
            agent_type="tool-calling",
            verbose=True,
            prompt=prompt,
            return_intermediate_steps=True,
        )
        return agent
    except TypeError:
        logger.warning("tool-calling sin return_intermediate_steps… reintentando")
        agent = create_sql_agent(
            llm=llm,
            db=db,
            agent_type="tool-calling",
            verbose=True,
            prompt=prompt,
        )
        return agent


_agent_executor: Any | None = None
_embedding_function: GoogleGenerativeAIEmbeddings | None = None
_agent_with_history: RunnableWithMessageHistory | None = None


def _get_components() -> Tuple[Any, GoogleGenerativeAIEmbeddings, RunnableWithMessageHistory]:
    global _agent_executor, _embedding_function, _agent_with_history
    if _agent_executor is None:
        _agent_executor = _create_sql_agent_executor()
    if _embedding_function is None:
        _embedding_function = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            output_dimensionality=EMBED_DIM,
        )
    if _agent_with_history is None:
        _agent_with_history = RunnableWithMessageHistory(
            _agent_executor,
            get_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )
    return _agent_executor, _embedding_function, _agent_with_history


def process_query(
    query_text: str,
    session_id: Optional[str] = None,
    debug: bool = False,
) -> Dict[str, Any]:
    try:
        q = (query_text or "").strip()
        if not q:
            return {"status": "error", "error": "La consulta está vacía."}

        if not session_id:
            session_id = EPHEMERAL_SESSION_ID

        logger.info("Consulta recibida (session=%s): %s", session_id, q)


        # 1) conteo por tema
        topic = _looks_count_by_topic(q)
        if topic:
            answer, sql = _handle_count_by_topic(engine, topic)
            resp = {"answer": answer, "sql_query": [sql]}
            if debug:
                resp["session_id"] = session_id
            return resp

        # 2) licitación por id
        lic_id = _looks_licitacion_by_id(q)
        if lic_id is not None:
            answer, sql = _handle_licitacion_by_id(engine, lic_id)
            resp = {"answer": _naturalize_answer(answer), "sql_query": [sql]}
            if debug:
                resp["session_id"] = session_id
            return resp

        # 3) fallback: agente
        agent_executor, embedding_function, agent_with_history = _get_components()

        formatted_input = q
        used_chunks: List[Dict] = []

        sem_needed = any(w in q.lower() for w in ["de qué trata", "similar", "contenido", "chunk", "texto"])
        if sem_needed:
            try:
                vec = embedding_function.embed_query(q)
                sql_chunks = text("""
                    SELECT lc.licitacion_id, lc.chunk_idx, lc.chunk_text
                    FROM public.licitacion_chunk lc
                    ORDER BY lc.embedding_vec <=> :vec
                    LIMIT 8
                """)
                with engine.begin() as conn:
                    rows = conn.execute(sql_chunks, {
                        "vec": "[" + ",".join(f"{x:.8f}" for x in vec) + "]"
                    }).mappings().all()
                used_chunks = [dict(r) for r in rows]
                support = "\n\n".join(
                    f"(lic {c['licitacion_id']} • chunk {c['chunk_idx']}) {c['chunk_text'][:700]}"
                    for c in used_chunks
                )
                formatted_input = f"{q}\n\nCONTEXT_CHUNKS:\n{support}"
            except Exception as e:
                logger.warning("Fallo retrieval semántico: %s", e)

        result = agent_with_history.invoke(
            {"input": formatted_input},
            config={"configurable": {"session_id": session_id}},
        )

        answer_raw = result.get("output") if isinstance(result, dict) else result
        answer_text = _to_text(answer_raw).strip() or "No pude derivar una respuesta válida desde la base de datos."
        sql_steps = _extract_sql_steps(result)

        if used_chunks and "Evidencia" not in answer_text:
            ev = "\n".join(
                f"- licitacion_id={c['licitacion_id']}, chunk_idx={c['chunk_idx']}"
                for c in used_chunks
            )
            answer_text += f"\n\nEvidencia:\n{ev}"

        natural = _naturalize_answer(answer_text)

        resp = {"answer": natural, "sql_query": sql_steps}
        if debug:
            resp["raw_answer"] = answer_text
            resp["session_id"] = session_id
        return resp

    except Exception as e:
        logger.error("Error crítico en process_query: %s", str(e), exc_info=True)
        return {
            "status": "error",
            "error": "Error procesando la consulta",
            "details": str(e),
        }
