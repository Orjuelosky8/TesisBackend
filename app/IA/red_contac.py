from __future__ import annotations

import unicodedata
import os
import json
import logging
import re
from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI

from IA.memory import get_history, EPHEMERAL_SESSION_ID

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if "GEMINI_API_KEY" not in os.environ:
    raise ValueError("La variable de entorno GEMINI_API_KEY no está configurada.")

DEFAULT_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")


# ========== NODOS / LABELS ==========
GRAPH_LABELS_MAP = {
    # Licitaciones
    "licitacion": "Licitacion",
    "licitación": "Licitacion",
    "licitaciones": "Licitacion",

    # Proveedores / contratistas
    "proveedor": "Proveedor",
    "proveedores": "Proveedor",
    "oferente": "Proveedor",
    "oferentes": "Proveedor",
    "contratista": "Contratista",
    "contratistas": "Contratista",
    "adjudicatario": "Proveedor",
    "adjudicatarios": "Proveedor",

    # Entidades
    "entidad": "Entidad",
    "entidades": "Entidad",

    # Ubicaciones
    "ubicacion": "Ubicacion",
    "ubicación": "Ubicacion",
    "ubicaciones": "Ubicacion",
    "ciudad": "Ubicacion",
    "ciudades": "Ubicacion",
    "departamento": "Ubicacion",
    "departamentos": "Ubicacion",

    # Categorías
    "categoria": "Categoria",
    "categoría": "Categoria",
    "categorias": "Categoria",
    "categorías": "Categoria",

    # Eventos / documentos / montos
    "evento": "Evento",
    "eventos": "Evento",
    "documento": "Documento",
    "documentos": "Documento",
    "monto": "Monto",
    "montos": "Monto",

    # Personas
    "persona": "Persona",
    "personas": "Persona",
}

# Forma canónica (plural, en minúscula) para el prompt
GRAPH_LABEL_CANON = {
    "Licitacion": "licitaciones",
    "Proveedor": "proveedores",
    "Contratista": "contratistas",
    "Entidad": "entidades",
    "Ubicacion": "ubicaciones",
    "Categoria": "categorias",
    "Evento": "eventos",
    "Documento": "documentos",
    "Monto": "montos",
    "Persona": "personas",
}

# Tipos de relación relevantes (solo contexto, para el prompt del LLM)
CORRELATION_CANON = {
    ("Licitacion", "Proveedor"):  ["OFERTADO_POR", "ADJUDICADO_A"],
    ("Licitacion", "Contratista"): ["ADJUDICADO_A"],
    ("Licitacion", "Entidad"):     ["PUBLICADO_POR"],
    ("Licitacion", "Ubicacion"):   ["SE_EJECUTA_EN"],
    ("Licitacion", "Categoria"):   ["CLASIFICADO_COMO"],
    ("Licitacion", "Evento"):      ["TIENE_EVENTO"],
    ("Licitacion", "Documento"):   ["TIENE_DOCUMENTO"],
}


# ========== CONTEXTO PARA EL LLM (OPCIONAL) ==========

BUSINESS_CONTEXT = """
Contexto de negocio (auditoría analítica de licitaciones):

Eres un agente de IA que ayuda a construir consultas de grafos sobre Neo4j.
Los nodos principales son: licitaciones, proveedores, contratistas, entidades,
ubicaciones, categorías, eventos, documentos, montos y personas.

Te interesa especialmente cómo se relacionan entre sí según estos tipos de relación:
{correlations}
""".format(correlations=CORRELATION_CANON)


NARRATIVE_STYLE_HINT = """
Tu objetivo es acompañar al usuario para terminar construyendo un único JSON
que servirá como cuerpo de la petición POST a /graphs/call-in.

El JSON DEBE tener exactamente estas claves:
- "prompt": string
- "depth": número (1 o 2)
- "limitNodes": número entero
- "limitEdges": número entero
- "anchor": objeto o null

Cuando ya tengas todo claro, la respuesta FINAL debe ser SOLO el JSON,
sin explicación alrededor.

Ejemplos de JSON válidos:

{"prompt":"licitaciones + proveedores anio:2024 ciudad:Medellin","depth":1,"limitNodes":200,"limitEdges":400}

{"prompt":"licitaciones + proveedores anio:2025 departamento:Antioquia","depth":2,"limitNodes":300,"limitEdges":800}

{"prompt":"licitaciones + proveedores ciudad:Medellín","depth":1,"limitNodes":150,"limitEdges":400}

{"prompt":"licitaciones + proveedores departamento:Valle del Cauca","depth":2,"limitNodes":250,"limitEdges":800}

{"prompt":"licitaciones + categorias + proveedores catcode:85101500","depth":2,"limitNodes":220,"limitEdges":900}

{"prompt":"licitaciones + proveedores + entidades anio:2023 ciudad:Bogota","depth":2,"limitNodes":260,"limitEdges":1000}

{"prompt":"licitaciones + ubicaciones + documentos anio:2022 departamento:Cundinamarca","depth":2,"limitNodes":240,"limitEdges":1000}

{"prompt":"licitaciones + eventos ciudad:Cali","depth":2,"limitNodes":180,"limitEdges":700}

{"prompt":"licitaciones + proveedores","depth":1,"limitNodes":120,"limitEdges":300}

{"prompt":"licitaciones + proveedores + entidades + ubicaciones","depth":2,"limitNodes":300,"limitEdges":1000}

{"prompt":"licitaciones + proveedores anio:2024 ciudad:Bucaramanga catcode:43230000","depth":2,"limitNodes":220,"limitEdges":850}

{"prompt":"licitaciones + proveedores","depth":2,"limitNodes":300,"limitEdges":1000,"anchor":{"label":"Licitacion","id":"1084"}}

{"prompt":"licitaciones + proveedores anio:2021","depth":1,"limitNodes":180,"limitEdges":500}

{"prompt":"licitaciones + entidades + proveedores ciudad:Barranquilla","depth":2,"limitNodes":240,"limitEdges":900}
"""

AUDIT_MODE_HINT = """
Si el usuario parece no hablar de grafos, explícale brevemente que este asistente
solo sirve para construir el JSON de grafo y sugiérele usar la otra pestaña
de consultas si quiere estadísticas o detalles tabulares.
"""

SYSTEM_PROMPT_TEMPLATE = """
Eres un asistente experto en Neo4j que ayuda a construir consultas de grafos.

OBJETIVO (OBLIGATORIO):
- A partir del diálogo con el usuario, debes terminar generando UN único objeto JSON.
- Ese JSON será el cuerpo exacto que se enviará al endpoint /graphs/call-in.
- Cuando generes el JSON final, no añadas explicaciones ni texto alrededor: solo el JSON.

REGLAS PARA EL JSON:
- "prompt": lista de nodos principales separados por " + " (el primero es el nodo padre o base),
  seguida de filtros escritos como pares clave:valor separados por espacios.
  Ejemplos de nodos válidos (usa la forma plural canónica):
  {label_examples}

- "depth": profundidad de exploración del grafo.
  Usa 1 para un solo salto (conexiones directas) y 2 para dos saltos (vecindario amplio).

- "limitNodes": límite máximo de nodos (usa valores típicos entre 120 y 300;
  si el usuario no especifica, elige uno razonable).

- "limitEdges": límite máximo de aristas (usa valores típicos entre 300 y 1000;
  si el usuario no especifica, elige uno razonable).

- "anchor": normalmente null. Si el usuario menciona una licitación específica
  (por ejemplo "licitación 1084" o "id 1084"), usa:
  "anchor": {{"label":"Licitacion","id":"1084"}}
  (con el id en texto).

FILTROS:
- año -> "anio:2024"
- ciudad -> "ciudad:Medellin"
- departamento -> "departamento:Antioquia"
- categoría/código -> "catcode:85101500"
- nombre de persona -> "nombre:Laura"

Usa SIEMPRE este formato de filtros: clave:valor sin espacios en la clave,
y permite espacios en el valor (por ejemplo "departamento:Valle del Cauca").

{business_context}

{audit_hint}

{narrative_hint}
""".format(
    label_examples=", ".join(sorted(set(GRAPH_LABEL_CANON.values()))),
    business_context=BUSINESS_CONTEXT,
    audit_hint=AUDIT_MODE_HINT,
    narrative_hint=NARRATIVE_STYLE_HINT,
)


# ========== HELPERS COMUNES ==========

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


def _norm_es(txt: str) -> str:
    """Normaliza texto en español: minúsculas, sin acentos."""
    t = unicodedata.normalize("NFD", txt.lower())
    return "".join(ch for ch in t if unicodedata.category(ch) != "Mn")


# ========== PARSER DETERMINISTA (SIN LLM) ==========

def _parse_graph_intent(
    query_text: str,
    default_depth: int = 2,
    default_limit_nodes: int = 300,
    default_limit_edges: int = 1000,
) -> Dict[str, Any]:
    """
    A partir de un texto (ya con palabras tipo 'licitaciones', 'proveedores',
    'anio:2024', 'ciudad:Medellin', etc.) construye el payload JSON:

    {
      "prompt": "licitaciones + proveedores anio:2024 ciudad:Medellin nombre:Laura",
      "depth": 2,
      "limitNodes": 300,
      "limitEdges": 1000,
      "anchor": null
    }
    """
    raw = (query_text or "").strip()
    if not raw:
        raise ValueError("El texto de la consulta está vacío para modo grafo.")

    q_norm = _norm_es(raw)

    # 1) Detectar labels en orden de aparición
    label_positions: Dict[str, int] = {}
    for key, lbl in GRAPH_LABELS_MAP.items():
        m = re.search(rf"\b{re.escape(key)}\b", q_norm)
        if m:
            pos = m.start()
            if lbl not in label_positions or pos < label_positions[lbl]:
                label_positions[lbl] = pos

    labels: List[str] = sorted(label_positions.keys(), key=lambda l: label_positions[l])

    # Si no encontramos nada, asumimos licitaciones
    if not labels:
        labels = ["Licitacion"]

    # 2) Filtros clásicos: año, ciudad, departamento, catcode
    filters: Dict[str, Any] = {}

    m_year = re.search(r"(?:anio|año)\s*[:=]\s*(\d{4})", q_norm)
    if m_year:
        filters["anio"] = int(m_year.group(1))

    m_dep = re.search(r"departamento\s*[:=]\s*([a-z0-9\-\s]+)", q_norm)
    if m_dep:
        filters["departamento"] = m_dep.group(1).strip()

    m_city = re.search(r"ciudad\s*[:=]\s*([a-z0-9\-\s]+)", q_norm)
    if m_city:
        filters["ciudad"] = m_city.group(1).strip()

    m_cat = re.search(r"(?:categoria|categoría|codigo|código|catcode)\s*[:=]\s*([a-z0-9\.\-]+)", q_norm)
    if m_cat:
        filters["catcode"] = m_cat.group(1).strip()

    # Filtro por nombre (personas), tomado del texto original para respetar mayúsculas/acentos
    m_nombre_raw = re.search(
        r"(?i)nombre\s*[:=]\s*([A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9\-\s]+)",
        raw,
    )
    if m_nombre_raw:
        filters["nombre"] = m_nombre_raw.group(1).strip()

    # 3) Anchor por licitación específica (ej: licitacion 1084)
    anchor = None
    m_lic_id = re.search(r"licitaci[oó]n\s*#?\s*(\d+)", q_norm)
    if m_lic_id:
        lic_id = m_lic_id.group(1)
        anchor = {"label": "Licitacion", "id": str(lic_id)}

    # 4) Heurística de profundidad y límites
    if len(labels) == 1:
        depth = 1
    else:
        depth = default_depth

    depth = max(1, min(depth, 2))
    limit_nodes = max(10, min(default_limit_nodes, 20000))
    limit_edges = max(10, min(default_limit_edges, 50000))

    # 5) Construir el "prompt" que entiende /graphs/call-in
    label_tokens = [
        GRAPH_LABEL_CANON.get(lbl, lbl.lower())
        for lbl in labels
    ]
    prompt_parts: List[str] = [" + ".join(label_tokens)]

    if "anio" in filters:
        prompt_parts.append(f"anio:{filters['anio']}")
    if "departamento" in filters:
        prompt_parts.append(f"departamento:{filters['departamento']}")
    if "ciudad" in filters:
        prompt_parts.append(f"ciudad:{filters['ciudad']}")
    if "catcode" in filters:
        prompt_parts.append(f"catcode:{filters['catcode']}")
    if "nombre" in filters:
        prompt_parts.append(f"nombre:{filters['nombre']}")

    final_prompt = " ".join(prompt_parts).strip()

    payload: Dict[str, Any] = {
        "prompt": final_prompt,
        "depth": depth,
        "limitNodes": limit_nodes,
        "limitEdges": limit_edges,
        "anchor": anchor,
    }
    return payload


# ========== AGENTE LLM (OPCIONAL, PARA LENGUAJE NATURAL) ==========

_graph_agent: Optional[RunnableWithMessageHistory] = None


def _get_graph_agent() -> RunnableWithMessageHistory:
    """
    Crea (una sola vez) un chain LLM + memoria que, dado texto libre del usuario,
    devuelve un JSON con el formato requerido. Si no quieres usar LLM, puedes
    NO llamar a esto y usar solo _parse_graph_intent.
    """
    global _graph_agent
    if _graph_agent is None:
        llm = ChatGoogleGenerativeAI(
            model=DEFAULT_MODEL,
            temperature=0,
            convert_system_message_to_human=True,
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=SYSTEM_PROMPT_TEMPLATE),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessage(content="{input}"),
            ]
        )
        chain = prompt | llm
        _graph_agent = RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )
    return _graph_agent


# ========== ENTRYPOINT DEL ENDPOINT NUEVO ==========

def process_query_graph(
    query_text: str,
    session_id: Optional[str] = None,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Endpoint NUEVO para modo grafo.

    Devuelve SIEMPRE un JSON del estilo:

    {
      "prompt": "...",
      "depth": 1/2,
      "limitNodes": N,
      "limitEdges": M,
      "anchor": {...} o null
    }
    """
    try:
        q = (query_text or "").strip()
        if not q:
            return {"status": "error", "error": "La consulta está vacía."}

        if not session_id:
            session_id = EPHEMERAL_SESSION_ID

        logger.info("Consulta grafo recibida (session=%s): %s", session_id, q)

        # 1) Intento primero con el parser determinista (si ya viene más o menos estructurado)
        try:
            payload = _parse_graph_intent(q)
            if debug:
                payload["_source"] = "parser"
            return payload
        except Exception:
            logger.info("El parser directo no pudo construir el JSON, delegando al LLM.")

        # 2) Si no se pudo, usamos el LLM para que construya el JSON
        agent = _get_graph_agent()
        result = agent.invoke(
            {"input": q},
            config={"configurable": {"session_id": session_id}},
        )
        answer_text = _to_text(result).strip()

        try:
            payload = json.loads(answer_text)
        except json.JSONDecodeError as e:
            logger.error("El LLM no devolvió un JSON válido: %s; texto=%r", e, answer_text)
            # Último fallback: intentar parser sobre la salida del modelo
            payload = _parse_graph_intent(answer_text)

        if debug:
            payload["_source"] = payload.get("_source", "llm")
            payload["_raw"] = answer_text
            payload["_session_id"] = session_id

        return payload

    except Exception as e:
        logger.error("Error crítico en process_query (grafo): %s", str(e), exc_info=True)
        return {
            "status": "error",
            "error": "Error procesando la consulta de grafo",
            "details": str(e),
        }


# ==========
# ENTRYPOINT principal
# ==========

def process_query_sql(
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

        # =========================
        # 0) MODO GRAFO (Neo4j)
        # =========================
        # Si el texto empieza por "grafo:" lo interpretamos como
        # intención de construir el JSON para /graphs/call-in.
        if q.lower().startswith("grafo:"):
            graph_text = q.split(":", 1)[1].strip()
            try:
                graph_payload = _parse_graph_intent(graph_text)
            except ValueError as ve:
                return {
                    "status": "error",
                    "error": f"No se pudo interpretar la consulta para grafo: {ve}",
                }

            # Respuesta especial de modo grafo:
            # - mode = "graph": no se hizo SQL.
            # - graph_request: payload listo para enviarse a /graphs/call-in.
            resp: Dict[str, Any] = {
                "mode": "graph",
                "graph_request": graph_payload,
            }
            if debug:
                resp["session_id"] = session_id
            return resp

        # =========================
        # 1) conteo por tema (modo SQL)
        # =========================
        topic = _looks_count_by_topic(q)
        if topic:
            answer, sql = _handle_count_by_topic(engine, topic)
            resp = {"answer": answer, "sql_query": [sql]}
            if debug:
                resp["session_id"] = session_id
            return resp

        # 2) licitación por id (modo SQL)
        lic_id = _looks_licitacion_by_id(q)
        if lic_id is not None:
            answer, sql = _handle_licitacion_by_id(engine, lic_id)
            resp = {"answer": _naturalize_answer(answer), "sql_query": [sql]}
            if debug:
                resp["session_id"] = session_id
            return resp

        # 3) fallback: agente SQL + retrieval semántico
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
