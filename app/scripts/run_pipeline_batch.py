# app/scripts/run_pipeline_batch.py
import argparse, os, sys, time, json, requests


def load_holidays(path: str):
    holidays = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                holidays.append(line)
    return holidays


def run_batches(
    base,
    flow="all",
    batch=500,
    start_id=0,
    where_extra=None,
    sleep=0.2,
    timeout=60,
    holidays=None,   # 👈 nuevo
):
    last_id = start_id
    total = 0
    session = requests.Session()

    while True:
        where = f"id > {last_id}"
        if where_extra:
            where = f"({where}) AND ({where_extra})"

        payload = {
            "flow": flow,
            "where": where,
            "limit": batch,
        }

        # 👇 si hay festivos, los mandamos al backend
        if holidays:
            payload["holidays"] = holidays

        r = session.post(f"{base}/pipelines/batch", json=payload, timeout=timeout)
        r.raise_for_status()
        items = r.json()

        if not items:
            print(f"[done] No hay más filas (total procesadas: {total}).")
            break

        # Avanza el cursor con el mayor licitacion_id devuelto
        last_id = max(x.get("licitacion_id", last_id) for x in items)
        total += len(items)
        print(f"[ok] batch={len(items)} last_id={last_id} total={total}")

        if sleep:
            time.sleep(sleep)

    return total


def run_red_contactos(base, lic_ids, json_file, timeout=60):
    """
    Envía payload a /pipes/red-contactos/run
    json_file debe contener algo como:
    {
      "personas": [...],
      "contratistas": [...]
    }
    """
    import os, json, requests

    if not json_file or not os.path.exists(json_file):
        print(f"[error] No existe --json-file: {json_file}")
        return 2

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    payload = {
        "licitacion_ids": lic_ids,
        "data": data
    }

    url = f"{base}/pipes/red-contactos/run"
    r = requests.post(url, json=payload, timeout=timeout)

    if r.status_code >= 400:
        print(f"[server-error {r.status_code}] {url}")
        # Imprime lo que devolvió FastAPI (a veces es JSON, a veces HTML)
        print(r.text)
        return 1

    items = r.json()
    print(json.dumps(items, ensure_ascii=False, indent=2))
    print(f"[resumen] total procesadas: {len(items)}")
    return 0



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.getenv("BASE_URL", "http://localhost:8000"))
    ap.add_argument("--flow", default="all", help="all | red_precio | gap_fechas | red_contactos")
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--start-id", type=int, default=0)
    ap.add_argument(
        "--where-extra",
        default=None,
        help="Condición SQL adicional, p.ej. EXISTS(...) sobre keymap/calendario",
    )
    ap.add_argument("--timeout", type=int, default=60)
    # ---- parámetros para red_contactos ----
    ap.add_argument("--json-file", default=None, help="Ruta a JSON con 'personas' y opcional 'contratistas'")
    ap.add_argument("--ids", default=None, help="Lista de IDs de licitaciones (ej: 101,102,105)")

    # 👇 NUEVO
    ap.add_argument(
        "--holidays-file",
        default=None,
        help="Archivo de festivos (uno por línea, YYYY-MM-DD)",
    )

    args = ap.parse_args()

    # Quick health
    try:
        h = requests.get(f"{args.base}/health", timeout=args.timeout)
        h.raise_for_status()
        print("[health]", h.json())
    except Exception as e:
        print(f"[error] No conecta con {args.base}: {e}")
        sys.exit(2)

    # Si es el flujo especial
    if args.flow == "red_contactos":
        if not args.ids:
            print("[error] Para red_contactos debes pasar --ids=1,2,3 y --json-file=path.json")
            sys.exit(2)
        lic_ids = [int(x.strip()) for x in args.ids.split(",") if x.strip().isdigit()]
        code = run_red_contactos(args.base, lic_ids, args.json_file, timeout=args.timeout)
        sys.exit(code)

    # 👇 si hay archivo de festivos, lo cargamos
    holidays = None
    if args.holidays_file:
        holidays = load_holidays(args.holidays_file)
        print(f"[info] Festivos cargados: {len(holidays)}")

    # Flujos computables (batch)
    total = run_batches(
        base=args.base,
        flow=args.flow,
        batch=args.batch,
        start_id=args.start_id,
        where_extra=args.where_extra,
        timeout=args.timeout,
        holidays=holidays,   # 👈 lo pasamos
    )
    print(f"[resumen] total procesadas: {total}")


if __name__ == "__main__":
    main()
