
import os, json, time, argparse, sys
try:
    from confluent_kafka import Consumer, KafkaException
    _HAVE_KAFKA = True
except Exception:
    Consumer = None
    KafkaException = Exception
    _HAVE_KAFKA = False

from services.ml.ml_utils import label_from_window as resolve_window_label
from services.ml.ml_utils import record_to_window_text, resolve_record_label_quality_tier

BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "127.0.0.1:9092")
IN_TOPIC  = os.getenv("IN_TOPIC", "unsw_nb15.windowed")
GROUP_ID  = os.getenv("GROUP_ID", f"dataset-export-{int(time.time())}")
OFFSET_RESET = os.getenv("OFFSET_RESET", "earliest")  # export usually wants history
THRESH_FAIL_NO_MFA = int(os.getenv("THRESH_FAIL_NO_MFA", "3"))


def window_to_text(w: dict) -> str:
    return record_to_window_text(w)[1]

def label_from_window(w: dict, threshold: int) -> int:
    return resolve_window_label(w, threshold)[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.getenv("OUT_PATH", "data/windows_dataset.jsonl"))
    ap.add_argument("--max", type=int, default=int(os.getenv("MAX_WINDOWS", "5000")))
    ap.add_argument("--seconds", type=int, default=int(os.getenv("MAX_SECONDS", "0")),
                   help="Stop after N seconds (0 = no time limit).")
    ap.add_argument("--input-file", default=None, help="Path to file with window JSON lines (one per line). Use '-' for stdin. If not set, reads from Kafka topic.")
    args = ap.parse_args()

    start = time.time()
    written = 0

    if args.input_file is not None:
        # read from file/stdin
        if args.input_file == "-":
            stream = sys.stdin
        else:
            stream = open(args.input_file, "r", encoding="utf-8")

        with open(args.out, "w", encoding="utf-8") as f:
            try:
                for line in stream:
                    if args.seconds and (time.time() - start) >= args.seconds:
                        break
                    if written >= args.max:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        window = json.loads(line)
                    except Exception:
                        continue

                    label, label_source = resolve_window_label(window, THRESH_FAIL_NO_MFA)
                    row = {
                        "id": f"{window.get('window_start_ms')}-{window.get('window_end_ms')}",
                        "window_start_ms": window.get("window_start_ms"),
                        "window_end_ms": window.get("window_end_ms"),
                        "text": window_to_text(window),
                        # Label comes from simulation metadata when present, otherwise the legacy heuristic.
                        "label": label,
                        "label_source": label_source,
                        "label_quality_tier": resolve_record_label_quality_tier({"label_source": label_source}),
                        # Keep raw payload for debugging / future feature work
                        "window": window,
                    }

                    f.write(json.dumps(row) + "\n")
                    written += 1

            except KeyboardInterrupt:
                pass
            finally:
                if args.input_file != "-":
                    stream.close()

        print(f"Wrote {written} rows to {args.out}")
        return

    # Default: read from Kafka topic as before
    consumer = Consumer({
        "bootstrap.servers": BOOTSTRAP,
        "group.id": GROUP_ID,
        "auto.offset.reset": OFFSET_RESET,
        "enable.auto.commit": False,
    })
    consumer.subscribe([IN_TOPIC])

    with open(args.out, "w", encoding="utf-8") as f:
        try:
            while True:
                if args.seconds and (time.time() - start) >= args.seconds:
                    break
                if written >= args.max:
                    break

                msg = consumer.poll(1.0)
                if msg is None:
                    continue
                if msg.error():
                    raise KafkaException(msg.error())

                window = json.loads(msg.value().decode("utf-8"))

                label, label_source = resolve_window_label(window, THRESH_FAIL_NO_MFA)
                row = {
                    "id": f"{window.get('window_start_ms')}-{window.get('window_end_ms')}",
                    "window_start_ms": window.get("window_start_ms"),
                    "window_end_ms": window.get("window_end_ms"),
                    "text": window_to_text(window),
                    # Label comes from simulation metadata when present, otherwise the legacy heuristic.
                    "label": label,
                    "label_source": label_source,
                    "label_quality_tier": resolve_record_label_quality_tier({"label_source": label_source}),
                    # Keep raw payload for debugging / future feature work
                    "window": window,
                }

                f.write(json.dumps(row) + "\n")
                written += 1

                # commit after writing so restart doesn't duplicate as much
                consumer.commit(message=msg, asynchronous=False)

        except KeyboardInterrupt:
            pass
        finally:
            consumer.close()

    print(f"Wrote {written} rows to {args.out}")

if __name__ == "__main__":
    main()

