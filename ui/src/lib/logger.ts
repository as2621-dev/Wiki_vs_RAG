/**
 * Minimal structured JSON logger (per CLAUDE.md logging rules).
 * Emits `{event, ...fields}` to the console so client-side events are traceable.
 */

type LogFields = Record<string, unknown>;

function emit(level: "info" | "error", event: string, fields: LogFields): void {
  const line = JSON.stringify({ level, event, ...fields });
  if (level === "error") {
    console.error(line);
  } else {
    console.info(line);
  }
}

export const logger = {
  info: (event: string, fields: LogFields = {}): void => emit("info", event, fields),
  error: (event: string, fields: LogFields = {}): void => emit("error", event, fields),
};
