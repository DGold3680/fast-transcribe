import asyncio
import json
import os
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from vosk import KaldiRecognizer, Model, SetLogLevel

MODEL_PATH = os.environ.get("VOSK_MODEL_PATH", "../vosk-model-small-en-us-0.15")
SAMPLE_RATE_DEFAULT = 16000

if not os.path.isdir(MODEL_PATH):
  raise RuntimeError(f"Vosk model directory not found at {MODEL_PATH}")

SetLogLevel(0)
MODEL = Model(MODEL_PATH)

app = FastAPI(title="Cloudroom Transcription Service")


def create_recognizer(sample_rate: int) -> KaldiRecognizer:
  recognizer = KaldiRecognizer(MODEL, sample_rate)
  recognizer.SetWords(True)
  return recognizer


@app.get("/health")
async def health() -> JSONResponse:
  return JSONResponse({"status": "ok"})


@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket) -> None:
  await websocket.accept()

  recognizer: Optional[KaldiRecognizer] = None
  participant_identity = "unknown"
  room_id = "unknown"

  try:
    init = await websocket.receive_text()
    init_payload = json.loads(init)
    sample_rate = int(init_payload.get("sampleRate", SAMPLE_RATE_DEFAULT))
    participant_identity = init_payload.get("participantIdentity", participant_identity)
    room_id = init_payload.get("roomId", room_id)
    print(f"[vosk-service] Initialization complete: sample_rate={sample_rate}, participant_identity={participant_identity}, room_id={room_id}")

    recognizer = create_recognizer(sample_rate)

    while True:
      message = await websocket.receive()

      if message.get("type") == "websocket.disconnect":
        break

      if "text" in message and message["text"] is not None:
        try:
          data = json.loads(message["text"])
        except json.JSONDecodeError:
          continue

        if data.get("type") == "stop":
          break
        continue

      audio_bytes = message.get("bytes")
      if audio_bytes is None:
        continue

      print(
        f"[vosk-service] Received audio chunk of {len(audio_bytes)} bytes from {participant_identity} (room={room_id})"
      )
      if recognizer.AcceptWaveform(audio_bytes):
        print(
          f"[vosk-service] Received {len(audio_bytes)} bytes from {participant_identity} (room={room_id})"
        )
        result = json.loads(recognizer.Result())
        text = result.get("text", "")
        if text:
          print(
            f"[vosk-service] Final segment room={room_id} participant={participant_identity} text={text}"
          )
          await websocket.send_json(
            {
              "type": "segment",
              "text": text,
              "confidence": result.get("confidence"),
              "participantIdentity": participant_identity,
              "roomId": room_id,
            }
          )
      else:
        print(
          f"[vosk-service] Received {len(audio_bytes)} bytes from {participant_identity} (room={room_id})"
        )
        partial = json.loads(recognizer.PartialResult()).get("partial")
        if partial:
          print(
            f"[vosk-service] Partial transcript room={room_id} participant={participant_identity} text={partial}"
          )
          await websocket.send_json(
            {
              "type": "partial",
              "text": partial,
              "participantIdentity": participant_identity,
              "roomId": room_id,
            }
          )
  except WebSocketDisconnect:
    pass
  except Exception as exc:
    try:
      await websocket.send_json({"type": "error", "message": str(exc)})
    except Exception:
      pass
  finally:
    if recognizer:
      final = json.loads(recognizer.FinalResult())
      text = final.get("text")
      if text:
        await websocket.send_json(
          {
            "type": "segment",
            "text": text,
            "confidence": final.get("confidence"),
            "participantIdentity": participant_identity,
            "roomId": room_id,
          }
        )
    try:
      await websocket.close()
    except Exception:
      pass


if __name__ == "__main__":
  uvicorn.run(
    "main:app",
    host=os.environ.get("HOST", "0.0.0.0"),
    port=int(os.environ.get("PORT", "4000")),
    reload=False,
  )

