import logging
import json
import os
import tempfile
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self) -> None:
        # Load previous check-ins (if any) to include a short memory reference in the instructions
        last_entry = self._load_last_entry()

        memory_hint = ""
        if last_entry:
            # keep this short and factual to avoid leakage of sensitive details
            last_dt = last_entry.get("timestamp")
            last_mood = last_entry.get("mood")
            last_energy = last_entry.get("energy")
            memory_hint = (
                f"Last check-in was on {last_dt}. They reported mood: '{last_mood}' and energy: '{last_energy}'. "
                "When appropriate, briefly reference this to compare today to last time."
            )

        super().__init__(
            instructions=(
                "You are a calm, supportive daily health & wellness voice companion. The user interacts with you by voice. "
                "You are NOT a clinician and must not offer medical diagnoses or medical advice. Your role is supportive: help the user reflect, set small daily intentions, and offer simple, practical, non-medical suggestions. "
                + memory_hint + "\n\n"
                "Primary check-in flow (keep it short and conversational):\n"
                "1) Greet briefly and ask about mood: 'How are you feeling today?' (allow open text or a simple scale).\n"
                "2) Ask about energy: 'What's your energy like today?'\n"
                "3) Ask about stressors: 'Anything stressing you out right now?' (optional).\n"
                "4) Ask for 1–3 objectives: 'What are 1–3 things you'd like to get done today?'\n"
                "5) Ask about self-care: 'Is there anything you want to do for yourself (rest, exercise, hobbies)?'\n"
                "6) Offer one or two short, actionable, non-medical suggestions tailored to their answers (e.g., break a large task into a smaller step, take a 5-minute walk, try a breathing exercise, schedule a short break). Keep suggestions realistic and brief.\n"
                "7) Close by repeating back a short recap: today's mood summary, the main 1–3 objectives, and ask 'Does this sound right?'.\n\n"
                "When the check-in finishes, call the tool 'save_checkin' to persist a brief structured summary of the session. The saved entry should include date/time, mood, energy, objectives, optional self-care, and a short agent-generated summary sentence.\n"
                "Keep responses concise and grounded; avoid long multi-paragraph replies. Use friendly, empathetic language and respect user privacy."
            )
        )

    @function_tool
    async def save_checkin(self, context: RunContext, mood: str, energy: str, objectives: str, self_care: str = ""):
        """Persist a wellness check-in to a single JSON file.

        Args:
            mood: Short text or scale describing mood.
            energy: Short text or scale describing energy.
            objectives: User-stated objectives (1-3 items, comma- or newline-separated).
            self_care: Optional self-care intention.
        Returns:
            A confirmation string for the user.
        """

        logger.info("Saving wellness check-in")

        # normalize objectives into a list
        objectives_list = [o.strip() for o in objectives.replace("\r", "\n").split("\n") if o.strip()]
        if len(objectives_list) == 1 and "," in objectives_list[0]:
            objectives_list = [o.strip() for o in objectives_list[0].split(",") if o.strip()]

        # Agent-generated short summary
        summary_parts = []
        if mood:
            summary_parts.append(f"mood: {mood}")
        if energy:
            summary_parts.append(f"energy: {energy}")
        if objectives_list:
            summary_parts.append(f"objectives: {', '.join(objectives_list[:3])}")

        agent_summary = " | ".join(summary_parts)

        entry = {
            "timestamp": datetime.now().isoformat(),
            "mood": mood,
            "energy": energy,
            "objectives": objectives_list,
            "self_care": self_care,
            "agent_summary": agent_summary,
        }

        # write to a single JSON file in the backend directory
        backend_dir = Path(__file__).resolve().parents[1]
        json_path = backend_dir / "wellness_log.json"

        # Load existing entries
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                    if not isinstance(existing, list):
                        existing = []
            except Exception:
                existing = []
        else:
            existing = []

        existing.append(entry)

        # atomic write
        with tempfile.NamedTemporaryFile("w", delete=False, dir=str(backend_dir), encoding="utf-8") as tmp:
            json.dump(existing, tmp, indent=2, ensure_ascii=False)
            tmp_name = tmp.name

        os.replace(tmp_name, str(json_path))

        logger.info(f"Check-in added to {json_path}")

        return "Thanks — I've saved this check-in. I'll remember it for next time."

    def _load_last_entry(self):
        try:
            backend_dir = Path(__file__).resolve().parents[1]
            json_path = backend_dir / "wellness_log.json"
            if not json_path.exists():
                return None
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list) and data:
                    return data[-1]
        except Exception:
            return None
        return None


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, AssemblyAI, and the LiveKit turn detector
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt=deepgram.STT(model="nova-3"),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm=google.LLM(
                model="gemini-2.5-flash",
            ),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts=murf.TTS(
                voice="en-US-matthew", 
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True
            ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    # To use a realtime model instead of a voice pipeline, use the following session setup instead.
    # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/))
    # 1. Install livekit-agents[openai]
    # 2. Set OPENAI_API_KEY in .env.local
    # 3. Add `from livekit.plugins import openai` to the top of this file
    # 4. Use the following session setup instead of the version above
    # session = AgentSession(
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    # Metrics collection, to measure pipeline performance
    # For more information, see https://docs.livekit.io/agents/build/metrics/
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))