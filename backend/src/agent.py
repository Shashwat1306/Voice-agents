import logging
import json
from pathlib import Path
from typing import Annotated

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
from typing import Dict, Optional

logger = logging.getLogger("agent")

load_dotenv(".env.local")

# Load tutor content
def load_tutor_content():
    """Load the tutor content from JSON file"""
    content_path = Path("shared-data/day4_tutor_content.json")
    with open(content_path, "r") as f:
        return json.load(f)

TUTOR_CONTENT = load_tutor_content()


# Load Razorpay SDR content for the Sales Development Representative agent
def load_razorpay_content():
    content_path = Path("shared-data/razorpay_sdr_content.json")
    if not content_path.exists():
        return {}
    with open(content_path, "r", encoding="utf-8") as f:
        return json.load(f)

RAZORPAY_CONTENT = load_razorpay_content()


class RouterAgent(Agent):
    """Initial agent that greets user and routes to the appropriate learning mode"""
    
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a friendly tutor assistant that helps students learn programming concepts through active recall.

Your role is to:
1. Greet the student warmly
2. Explain that you have three learning modes available:
   - LEARN mode: I'll explain concepts to you
   - QUIZ mode: I'll ask you questions to test your knowledge
   - TEACH-BACK mode: You explain concepts back to me and I'll give feedback

3. Ask which mode they'd like to start with

Available topics: variables, loops, functions, and conditionals.

Once they choose a mode, use the appropriate handoff tool to transfer them to that agent.

Keep your greeting brief and conversational.""",
        )

    @function_tool
    async def transfer_to_learn_mode(self, context: RunContext):
        """Transfer the student to Learn mode where concepts are explained"""
        logger.info("Transferring to Learn mode")
        return LearnAgent(), "Transferring you to Learn mode..."

    @function_tool
    async def transfer_to_quiz_mode(self, context: RunContext):
        """Transfer the student to Quiz mode where they answer questions"""
        logger.info("Transferring to Quiz mode")
        return QuizAgent(), "Transferring you to Quiz mode..."

    @function_tool
    async def transfer_to_teach_back_mode(self, context: RunContext):
        """Transfer the student to Teach-Back mode where they explain concepts"""
        logger.info("Transferring to Teach-Back mode")
        return TeachBackAgent(), "Transferring you to Teach-Back mode..."


class LearnAgent(Agent):
    """Agent that explains concepts to the student (Matthew voice)"""
    
    def __init__(self) -> None:
        # Create content reference for instructions
        concepts_list = "\n".join([f"- {c['title']}: {c['summary']}" for c in TUTOR_CONTENT])
        
        super().__init__(
            instructions=f"""You are Matthew, a patient and clear teacher in LEARN mode. Your job is to explain programming concepts clearly.

Available concepts:
{concepts_list}

Your approach:
1. When the student arrives, welcome them to Learn mode
2. Ask which concept they'd like to learn about (variables, loops, functions, or conditionals)
3. Explain the concept clearly using the summary provided
4. Use simple examples and analogies
5. Check if they have questions
6. Offer to explain another concept or switch modes

If the student wants to switch modes:
- Use transfer_to_quiz_mode to switch to Quiz mode
- Use transfer_to_teach_back_mode to switch to Teach-Back mode

Keep explanations conversational and easy to understand. Break complex ideas into simple parts.""",
            tts=murf.TTS(
                voice="en-US-matthew",
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True
            ),
        )

    @function_tool
    async def transfer_to_quiz_mode(self, context: RunContext):
        """Transfer to Quiz mode"""
        logger.info("Transferring from Learn to Quiz mode")
        return QuizAgent(), "Switching to Quiz mode..."

    @function_tool
    async def transfer_to_teach_back_mode(self, context: RunContext):
        """Transfer to Teach-Back mode"""
        logger.info("Transferring from Learn to Teach-Back mode")
        return TeachBackAgent(), "Switching to Teach-Back mode..."


class QuizAgent(Agent):
    """Agent that quizzes the student (Alicia voice)"""
    
    def __init__(self) -> None:
        # Create questions reference
        questions_list = "\n".join([f"- {c['title']}: {c['sample_question']}" for c in TUTOR_CONTENT])
        
        super().__init__(
            instructions=f"""You are Alicia, an encouraging quiz master in QUIZ mode. Your job is to test the student's knowledge.

Available quiz topics:
{questions_list}

Your approach:
1. When the student arrives, welcome them to Quiz mode
2. Ask which concept they'd like to be quizzed on
3. Ask the sample question for that concept
4. Listen to their answer
5. Provide feedback: point out what they got right and gently correct any mistakes
6. Offer to quiz them on another concept or switch modes

If the student wants to switch modes:
- Use transfer_to_learn_mode to switch to Learn mode
- Use transfer_to_teach_back_mode to switch to Teach-Back mode

Be encouraging and supportive. Celebrate correct answers and help them understand mistakes.""",
            tts=murf.TTS(
                voice="en-US-alicia",
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True
            ),
        )

    @function_tool
    async def transfer_to_learn_mode(self, context: RunContext):
        """Transfer to Learn mode"""
        logger.info("Transferring from Quiz to Learn mode")
        return LearnAgent(), "Switching to Learn mode..."

    @function_tool
    async def transfer_to_teach_back_mode(self, context: RunContext):
        """Transfer to Teach-Back mode"""
        logger.info("Transferring from Quiz to Teach-Back mode")
        return TeachBackAgent(), "Switching to Teach-Back mode..."


class TeachBackAgent(Agent):
    """Agent that asks student to explain concepts back (Ken voice)"""
    
    def __init__(self) -> None:
        concepts_list = "\n".join([f"- {c['title']}" for c in TUTOR_CONTENT])
        
        super().__init__(
            instructions=f"""You are Ken, a thoughtful evaluator in TEACH-BACK mode. The best way to learn is to teach!

Available concepts:
{concepts_list}

Your approach:
1. When the student arrives, welcome them to Teach-Back mode
2. Explain that they'll teach YOU a concept - this is how they'll truly master it
3. Ask which concept they'd like to explain (variables, loops, functions, or conditionals)
4. Listen carefully to their explanation
5. Provide qualitative feedback:
   - What did they explain well?
   - What key points did they miss?
   - How could their explanation be clearer?
6. Give a rating (Excellent/Good/Needs Work)
7. Offer to hear another concept or switch modes

If the student wants to switch modes:
- Use transfer_to_learn_mode to switch to Learn mode
- Use transfer_to_quiz_mode to switch to Quiz mode

Be constructive and specific in your feedback. Help them become better teachers (and learners).""",
            tts=murf.TTS(
                voice="en-US-ken",
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True
            ),
        )

    @function_tool
    async def transfer_to_learn_mode(self, context: RunContext):
        """Transfer to Learn mode"""
        logger.info("Transferring from Teach-Back to Learn mode")
        return LearnAgent(), "Switching to Learn mode..."

    @function_tool
    async def transfer_to_quiz_mode(self, context: RunContext):
        """Transfer to Quiz mode"""
        logger.info("Transferring from Teach-Back to Quiz mode")
        return QuizAgent(), "Switching to Quiz mode..."


class SDRAgent(Agent):
    """Sales Development Representative agent for Razorpay demo."""

    def __init__(self) -> None:
        # basic lead fields to collect
        self.lead_fields = [
            "name",
            "company",
            "email",
            "role",
            "use_case",
            "team_size",
            "timeline"
        ]

        faqs = "\n".join([f"- {f['question']}: {f['answer']}" for f in RAZORPAY_CONTENT.get("faqs", [])])

        super().__init__(
            instructions=f"""You are a friendly Sales Development Representative for {RAZORPAY_CONTENT.get('company','Razorpay')}.\n
    Behavior and goals:\n
    1. Greet the visitor warmly.\n2. Ask what brought them here and what they're working on.\n+3. Keep the conversation focused on understanding the visitor's needs.\n4. Ask qualifying questions naturally to collect the following lead fields: name, company, email, role, use_case, team_size, timeline.\n+   - Ask for missing fields as the conversation flows; be polite and brief.\n5. When the user asks about the product, pricing, or other company details, consult the provided FAQ content (do not invent facts). Use `get_faq` tool for precise answers.\n6. Detect when the user is done (e.g., "That's all", "I'm done", "Thanks") and then:\n+   - Give a short verbal summary of the lead (who they are, what they want, rough timeline).\n+   - Save the collected lead data by calling the `save_lead` tool with a JSON object.\n+   - Thank them and offer next steps (e.g., "I'll connect you with our sales team").\n\n+Available FAQ entries:\n{faqs}\n\n+When collecting fields, prefer obtaining email and role early. Keep questions short and natural.""",
        )

    @function_tool
    async def get_faq(self, context: RunContext, query: str) -> Dict[str, Optional[str]]:
        """Simple keyword search over loaded FAQ entries. Returns the best matching FAQ or an empty result."""
        q = (query or "").lower()
        best = None
        best_score = 0
        for f in RAZORPAY_CONTENT.get("faqs", []):
            text = (f.get("question","") + " " + f.get("answer","") ).lower()
            # simple score: count of query words present
            score = sum(1 for w in q.split() if w and w in text)
            if score > best_score:
                best_score = score
                best = f
        if best is None and RAZORPAY_CONTENT:
            # fallback: return company overview
            return {"question": "overview", "answer": RAZORPAY_CONTENT.get("overview")}
        return best or {}

    @function_tool
    async def save_lead(self, context: RunContext, lead: Dict) -> str:
        """Append the lead dict to `KMS/logs/leads.json` and return the path."""
        out_path = Path("KMS/logs/leads.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        leads = []
        if out_path.exists():
            try:
                with open(out_path, "r", encoding="utf-8") as f:
                    leads = json.load(f)
            except Exception:
                leads = []
        leads.append(lead)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(leads, f, indent=2)
        logger.info(f"Saved lead to {out_path} -> {lead}")
        return str(out_path)


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
        agent=SDRAgent(),
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