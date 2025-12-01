import logging
import json
from pathlib import Path
from typing import Annotated, Optional, List, Dict
import random
from datetime import datetime
import uuid
from pydantic import Field

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
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class ImprovHostAgent(Agent):
    """Host for a single-player improv show called 'Improv Battle'."""

    def __init__(self) -> None:
        self.improv_state: Dict = {
            "player_name": None,
            "current_round": 0,
            "max_rounds": 3,
            "rounds": [],
            "phase": "intro",
        }

        # Small curated list of clear, character-focused scenarios
        self.scenarios: List[str] = [
            "You are a barista who has to tell a customer that their latte is actually a portal to another dimension.",
            "You are a time-travelling tour guide explaining modern smartphones to someone from the 1800s.",
            "You are a restaurant waiter who must calmly tell a customer that their order has escaped the kitchen.",
            "You are a customer trying to return an obviously cursed object to a very skeptical shop owner.",
            "You are a weather forecaster who must explain why it's raining rubber ducks today.",
        ]

        super().__init__(
            instructions=(
                "You are the host of a TV improv show called 'Improv Battle'.\n"
                "Adopt a high-energy, witty, and clear style. Explain rules briefly, set scenarios, listen, then react.\n"
                "Reactions should vary: sometimes amused, sometimes unimpressed, sometimes pleasantly surprised."
                " Be respectful and never abusive. Light teasing and constructive critique are allowed.\n"
                "When the show ends, summarize the player's improv strengths and thank them."
            )
        )

    def _choose_scenario(self) -> str:
        return random.choice(self.scenarios)

    def _generate_reaction(self, scenario: str, performance: str) -> str:
        # Choose tone
        tone = random.choices(["supportive", "neutral", "mildly_critical"], weights=[0.5, 0.3, 0.2])[0]

        # Pick a short excerpt to reference if available
        ref = None
        if performance:
            sentences = [s.strip() for s in performance.split(".") if s.strip()]
            if sentences:
                ref = sentences[0]

        if tone == "supportive":
            lead = "That was great — you really committed to the bit."
            if ref:
                lead += f" I especially loved '{ref}.'"
        elif tone == "neutral":
            lead = "Nicely played. You hit some strong choices there."
            if ref:
                lead += f" The line '{ref}.' gave a clear direction." 
        else:
            lead = "Interesting choices — a few things could land stronger."
            if ref:
                lead += f" Try leaning more into '{ref}.' next time to sell the premise."

        # Add a quick actionable tip
        tips = [
            "Try stretching the moment and committing to smaller details.",
            "Push the stakes higher — what's the worst that could happen?",
            "Play with your voice and physical choices to sell the character.",
            "You could have escalated the absurdity for bigger laughs.",
        ]

        tip = random.choice(tips)

        return f"{lead} {tip}"

    @function_tool
    async def get_improv_state(self):
        return self.improv_state

    @function_tool
    async def start_game(self, player_name: Annotated[Optional[str], Field(default=None, description="Contestant name")]=None, max_rounds: Annotated[int, Field(default=3)] = 3):
        # Initialize or reset state
        if player_name:
            self.improv_state['player_name'] = player_name
        elif not self.improv_state.get('player_name'):
            self.improv_state['player_name'] = "Contestant"

        self.improv_state['current_round'] = 0
        self.improv_state['max_rounds'] = max_rounds
        self.improv_state['rounds'] = []
        self.improv_state['phase'] = 'intro'

        intro = (
            f"Welcome to Improv Battle, {self.improv_state['player_name']}! "
            "Rules: I'll give you a scenario, you'll improvise in-character, then say 'End scene' when you're done. "
            f"We'll play {max_rounds} rounds. Ready? Let's begin!"
        )

        # Move to first scenario immediately
        return {"message": intro}

    @function_tool
    async def propose_scenario(self):
        # Move to next round and present scenario
        if self.improv_state['phase'] == 'done':
            return {"message": "The show is over. Thank you for playing!"}

        if self.improv_state['current_round'] >= self.improv_state['max_rounds']:
            # finalize
            self.improv_state['phase'] = 'done'
            return {"message": "We've finished all rounds."}

        self.improv_state['current_round'] += 1
        scenario = self._choose_scenario()
        round_entry = {"scenario": scenario, "host_reaction": None, "performance": None}
        self.improv_state['rounds'].append(round_entry)
        self.improv_state['phase'] = 'awaiting_improv'

        prompt = (
            f"Round {self.improv_state['current_round']} of {self.improv_state['max_rounds']}: {scenario} "
            "Go! Improvise in-character and say 'End scene' when finished."
        )

        return {"message": prompt, "scenario": scenario}

    @function_tool
    async def record_performance(self, performance_text: Annotated[Optional[str], Field(default=None, description="What the player said/acted")]=None):
        # If player name not set, try to set from first performance line
        if not self.improv_state.get('player_name') and performance_text:
            first_line = performance_text.strip().split('\n')[0]
            if first_line:
                # Take first two words as a simple name heuristic
                parts = first_line.split()
                if parts:
                    self.improv_state['player_name'] = parts[0]

        # Early exit phrases
        if performance_text:
            low = performance_text.lower()
            if any(p in low for p in ["stop game", "end show", "quit game", "stop show"]):
                self.improv_state['phase'] = 'done'
                return {"message": "Got it — ending the show. Thanks for playing!", "ended": True}

        # Find current round entry
        if not self.improv_state['rounds']:
            return {"message": "No active round. Call propose_scenario first."}

        cur_idx = self.improv_state['current_round'] - 1
        if cur_idx < 0:
            return {"message": "No active round. Call propose_scenario first."}

        # Save performance
        self.improv_state['rounds'][cur_idx]['performance'] = performance_text
        self.improv_state['phase'] = 'reacting'

        # Generate host reaction
        scenario = self.improv_state['rounds'][cur_idx]['scenario']
        reaction = self._generate_reaction(scenario, performance_text or "")
        self.improv_state['rounds'][cur_idx]['host_reaction'] = reaction

        # Move phase and check if finished
        if self.improv_state['current_round'] >= self.improv_state['max_rounds']:
            self.improv_state['phase'] = 'done'
            # produce closing summary
            summary = self._closing_summary()
            return {"reaction": reaction, "summary": summary}

        self.improv_state['phase'] = 'awaiting_improv'

        return {"reaction": reaction, "ended": False}

    @function_tool
    async def reset_game(self):
        """Clear the improv state so a fresh session can begin."""
        self.improv_state = {
            "player_name": None,
            "current_round": 0,
            "max_rounds": 3,
            "rounds": [],
            "phase": "intro",
        }
        return {"message": "State cleared. Ready for a fresh game."}

    def _closing_summary(self) -> str:
        # Analyze rounds to create a short summary
        rounds = self.improv_state.get('rounds', [])
        if not rounds:
            return "Nice try — you didn't play any rounds."

        strengths = []
        for r in rounds:
            perf = (r.get('performance') or "").lower()
            if any(w in perf for w in ["i", "me", "my"]):
                strengths.append("personal choices")
            if any(w in perf for w in ["loud", "yell", "shout"]):
                strengths.append("vocal energy")

        strengths = list(dict.fromkeys(strengths))[:3]
        if not strengths:
            strengths_text = "character choices and playful risks"
        else:
            strengths_text = ", ".join(strengths)

        standout_lines = []
        for r in rounds:
            p = r.get('performance') or ""
            if p:
                first = p.strip().split('.')
                if first:
                    standout_lines.append(first[0].strip())

        standout = standout_lines[0] if standout_lines else "a few memorable moments"

        return (
            f"That's a wrap! {self.improv_state.get('player_name','Contestant')} seemed to favor {strengths_text}. "
            f"Standout moment: '{standout}'. Thanks for playing Improv Battle — hope to see you on stage again!"
        )


async def prewarm(proc: JobProcess):
    """Prewarm the model and resources before handling requests"""
    await proc.userdata  # Wait for user data to be ready


async def entrypoint(ctx: JobContext):
    """Main entry point for the agent"""
    logger.info("Starting Improv Battle host agent")

    # Create session
    session = AgentSession(
        llm=google.llm.LLM(model="gemini-2.5-flash"),
        stt=deepgram.STT(model="nova-3"),
        tts=murf.TTS(voice="en-US-matthew", model="FALCON"),
    )

    # Track usage metrics
    usage_collector = metrics.UsageCollector()

    @session.on(MetricsCollectedEvent)
    def _on_metrics_collected(event: MetricsCollectedEvent):
        metrics.log_metrics(event.metrics)
        usage_collector.collect(event.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Instantiate and start the session with Improv host agent
    agent = ImprovHostAgent()

    # Attempt to seed the agent's player name from the room configuration
    # (the connection token may include a `roomConfig` with agents info).
    try:
        detected_name = None
        room_obj = getattr(ctx, 'room', None)
        if room_obj is not None:
            # Try a few common attribute names that may contain the room config
            for attr in ('room_config', 'roomConfig', 'config', 'roomConfigJson'):
                conf = getattr(room_obj, attr, None)
                if conf:
                    try:
                        # conf may already be a dict-like object
                        if isinstance(conf, dict):
                            agents = conf.get('agents')
                        else:
                            # some runtimes provide an object/string; try to coerce to json
                            agents = None
                            try:
                                parsed = json.loads(str(conf))
                                agents = parsed.get('agents')
                            except Exception:
                                agents = None

                        if agents and isinstance(agents, (list, tuple)) and len(agents) > 0:
                            a0 = agents[0]
                            if isinstance(a0, dict):
                                detected_name = a0.get('agent_name') or a0.get('agentName')
                            else:
                                # try to coerce
                                try:
                                    a0j = json.loads(str(a0))
                                    detected_name = a0j.get('agent_name') or a0j.get('agentName')
                                except Exception:
                                    detected_name = None
                            if detected_name:
                                break
                    except Exception:
                        # keep trying other attrs
                        continue

        if detected_name:
            logger.info('Seeding improv player_name from room config: %s', detected_name)
            # call start_game synchronously (no TTS) to set the state so any auto-greeting uses the name
            try:
                await agent.start_game(player_name=detected_name)
            except Exception:
                logger.exception('Failed to seed agent state with detected player name')
    except Exception:
        logger.exception('Error while attempting to detect agentName from ctx.room')

    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Listen for data messages on the room to trigger game start.
    # The frontend publishes a JSON payload on topic 'improv' with {type:'improv_start', player_name: '...' }
    def _handle_data_event(*args, **kwargs):
        # run in background asyncio task
        async def _process():
            try:
                logger.info(f"data event args={args} kwargs={kwargs}")

                # attempt to find a payload string in args or kwargs (many SDKs differ)
                payload_text = None
                topic = None

                def _extract_from_obj(obj):
                    # tries multiple ways to find payload text on an object
                    if obj is None:
                        return None, None
                    # raw bytes
                    if isinstance(obj, (bytes, bytearray)):
                        try:
                            return obj.decode('utf-8'), None
                        except Exception:
                            return None, None
                    # raw string
                    if isinstance(obj, str):
                        if obj.strip().startswith('{'):
                            return obj, None
                        return None, None
                    # dict-like
                    if isinstance(obj, dict):
                        # look for common keys
                        for k in ('data', 'payload', 'body', 'message'):
                            if k in obj:
                                v = obj[k]
                                if isinstance(v, (bytes, bytearray)):
                                    try:
                                        return v.decode('utf-8'), obj.get('topic') or obj.get('topicName')
                                    except Exception:
                                        return None, None
                                if isinstance(v, str) and v.strip().startswith('{'):
                                    return v, obj.get('topic') or obj.get('topicName')
                        return None, None
                    # object with attributes
                    for attr in ('data', 'payload', 'body', 'message'):
                        if hasattr(obj, attr):
                            v = getattr(obj, attr)
                            if isinstance(v, (bytes, bytearray)):
                                try:
                                    return v.decode('utf-8'), getattr(obj, 'topic', None)
                                except Exception:
                                    return None, None
                            if isinstance(v, str) and v.strip().startswith('{'):
                                return v, getattr(obj, 'topic', None)
                    return None, None

                # scan positional args first
                for a in args:
                    txt, t = _extract_from_obj(a)
                    if txt:
                        payload_text = txt
                        topic = topic or t
                        break

                # if not found, scan kwargs values
                if not payload_text:
                    for v in kwargs.values():
                        txt, t = _extract_from_obj(v)
                        if txt:
                            payload_text = txt
                            topic = topic or t
                            break

                if not payload_text:
                    # last resort: log the reprs to aid debugging
                    try:
                        args_repr = [repr(a) for a in args]
                        kwargs_repr = {k: repr(v) for k, v in kwargs.items()}
                    except Exception:
                        args_repr = str(args)
                        kwargs_repr = str(kwargs)
                    logger.debug('No payload text found in data event. args=%s kwargs=%s', args_repr, kwargs_repr)
                    return

                try:
                    payload = json.loads(payload_text)
                except Exception:
                    logger.debug('data payload not json: %s', payload_text)
                    return

                if payload.get('type') != 'improv_start':
                    # handle end message as well
                    if payload.get('type') == 'improv_end':
                        try:
                            await agent.reset_game()
                            logger.info('Received improv_end: cleared agent state')
                        except Exception:
                            logger.exception('Failed to reset agent state on improv_end')
                    return

                player_name = payload.get('player_name') or payload.get('name')

                # call agent tools to initialize and start the first scenario
                try:
                    # set player name and get intro message
                    intro = await agent.start_game(player_name=player_name)
                    # ask the agent session to generate a reply (speak the intro)
                    try:
                        await agent.session.generate_reply(user_input=intro.get('message'), tool_choice='none')
                    except Exception as e:
                        logger.exception('Failed to generate intro reply: %s', e)

                    # propose the first scenario and speak it
                    scenario_resp = await agent.propose_scenario()
                    try:
                        await agent.session.generate_reply(user_input=scenario_resp.get('message'), tool_choice='none')
                    except Exception as e:
                        logger.exception('Failed to generate scenario reply: %s', e)
                except Exception:
                    logger.exception('Failed to start improv game from data message')

            except Exception:
                logger.exception('Error processing data event')

        import asyncio

        asyncio.create_task(_process())

    # register for data events (some runtimes use 'data_received')
    try:
        ctx.room.on('data_received', _handle_data_event)
    except Exception:
        try:
            ctx.room.on('data', _handle_data_event)
        except Exception:
            logger.warning('Unable to register data event handler on room')

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))