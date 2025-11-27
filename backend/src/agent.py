import logging
import json
from pathlib import Path
from typing import Annotated
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

# Load fraud cases database
def load_fraud_database():
    """Load the fraud cases database from JSON file"""
    content_path = Path("shared-data/fraud_cases_database.json")
    with open(content_path, "r") as f:
        return json.load(f)

def save_fraud_database(database):
    """Save the updated fraud cases database to JSON file"""
    content_path = Path("shared-data/fraud_cases_database.json")
    with open(content_path, "w") as f:
        json.dump(database, f, indent=2)

FRAUD_DATABASE = load_fraud_database()


class FraudAlertAgent(Agent):
    """Fraud detection voice agent for bank security department"""
    
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a professional fraud detection representative from National-Bank India's Fraud Prevention Department.

YOUR ROLE AND MISSION:
You are calling customers about suspicious transactions detected on their accounts. Your goal is to verify the transaction and protect the customer from potential fraud.

CALL FLOW:
1. **Introduction**: 
   - Greet the customer warmly
   - Introduce yourself: "This is the Fraud Prevention Department from SecureBank India"
   - Explain you're calling about a suspicious transaction on their account
   - Keep a calm, reassuring, professional tone

2. **Get Customer Name**:
   - Ask: "May I have your full name please?"
   - Use the get_fraud_case tool with their name to load their case
   - If no case found, apologize and say there may have been an error

3. **Identity Verification**:
   - Once you have their case, ask them the security question from their profile
   - DO NOT ask for card numbers, PINs, passwords, or any sensitive banking credentials
   - Only use the security question stored in their case
   - Use verify_security_answer tool to check their answer
   - If they answer correctly, proceed with the call
   - If they answer incorrectly after 2 attempts, politely end the call using mark_verification_failed

4. **Read Transaction Details**:
   - Once verified, you can use read_transaction_details tool OR read the details from the case directly
   - Read out the suspicious transaction details clearly:
     - Merchant name
     - Amount
     - Date and time
     - Location
     - Card ending in [last 4 digits]
   - Example: "We detected a transaction of $1,247.50 at ABC Electronics Industry on alibaba.com from Shanghai, China on November 26th at 3:42 PM using your card ending in 4242."

5. **Confirm Transaction**:
   - Ask clearly: "Did you make or authorize this transaction?"
   - Listen for their response (yes/no)
   - If YES → Use mark_case_safe tool
   - If NO → Use mark_case_fraudulent tool

6. **Closing**:
   - Thank them for their time
   - Summarize the action taken
   - Reassure them their account is secure
   - End the call professionally

IMPORTANT GUIDELINES:
- Never ask for full card numbers, CVV, PIN, or passwords
- Stay calm and professional even if the customer is worried
- Be patient and empathetic
- Speak clearly when reading transaction details
- Confirm their answers before taking action
- Use function tools to load cases and update status

Remember: You're here to protect the customer from fraud while providing excellent service.""",
        )
        
        # Initialize fraud case data
        self.current_case = None
        self.verification_attempts = 0
        self.max_verification_attempts = 2
        self.is_verified = False

    @function_tool
    async def get_fraud_case(
        self, 
        customer_name: Annotated[str, "The customer's full name to look up their fraud case"],
        context: RunContext
    ):
        """Load the fraud case for a specific customer by their name"""
        logger.info(f"Looking up fraud case for: {customer_name}")
        
        # Search for the customer in the database
        for case in FRAUD_DATABASE['fraud_cases']:
            if case['userName'].lower() == customer_name.lower():
                self.current_case = case
                logger.info(f"Found case for {customer_name}: {case['transactionName']}")
                return f"Case loaded for {customer_name}. Security identifier: {case['securityIdentifier']}. Now proceed with identity verification by asking their security question: '{case['securityQuestion']}'"
        
        logger.warning(f"No fraud case found for: {customer_name}")
        return f"I apologize, but I cannot find a fraud case for {customer_name} in our system. There may have been an error. Please contact our fraud department directly."

    @function_tool
    async def verify_security_answer(
        self, 
        customer_answer: Annotated[str, "The customer's answer to their security question"],
        context: RunContext
    ):
        """Verify the customer's identity using their security answer"""
        if not self.current_case:
            return "Error: No case loaded. Please get the customer's name first."
        
        self.verification_attempts += 1
        correct_answer = self.current_case['securityAnswer'].lower()
        provided_answer = customer_answer.lower()
        
        logger.info(f"Verification attempt {self.verification_attempts}/{self.max_verification_attempts}")
        
        if provided_answer == correct_answer:
            self.is_verified = True
            logger.info(f"Verification successful for {self.current_case['userName']}")
            return f"Thank you for verifying your identity. Now I need to inform you about the suspicious transaction we detected."
        else:
            if self.verification_attempts >= self.max_verification_attempts:
                logger.warning(f"Verification failed after {self.verification_attempts} attempts")
                return f"I'm sorry, but I cannot verify your identity. For your security, I cannot proceed with this call. Please visit your nearest branch or call our main customer service line. Goodbye."
            else:
                remaining = self.max_verification_attempts - self.verification_attempts
                return f"I'm sorry, that answer doesn't match our records. You have {remaining} more attempt(s). Let me ask the security question again."

    @function_tool
    async def mark_verification_failed(self, context: RunContext):
        """Mark the case as verification failed and end the call"""
        if not self.current_case:
            return "Error: No case loaded."
        
        # Update the database
        for case in FRAUD_DATABASE['fraud_cases']:
            if case['securityIdentifier'] == self.current_case['securityIdentifier']:
                case['status'] = 'verification_failed'
                case['outcome_note'] = 'Customer failed identity verification. Call terminated for security.'
                break
        
        save_fraud_database(FRAUD_DATABASE)
        logger.info(f"Case marked as verification_failed for {self.current_case['userName']}")
        
        return "Case updated. Thank you for calling. Goodbye."

    @function_tool
    async def mark_case_safe(self, context: RunContext):
        """Mark the fraud case as safe - customer confirmed they made the transaction"""
        if not self.current_case:
            return "Error: No case loaded."
        
        if not self.is_verified:
            return "Error: Customer must be verified first before updating case status."
        
        # Update the database
        for case in FRAUD_DATABASE['fraud_cases']:
            if case['securityIdentifier'] == self.current_case['securityIdentifier']:
                case['status'] = 'confirmed_safe'
                case['case'] = 'safe'
                case['outcome_note'] = f'Customer {case["userName"]} confirmed they authorized this transaction. No fraud detected.'
                break
        
        save_fraud_database(FRAUD_DATABASE)
        logger.info(f"Case marked as SAFE for {self.current_case['userName']}")
        
        transaction_details = f"{self.current_case['transactionAmount']} at {self.current_case['transactionName']}"
        
        return f"""Perfect! I've updated our records to show that the transaction of {transaction_details} was authorized by you. 

Your account remains secure and no further action is needed. Thank you for confirming this with us - we take your security very seriously.

Is there anything else I can help you with today?"""

    @function_tool
    async def mark_case_fraudulent(self, context: RunContext):
        """Mark the fraud case as fraudulent - customer did not authorize the transaction"""
        if not self.current_case:
            return "Error: No case loaded."
        
        if not self.is_verified:
            return "Error: Customer must be verified first before updating case status."
        
        # Update the database
        for case in FRAUD_DATABASE['fraud_cases']:
            if case['securityIdentifier'] == self.current_case['securityIdentifier']:
                case['status'] = 'confirmed_fraud'
                case['case'] = 'fraudulent'
                case['outcome_note'] = f'Customer {case["userName"]} denied making this transaction. Fraud confirmed. Card blocked and dispute initiated.'
                break
        
        save_fraud_database(FRAUD_DATABASE)
        logger.info(f"Case marked as FRAUDULENT for {self.current_case['userName']}")
        
        card_ending = self.current_case['cardEnding']
        transaction_details = f"{self.current_case['transactionAmount']} at {self.current_case['transactionName']}"
        
        return f"""I understand. Thank you for letting us know. For your protection, I have immediately taken the following actions:

1. Blocked your card ending in {card_ending}
2. Initiated a fraud dispute for the {transaction_details} transaction
3. Flagged this transaction for investigation
4. Prevented any further charges on this card

You will receive a new card at your registered address within 5-7 business days. We will also reverse the fraudulent charge within 7-10 business days.

Please monitor your account for any other suspicious activity and contact us immediately if you notice anything unusual. 

Your account security is our top priority. Is there anything else you need help with?"""

    @function_tool
    async def read_transaction_details(self, context: RunContext):
        """Read out the suspicious transaction details to the customer"""
        if not self.current_case:
            return "Error: No case loaded. Please get the customer's name first."
        
        if not self.is_verified:
            return "Error: Customer must be verified first before reading transaction details."
        
        case = self.current_case
        details = f"""We detected a suspicious transaction on your account. Here are the details:

- Amount: {case['transactionAmount']}
- Merchant: {case['transactionName']}
- Website: {case['transactionSource']}
- Location: {case['transactionLocation']}
- Date and Time: {case['transactionTime']}
- Card used: ending in {case['cardEnding']}
- Category: {case['transactionCategory']}

Did you make or authorize this transaction?"""
        
        logger.info(f"Read transaction details to {case['userName']}")
        return details


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
        agent=FraudAlertAgent(),
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