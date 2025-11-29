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

# Load catalog and recipes for food ordering
def load_catalog():
    """Load the food catalog from JSON file"""
    content_path = Path("shared-data/catalog.json")
    with open(content_path, "r") as f:
        return json.load(f)

def load_recipes():
    """Load the recipes mapping from JSON file"""
    content_path = Path("shared-data/recipes.json")
    with open(content_path, "r") as f:
        return json.load(f)


class GameMasterAgent(Agent):
    """A single-player Game Master (GM) for an interactive Cyberpunk 2077-style adventure.

    The GM speaks in a cinematic, noir-cyberpunk tone, describes scenes, and prompts the player
    to respond by voice. The GM remembers the player's past decisions, named characters,
    and locations to maintain continuity across the session.
    """

    def __init__(self) -> None:
        # Memory structures the GM will maintain
        self.memory = {
            "player_name": None,
            "decisions": [],
            "named_characters": {},
            "locations": {},
            "notes": [],
        }

        # Path to write a simple transcript / history so the frontend or logs can display it
        self.history_path = Path("shared-data/gameplay_log.json")
        self.history_path.parent.mkdir(parents=True, exist_ok=True)

        instructions = (
            "Universe: Cyberpunk 2077. "
            "Tone: dramatic, noir, cinematic with occasional dry humor. "
            "Role: You are the GM. You describe scenes and ask the player what they do. "
            "Drive an interactive story using voice. Maintain continuity with the conversation history. "
            "End every response with a short, direct prompt for player action: 'What do you do?'.\n\n"
            "Behavior and constraints:\n"
            "- Always present vivid sensory details (visual, audio, tactile) appropriate to a cyberpunk city.\n"
            "- Keep the player's agency central: after each scene description, ask a single concise question prompting choice.\n"
            "- Remember and reference the player's past decisions, names, and locations (store in memory).\n"
            "- When the player speaks, transcribe and log the player's utterance so it is visible in the session transcript.\n"
            "- Provide short stateless checks (e.g., success/failure) based on clear choices — keep mechanics simple and narrative-first.\n"
            "- Never break character as the GM. Maintain immersion.\n\n"
            "Start the session by introducing the setting, the player's apparent situation, and one immediate problem to solve. Use no longer than 3 short paragraphs for scene setup. End with 'What do you do?'")

        super().__init__(instructions=instructions)

    async def _append_history(self, role: str, text: str):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "role": role,
            "text": text,
        }
        try:
            if self.history_path.exists():
                with open(self.history_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = []
            data.append(entry)
            with open(self.history_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            logger.exception("Failed to append gameplay history")

    @function_tool
    async def log_player_speech(self, transcript: Annotated[str, "The player's transcribed speech"]) -> str:
        """Log the player's transcribed speech so it can be shown in UI and the GM can reference it."""
        logger.info(f"Player said: {transcript}")
        await self._append_history("player", transcript)
        # Store an indexed note of last action
        self.memory["decisions"].append({"type": "utterance", "text": transcript, "time": datetime.now().isoformat()})
        return "Logged player speech."

    @function_tool
    async def log_gm_message(self, message: Annotated[str, "The GM's message to the player"]) -> str:
        """Record a GM message in the transcript so a UI can present both sides of the conversation."""
        logger.info(f"GM: {message}")
        await self._append_history("gm", message)
        return "Logged GM message."

    @function_tool
    async def get_game_state(self) -> str:
        """Return a concise summary of what the GM remembers about the session."""
        state = {
            "player_name": self.memory.get("player_name"),
            "decisions_count": len(self.memory.get("decisions", [])),
            "named_characters": list(self.memory.get("named_characters", {}).keys()),
            "locations_known": list(self.memory.get("locations", {}).keys()),
        }
        return json.dumps(state)

    @function_tool
    async def reset_game(self) -> str:
        """Reset the current session memory (keeps transcript file)."""
        self.memory = {"player_name": None, "decisions": [], "named_characters": {}, "locations": {}, "notes": []}
        return "Game memory reset. The transcript file remains for review." 

    async def on_enter(self) -> None:
        """Called when the session starts — ask the player's name and character type.

        This prompt is deliberately short and actionable so the frontend receives audio
        immediately and the player can respond by voice. It also logs the GM's prompt
        to the session transcript for display in the UI.
        """

        prompt = (
            "Night City hums and bleeds neon. Before we dive in, tell me your name, "
            "and choose what type of character you'd like to be in this world. "
            "You can pick one of the options below or describe your own.\n\n"
            "1) Netrunner — a shadowy hacker who bends the Net and breaches corporate systems.\n"
            "2) Solo — a hardened combat specialist, a walking arsenal trained for violence.\n"
            "3) Techie — a hardware and software engineer who rigs, mods, and invents.\n"
            "4) Fixer — a well-connected broker who arranges jobs, favors, and black-market deals.\n"
            "5) Nomad — a wanderer from the Badlands, skilled in survival and driving.\n\n"
            "Tell me your name and pick a type (for example: 'My name is Alex, I'm a Netrunner'). What do you do?"
        )

        try:
            await self._append_history("gm", prompt)
        except Exception:
            logger.exception("failed to write initial gm history")

        try:
            handle = self.session.say(prompt)
            # wait for playout so the session shows the agent as speaking
            await handle
        except Exception:
            logger.exception("failed to play initial GM prompt")

    @function_tool
    async def search_catalog(
        self, 
        item_name: Annotated[str, "The name of the item to search for (e.g., 'bread', 'milk', 'chips')"]
    ):
        """Search the catalog for items matching the customer's query"""
        logger.info(f"Searching catalog for: {item_name}")
        
        search_term = item_name.lower()
        results = []
        
        # Search across all categories
        for category, items in self.catalog['categories'].items():
            for item in items:
                if search_term in item['name'].lower() or any(search_term in tag.lower() for tag in item['tags']):
                    results.append({
                        "id": item['id'],
                        "name": item['name'],
                        "price": item['price'],
                        "category": category,
                        "brand": item.get('brand', 'N/A'),
                        "unit": item['unit']
                    })
        
        if not results:
            logger.info(f"No items found for '{item_name}'")
            return f"Sorry, I couldn't find any items matching '{item_name}'. Could you try describing it differently or browse our categories: {', '.join(self.catalog['categories'].keys())}?"
        
        if len(results) == 1:
            item = results[0]
            return f"I found: {item['name']} ({item['brand']}) - ${item['price']} per {item['unit']}. Item ID: {item['id']}. Would you like to add this to your cart?"
        
        # Multiple results
        result_list = "\n".join([f"- {r['name']} ({r['brand']}) - ${r['price']} per {r['unit']} [ID: {r['id']}]" for r in results[:5]])
        return f"I found {len(results)} items:\n{result_list}\n\nWhich one would you like? You can tell me the item ID or name."

    @function_tool
    async def add_to_cart(
        self,
        item_id: Annotated[str, "The ID of the item to add (e.g., 'GR001', 'SN003') OR the item name (e.g., 'bread', 'milk')"],
        quantity: Annotated[int, "The quantity to add (default 1)"] = 1
    ):
        """Add an item to the shopping cart by item ID or name"""
        logger.info(f"Adding {quantity}x {item_id} to cart")
        
        # First, try to find by ID
        item_found = None
        actual_item_id = item_id
        
        for category, items in self.catalog['categories'].items():
            for item in items:
                if item['id'] == item_id:
                    item_found = item
                    actual_item_id = item['id']
                    break
            if item_found:
                break
        
        # If not found by ID, try to find by name
        if not item_found:
            search_term = item_id.lower()
            matches = []
            
            for category, items in self.catalog['categories'].items():
                for item in items:
                    if search_term in item['name'].lower():
                        matches.append(item)
            
            if len(matches) == 1:
                item_found = matches[0]
                actual_item_id = item_found['id']
            elif len(matches) > 1:
                match_list = "\n".join([f"- {m['name']} ({m['brand']}) - ${m['price']} [ID: {m['id']}]" for m in matches[:5]])
                return f"I found multiple items matching '{item_id}':\n{match_list}\n\nPlease specify which one by using its ID or being more specific with the name."
            else:
                return f"Error: Item '{item_id}' not found in catalog. Try using search_catalog first to find the exact item."
        
        # Add to cart
        if actual_item_id in self.cart:
            self.cart[actual_item_id]['quantity'] += quantity
            new_qty = self.cart[actual_item_id]['quantity']
            return f"Updated! You now have {new_qty} {item_found['name']} in your cart."
        else:
            self.cart[actual_item_id] = {
                "item": item_found,
                "quantity": quantity
            }
            return f"Added {quantity} {item_found['name']} to your cart (${item_found['price']} per {item_found['unit']})."

    @function_tool
    async def get_recipe_ingredients(
        self,
        dish_name: Annotated[str, "The name of the dish to get ingredients for (e.g., 'peanut butter sandwich', 'pasta')"]
    ):
        """Get all ingredients for a specific dish and add them to cart automatically"""
        logger.info(f"Looking up recipe for: {dish_name}")
        
        search_term = dish_name.lower()
        
        # Search recipes - recipes is a dictionary with recipe names as keys
        recipe_found = None
        for recipe_key, recipe_data in self.recipes['recipes'].items():
            if search_term in recipe_key.lower() or search_term in recipe_data['name'].lower():
                recipe_found = recipe_data
                break
        
        if not recipe_found:
            return f"Sorry, I don't have a recipe for '{dish_name}'. I can help you order individual items though!"
        
        # Add all ingredients to cart
        added_items = []
        for ingredient_id in recipe_found['ingredients']:
            # Find item in catalog
            for category, items in self.catalog['categories'].items():
                for item in items:
                    if item['id'] == ingredient_id:
                        # Add to cart
                        if ingredient_id in self.cart:
                            self.cart[ingredient_id]['quantity'] += 1
                        else:
                            self.cart[ingredient_id] = {
                                "item": item,
                                "quantity": 1
                            }
                        added_items.append(item['name'])
                        break
        
        logger.info(f"Added {len(added_items)} ingredients for {recipe_found['name']}")
        items_list = ", ".join(added_items)
        return f"Great! I've added all the ingredients for {recipe_found['name']}: {items_list}. Anything else you need?"

    @function_tool
    async def view_cart(self):
        """View all items currently in the shopping cart"""
        logger.info("Viewing cart")
        
        if not self.cart:
            return "Your cart is empty. What would you like to order?"
        
        cart_items = []
        total = 0.0
        
        for item_id, cart_data in self.cart.items():
            item = cart_data['item']
            qty = cart_data['quantity']
            price = float(item['price'])
            subtotal = price * qty
            total += subtotal
            
            cart_items.append(f"- {item['name']} x {qty} = ${subtotal:.2f}")
        
        items_text = "\n".join(cart_items)
        return f"""Your cart:\n{items_text}\n\nTotal: ${total:.2f}\n\nReady to place your order?"""

    @function_tool
    async def update_cart_quantity(
        self,
        item_id: Annotated[str, "The ID of the item to update"],
        new_quantity: Annotated[int, "The new quantity (must be > 0)"]
    ):
        """Update the quantity of an item in the cart"""
        logger.info(f"Updating {item_id} quantity to {new_quantity}")
        
        if item_id not in self.cart:
            return f"Error: {item_id} is not in your cart."
        
        if new_quantity <= 0:
            return "Quantity must be greater than 0. Use remove_from_cart to delete items."
        
        item_name = self.cart[item_id]['item']['name']
        self.cart[item_id]['quantity'] = new_quantity
        
        return f"Updated {item_name} quantity to {new_quantity}."

    @function_tool
    async def remove_from_cart(
        self,
        item_id: Annotated[str, "The ID of the item to remove"]
    ):
        """Remove an item from the shopping cart"""
        logger.info(f"Removing {item_id} from cart")
        
        if item_id not in self.cart:
            return f"Error: {item_id} is not in your cart."
        
        item_name = self.cart[item_id]['item']['name']
        del self.cart[item_id]
        
        return f"Removed {item_name} from your cart."

    @function_tool
    async def place_order(self):
        """Place the order and save it to the orders directory"""
        logger.info("Placing order")
        
        if not self.cart:
            return "Your cart is empty! Add some items first."
        
        # Calculate total
        total = sum(
            float(cart_data['item']['price']) * cart_data['quantity']
            for cart_data in self.cart.values()
        )
        
        # Create order object
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        order = {
            "order_id": f"ORDER_{timestamp}",
            "timestamp": datetime.now().isoformat(),
            "customer_name": self.customer_name or "Guest",
            "items": [
                {
                    "id": cart_data['item']['id'],
                    "name": cart_data['item']['name'],
                    "price": cart_data['item']['price'],
                    "quantity": cart_data['quantity'],
                    "subtotal": float(cart_data['item']['price']) * cart_data['quantity']
                }
                for cart_data in self.cart.values()
            ],
            "total": total,
            "status": "placed"
        }
        
        # Save order (ensure directory exists)
        order_path = Path(f"food-orders/order_{timestamp}.json")
        try:
            order_path.parent.mkdir(parents=True, exist_ok=True)
            with open(order_path, "w", encoding="utf-8") as f:
                json.dump(order, f, indent=2)
            logger.info(f"Order placed: {order['order_id']}")
        except Exception as e:
            logger.exception("Failed to save order")
            return "Sorry — an internal error occurred while placing your order. Please try again in a moment."
        
        # Clear cart
        item_count = len(self.cart)
        self.cart = {}
        
        return f"""Perfect! Your order has been placed successfully!

Order ID: {order['order_id']}
Total: ${total:.2f}
Items: {item_count}

Your order will be delivered within 30-45 minutes. Thank you for shopping with {self.catalog['store_name']}!"""


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
        agent=GameMasterAgent(),
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