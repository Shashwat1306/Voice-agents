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


class FoodOrderingAgent(Agent):
    """Food and grocery ordering assistant for FreshCart Express"""
    
    def __init__(self) -> None:
        # Load catalog and recipes when agent initializes
        self.catalog = load_catalog()
        self.recipes = load_recipes()
        
        # Build catalog summary for instructions
        total_items = sum(len(items) for items in self.catalog['categories'].values())
        categories = list(self.catalog['categories'].keys())
        
        super().__init__(
            instructions=f"""You are a friendly food and grocery ordering assistant for {self.catalog['store_name']}.

YOUR ROLE:
You help customers order food and groceries quickly and easily. You have access to a catalog with {total_items} items across categories: {', '.join(categories)}.

GREETING:
- Welcome the customer warmly
- Introduce yourself: "Welcome to {self.catalog['store_name']}! I can help you order groceries, snacks, prepared food, and beverages."
- Ask what they'd like to order today

ORDERING CAPABILITIES:

1. **Specific Items** (FASTEST - Use This): 
   - Customer says: "I want bread" or "Add milk to cart"
   - Directly use add_to_cart with the item name (e.g., "bread", "milk")
   - The tool will handle searching automatically
   - Only use search_catalog if customer asks "What do you have?" or wants to browse
   - Default to quantity 1 if not specified

2. **Ingredients for Dishes** (SMART FEATURE):
   - Customer says: "I need ingredients for a peanut butter sandwich" or "Get me pasta ingredients"
   - Use get_recipe_ingredients tool with the dish name
   - This automatically adds ALL required ingredients to cart in one call
   - Confirm what was added

3. **Cart Management**:
   - Use view_cart only when customer explicitly asks "What's in my cart?"
   - Use remove_from_cart to remove items
   - Use update_cart_quantity to change quantities

4. **Order Completion**:
   - When customer says "That's all", "Place order", "Checkout"
   - Use place_order tool immediately (it will show cart and total)
   - Confirm order placed

CONVERSATION STYLE:
- Be quick and efficient - avoid unnecessary tool calls
- Respond immediately after tool results
- Keep responses concise
- Ask for clarification only when truly needed

IMPORTANT RULES:
- For "add X" requests: Use add_to_cart DIRECTLY (fastest)
- For "ingredients for X": Use get_recipe_ingredients DIRECTLY
- DON'T call search_catalog unless customer wants to browse
- DON'T call view_cart after every add operation
- Keep the conversation moving fast""",
        )
        
        # Initialize cart
        self.cart = {}  # Format: {item_id: {"item": item_data, "quantity": int}}
        self.customer_name = None

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
        agent=FoodOrderingAgent(),
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